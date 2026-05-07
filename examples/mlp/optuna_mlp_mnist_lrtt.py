# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for MLP + MNIST with LRTT.

Usage:
    python optuna_mlp_mnist_lrtt.py --n-trials 50
    python optuna_mlp_mnist_lrtt.py --visualize
    python optuna_mlp_mnist_lrtt.py --n-trials 50 --optimizer AnalogSGD --reinit-mode decay --no-wd --no-momentum --no-nesterov --no-io-noise --no-scale-transfer-lr --c-device linearstepideal
    python optuna_mlp_mnist_lrtt.py --n-trials 50 --optimizer AnalogSGD --reinit-mode gauss_b_zero --no-io-noise --no-scale-transfer-lr --c-device linearstepideal


All flags:
    python optuna_mlp_mnist_lrtt.py \
        --study-name <str>          # Study name (default: auto-generated)
        --n-trials <int>            # Number of Optuna trials (default: 50)
        --visualize                 # Visualize study results and exit
        --optimizer <str>           # AnalogSGD | AnalogAdam (default: AnalogAdam)
        --no-wd                     # Disable weight decay tuning (fix to 0)
        --no-momentum               # Disable momentum tuning (fix to 0, SGD only)
        --no-nesterov               # Disable nesterov tuning (fix to False, SGD only)
        --reinit-mode <str>         # Fix reinit mode: standard | decay | hybrid |
                                    #   orthogonal_zero | orthogonal_decay |
                                    #   gauss_b_zero | gauss_b_decay |
                                    #   gauss_a_zero | gauss_a_decay |
                                    #   selector_b_zero | selector_b_decay |
                                    #   selector_a_zero | selector_a_decay
                                    #   (default: tune among standard/decay/hybrid)
        --batch-size <int>          # Batch size (default: 128)
        --grad-accum-steps <int>    # Gradient accumulation steps (default: 1)
        --epochs <int>              # Number of epochs (default: 10)
        --step-lr-size <int>        # StepLR step size in epochs (default: 10)
        --step-lr-gamma <float>     # StepLR gamma (default: 0.5)
        --transfer-method <str>     # Transfer method: onehot | direct | set (default: onehot)
        --ab-device <str>           # A/B tile device: 6t1c | linearstep | linearstepideal | constantstep | constantstepideal | constantstep6t1cgamma | fp | ideal (default: 6t1c)
        --c-device <str>            # C tile device: softboundsideal | linearstepideal | constantstep | constantstepideal | constantstep6t1cgamma | ideal (default: softboundsideal)
        --no-io-noise               # Disable IO out_noise (resolution kept)
        --forward-inject            # Enable forward noise injection
        --is-perfect                # Use ideal FP forward/backward (no ADC/DAC/noise)
        --no-quant                  # Disable DAC/ADC quantization (inp_res/out_res)
        --lora-target <str>         # LoRA target: none | linear1 (default: linear1)
        --head-layer <str>          # Output head (classifier): train | freeze (default: train)
        --no-transfer               # Disable LRTT transfer (A/B frozen, skip LRTT param sweep)
        --no-adc-ab-proj            # Remove ADC/DAC between A/B projections (full precision)
        --no-learn-out-scaling      # Disable trainable out_scaling on C tile
        --encoder-analog            # Non-LRTT linear layers: frozen analog instead of digital
        --head-analog               # Convert classifier to frozen analog (default: digital)
        --backward-out-bound <float> # Backward pass output bound (default: 12.0)
        --auto-scale-mode <str>     # Auto-scale: none | shared | separate (default: none)
        --correct-gradient-magnitudes  # Correct transfer magnitude by dividing by effective A/B LR
        --transfer-rank-schedule <str>  # Transfer rank schedule: all | round_robin (default: all)
        --transfer-ranks-per-step <int> # Ranks per transfer step in round_robin mode (default: 1)
        --no-scale-transfer-lr          # Disable scaling transfer LR by SGD LR
        --ab-multilevel                 # Sweep ab_multilevel (1-12); w_max-w_min = 2^multilevel * ab_dw_min, B init scales
        --fi-continuous-alpha           # Use transfer LR as forward-injection α (continuity)
        --ab-pulse-type <str>           # A/B pulse type: default | none | none_with_device | stochastic_compressed | mean_count | deterministic_implicit


Inline flags (edit directly in script):
    DYNAMIC_TE = False              # Enable dynamic transfer every
    DYNAMIC_TE_POWER = 1.0          # Power for dynamic TE scaling
    TE_WARMUP_STEPS = 0             # Steps before reaching target TE
    TE_WARMUP_SCHEDULE = []         # Warmup TE schedule list
    REINIT_GAIN = 1.0               # Reinitialization gain
    TRAIN_SUBSET_SIZE = 0           # Training data subset (0 = full)
    EVAL_SUBSET_SIZE = 0            # Evaluation data subset (0 = full)
    HIDDEN_DIM = 128                # MLP hidden dimension


Enqueue example:

python3 << 'EOF'
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
study = optuna.load_study(
    study_name='mlp_mnist_lrtt_bs128_sgd_decay_nowd_nomom_nonest_onehot_noio_none',
    storage=JournalStorage(JournalFileBackend(
        'results/optuna_mlp_mnist_lrtt/optuna_mlp_mnist_lrtt_bs128_sgd_decay_nowd_nomom_nonest_onehot_noio_none.log')))
study.enqueue_trial({
    'learning_rate': 0.1,
    'transfer_lr': 0.5,
    'transfer_every': 32,
    'rank_exp': 3,
    'fast_lr': 1.0,
    'tau_sec': 46505.0,
})
print('Enqueued!')
EOF

"""

import os
import sys
import math
import json
import argparse
import gc

import torch
from torch import nn, no_grad, manual_seed
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
import numpy as np

import optuna
from optuna.trial import TrialState
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna_integration import BoTorchSampler
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import aihwkit.optim.lrtt_grad_accum_patch  # noqa: F401  — per-micro-batch tile.update + LRTT A/B snapshot

from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice, FloatingPointDevice, IdealDevice, ConstantStepDevice
from aihwkit.simulator.configs import SingleRPUConfig

# LRTT config imports (direct imports to avoid __init__.py dependency issues)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.parameters.mapping import MappingParameter


# =============================================================================
# ConfigAwareBoTorchSampler with Periodic Exploration
# =============================================================================

class ConfigAwareBoTorchSampler(BoTorchSampler):
    """BoTorchSampler that respects OPT_CONFIG and avoids duplicate running trials."""

    def _is_almost_identical_to_running(self, params, study, threshold=0.01):
        """Check if params are almost identical to any running trial (within 1%)."""
        running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        for rt in running_trials:
            identical = True
            for key, val in params.items():
                if key in rt.params and isinstance(val, (int, float)):
                    rt_val = rt.params[key]
                    if rt_val != 0:
                        if abs(val - rt_val) / abs(rt_val) > threshold:
                            identical = False
                            break
                    elif val != 0:
                        identical = False
                        break
            if identical:
                return True
        return False

    def _add_jitter(self, params, search_space, jitter_ratio=0.1):
        """Add small jitter to params while keeping within search space bounds."""
        import random
        jittered = dict(params)
        for key, val in params.items():
            if key in search_space and isinstance(val, float):
                dist = search_space[key]
                if hasattr(dist, 'low') and hasattr(dist, 'high'):
                    # Add ±10% jitter
                    jitter = val * random.uniform(-jitter_ratio, jitter_ratio)
                    new_val = val + jitter
                    # Clip to bounds
                    new_val = max(dist.low, min(dist.high, new_val))
                    jittered[key] = new_val
        return jittered

    def sample_relative(self, study, trial, search_space):
        params = super().sample_relative(study, trial, search_space)

        # If almost identical to running trial, add jitter to GP params
        if self._is_almost_identical_to_running(params, study):
            params = self._add_jitter(params, search_space)

        # Force reinit_mode if fixed in config
        if OPT_CONFIG['reinit_mode'] is not None and 'reinit_mode' in params:
            params['reinit_mode'] = OPT_CONFIG['reinit_mode']
        # Force optimizer if fixed in config
        if 'optimizer' in params:
            params['optimizer'] = OPT_CONFIG['optimizer']
        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        if param_name == 'reinit_mode' and OPT_CONFIG['reinit_mode'] is not None:
            return OPT_CONFIG['reinit_mode']
        if param_name == 'optimizer':
            return OPT_CONFIG['optimizer']
        return super().sample_independent(study, trial, param_name, param_distribution)


# =============================================================================
# Global Constants
# =============================================================================

DEFAULT_STUDY_NAME = "mlp_mnist_lrtt_main"

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
RESULTS = os.path.join(os.getcwd(), "results", "optuna_mlp_mnist_lrtt")
os.makedirs(RESULTS, exist_ok=True)

# Reproducibility
SEED = 42

# Model
HIDDEN_DIM = 256

# Training defaults
N_EPOCHS = 30
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
EVAL_BATCH_SIZE = 256
EARLY_STOP_PATIENCE = 5
TRAIN_LOSS_EARLY_STOP_PATIENCE = 2  # Stop if train loss doesn't improve for this many epochs
TRAIN_LOSS_THRESHOLD = 0.5  # Once train loss drops below this, rely on metric-based early stop only

# Scheduler (StepLR per epoch — matches MLP v2 MNIST)
STEP_LR_SIZE = 10  # Decay LR every N epochs
STEP_LR_GAMMA = 0.5  # Multiplier per decay

# Dynamic TE
DYNAMIC_TE = False
DYNAMIC_TE_POWER = 1.0
TE_WARMUP_STEPS = 0
TE_WARMUP_SCHEDULE = []

# Fixed LRTT parameters
REINIT_GAIN = 1.0
TRANSFER_METHOD = "onehot"  # "onehot", "direct", or "set"
AB_DEVICE = "6t1c"  # "6t1c", "linearstep", "linearstepideal", "constantstep", "constantstepideal", "constantstep6t1cgamma", "fp", or "ideal"
A_DEVICE = None  # Optional override for A tile device. None → use AB_DEVICE for both A and B (backward compatible).
B_DEVICE = None  # Optional override for B tile device. None → use AB_DEVICE for both A and B (backward compatible).

# Per-parameter shared/split toggle for A and B device parameters.
# False → single ab_<name> sweep variable, applied to both A and B (backward compatible).
# True  → independent a_<name> and b_<name> sweep variables.
#
# Listed here: per-tile splittable parameters that are currently exposed as sweep
# variables. Toggle True to give A and B their own value.
#
# reset_std defaults to True (split): A's reset is the 6T1C inherent floor and B's
# reset_std is the random Gaussian σ source for gauss_b_* modes — these have
# different intended values. Toggle to False to sweep a shared ab_reset_std (e.g.
# for "what if both A and B share the same reset noise" ablation).
#
# Not listed because they're hardcoded (not currently swept): write_noise_std,
# gamma_*, *_dtod, *_std, lifetime_dtod, etc. To split any of those, first expose
# as a sweep variable in objective(), thread through _create_ab_device, then add
# the entry here.
SPLIT_AB_PARAMS = {
    'dw_min': False,
    'multilevel': False,
    'tau_sec': False,
    'reset_std': True,
    'desired_bl': False,
    'weight_scaling_omega': False,
}
C_DEVICE = "softboundsideal"  # "softboundsideal", "linearstepideal", "constantstep", "constantstepideal", "constantstep6t1cgamma", or "ideal"
IO_NOISE = True  # If False, disable out_noise (resolution kept)
FORWARD_INJECT = False  # If True, enable forward noise injection
IS_PERFECT = False  # If True, forward/backward use ideal FP matmul (no ADC/DAC/noise)
AF_RATIO = 1.0  # Asymmetry factor: scales 6T1C gamma_up/gamma_down (used by scaledideal device)
UNR = 1.0       # Update noise ratio: scales 6T1C dw_min_std (used by scaledideal device)
NO_QUANT = False  # If True, disable DAC/ADC quantization (inp_res/out_res → -1)
ENCODER_ANALOG = False  # If True, non-LRTT linear layers become frozen analog instead of digital
HEAD_ANALOG = False  # If True, output head (classifier) → frozen analog instead of digital
BACKWARD_OUT_BOUND = 12.0  # Backward pass output bound (default 12.0)

# LoRA target options: which layers have trainable A/B tiles
# - none: no LRTT layers (fully digital baseline)
# - linear1: only the input projection (classifier output head always digital)
LORA_TARGET = "linear1"  # default, can be set via --lora-target
HEAD_LAYER = "train"  # default, can be set via --head-layer (train | freeze) - output head (classifier)
LORA_TARGET_MODULES = {
    "none": [],              # No LRTT (digital baseline)
    "linear1": ["linear1"],  # Only LRTT-able layer (classifier = output head, always digital)
}

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0

# Global config (set by argparse)
OPT_CONFIG = {
    'optimizer': 'AnalogSGD',
    'tune_wd': False,        # weight_decay = 0 (fixed)
    'tune_momentum': False,  # momentum = 0 (fixed)
    'tune_nesterov': False,  # nesterov = False (fixed)
    'reinit_mode': None,    # None = tune, or 'standard'/'decay'/'hybrid' = fixed
    'no_transfer': False,   # If True, disable transfer (transfer_every = inf)
    'no_adc_ab_proj': False,  # If True, remove ADC/DAC between A/B projections
    'learn_out_scaling': True,  # If True, C tile out_scaling is trainable
    'auto_scale_mode': 'none',
    'correct_gradient_magnitudes': False,
    'transfer_rank_schedule': 'all',
    'transfer_ranks_per_step': 1,
    'scale_transfer_lr': True,
    'fi_continuous_alpha': False,
    'ab_pulse_type': 'default',
    'ab_multilevel': False,  # If True, sweep ab_multilevel; else w_max=1.0 (no rescaling)
}


def get_study_name_suffix():
    """Generate study name suffix based on optimizer config."""
    opt = OPT_CONFIG['optimizer'].lower().replace('analog', '')
    suffix = opt

    # Add reinit_mode if fixed
    if OPT_CONFIG['reinit_mode'] is not None:
        suffix += f"_{OPT_CONFIG['reinit_mode']}"

    if not OPT_CONFIG['tune_wd']:
        suffix += "_nowd"
    if not OPT_CONFIG['tune_momentum']:
        suffix += "_nomom"
    if not OPT_CONFIG['tune_nesterov']:
        suffix += "_nonest"

    # Always add transfer method
    suffix += f"_{TRANSFER_METHOD}"

    # If A and B devices are explicitly split, encode both; otherwise keep the
    # legacy `_{AB_DEVICE}` suffix so existing log files remain comparable.
    if (A_DEVICE is not None or B_DEVICE is not None) and (A_DEVICE != B_DEVICE or (A_DEVICE or B_DEVICE) != AB_DEVICE):
        a_dev = (A_DEVICE if A_DEVICE is not None else AB_DEVICE).replace('-', '')
        b_dev = (B_DEVICE if B_DEVICE is not None else AB_DEVICE).replace('-', '')
        suffix += f"_a{a_dev}_b{b_dev}"
    elif AB_DEVICE != "6t1c":
        suffix += f"_{AB_DEVICE.replace('-', '')}"

    if C_DEVICE != "softboundsideal":
        suffix += f"_c{C_DEVICE}"

    if not IO_NOISE:
        suffix += "_noio"

    if FORWARD_INJECT:
        suffix += "_fwinj"

    if IS_PERFECT:
        suffix += "_perfect"
    if NO_QUANT:
        suffix += "_noquant"

    if OPT_CONFIG['no_transfer']:
        suffix += "_notrans"

    if OPT_CONFIG.get('no_adc_ab_proj', False):
        suffix += "_noadc"

    if not OPT_CONFIG.get('learn_out_scaling', True):
        suffix += "_noos"

    if ENCODER_ANALOG:
        suffix += "_encanalog"

    if HEAD_ANALOG:
        suffix += "_headanalog"

    if BACKWARD_OUT_BOUND != 12.0:
        suffix += f"_bob{BACKWARD_OUT_BOUND:g}"

    if OPT_CONFIG.get('auto_scale_mode', 'none') != 'none':
        suffix += f"_as-{OPT_CONFIG['auto_scale_mode']}"
    if OPT_CONFIG.get('correct_gradient_magnitudes', False):
        suffix += "_cgm"
    if OPT_CONFIG.get('transfer_rank_schedule', 'all') != 'all':
        suffix += f"_trs-{OPT_CONFIG['transfer_rank_schedule']}-{OPT_CONFIG['transfer_ranks_per_step']}"
    if not OPT_CONFIG.get('scale_transfer_lr', True):
        suffix += "_no-stlr"
    # Scaled device sweep markers
    if AB_DEVICE == "scaledideal" or A_DEVICE == "scaledideal" or B_DEVICE == "scaledideal":
        suffix += f"_af{AF_RATIO:g}_unr{UNR:g}"
    if OPT_CONFIG.get('fi_continuous_alpha', False):
        suffix += "_fica"
    if OPT_CONFIG.get('ab_pulse_type', 'default') != 'default':
        suffix += f"_abpt-{OPT_CONFIG['ab_pulse_type']}"
    if OPT_CONFIG.get('ab_multilevel', False):
        suffix += "_abml"

    # Split mode markers (only when at least one ab param is split per-tile).
    split_keys = sorted(k for k, v in SPLIT_AB_PARAMS.items() if v)
    if split_keys:
        suffix += "_split-" + "-".join(split_keys)

    # Add lora target (always include for clarity)
    suffix += f"_{LORA_TARGET}"

    # Add head_layer if frozen (not default)
    if HEAD_LAYER == "freeze":
        suffix += "_headfreeze"

    # Add epoch count
    suffix += f"_{N_EPOCHS}ep"

    return suffix

os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# LRTT Device Functions
# =============================================================================

def _create_ab_device(tau_sec=0.0, dw_min=0.001981, multilevel=None, device_name=None,
                      reset_std=0.01):
    """Create A/B tile device based on AB_DEVICE setting.

    Options:
        6t1c              - Full 6T1C with all noise/variation (realistic)
        linearstep        - LinearStepDevice (no nonlinearity)
        linearstepideal   - LinearStepDevice with all dtod/std=0, w_max=1, w_min=-1
        constantstep      - ConstantStepDevice (constant step)
        constantstepideal - ConstantStepDevice with all dtod/std=0, w_max=1, w_min=-1
        constantstep6t1cgamma - LinearStepDevice with all dtod/std=0 but gamma_up/gamma_down from 6t1c
        fp                - FloatingPointDevice (perfect, no quantization/bounds)
        ideal             - IdealDevice

    Args:
        tau_sec: Retention time constant. If 0, lifetime=0 (no decay).
        dw_min: Minimum weight update step size for the device.
        multilevel: If set, w_max - w_min = (2 ** multilevel) * dw_min (symmetric).
                    Default (None) keeps w_max=1, w_min=-1.
        device_name: Optional override for the device type. None → use AB_DEVICE.
                     Used to instantiate A and B tiles independently when A_DEVICE
                     or B_DEVICE differs from AB_DEVICE.
        reset_std: σ for the reset (capacitor-discharge) operation. Used as the random
                   Gaussian source for B in gauss_b_* reinit modes. Default 0.01 matches
                   the 6T1C inherent floor noise. Applied to all pulsed-device branches;
                   ignored for non-pulsed device classes (fp / ideal).

    Note: reset_dtod is hardcoded to 0.0 (PulsedDevice default) on every pulsed-device
    branch — written explicitly for readability, not configurable.
    """
    name = device_name if device_name is not None else AB_DEVICE

    # Compute retention lifetime from tau_sec
    if tau_sec > 0:
        dt_batch_sec = 1.0
        delta = 1 - math.exp(-dt_batch_sec / tau_sec)
        lifetime = 1.0 / delta if delta > 0 else 0.0
    else:
        lifetime = 0.0

    # Compute symmetric w_max from multilevel
    if multilevel is not None and multilevel > 0:
        w_max = (2 ** multilevel) * dw_min / 2.0
    else:
        w_max = 1.0
    w_min = -w_max

    if name == "fp":
        return FloatingPointDevice()
    if name == "ideal":
        return IdealDevice()
    if name == "linearstep":
        return LinearStepDevice(
            dw_min=dw_min,
            lifetime=lifetime,
            reset_std=reset_std,
            reset_dtod=0.0,
        )
    if name == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=w_max,
            w_min=w_min,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            gamma_up_dtod=0.0,
            gamma_down_dtod=0.0,
            write_noise_std=0.0,
            reset_std=reset_std,
            reset_dtod=0.0,
            up_down=0.0,
            mult_noise=False,
            lifetime=lifetime,
        )
    if name == "constantstep":
        return ConstantStepDevice(
            dw_min=dw_min,
            lifetime=lifetime,
            reset_std=reset_std,
            reset_dtod=0.0,
        )
    if name == "constantstepideal":
        return ConstantStepDevice(
            dw_min=dw_min,
            w_max=w_max,
            w_min=w_min,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            reset_std=reset_std,
            reset_dtod=0.0,
            up_down=0.0,
            lifetime=lifetime,
        )
    if name == "constantstep6t1cgamma":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=w_max,
            w_min=w_min,
            gamma_up=-0.1678,
            gamma_down=0.1410,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            gamma_up_dtod=0.0,
            gamma_down_dtod=0.0,
            write_noise_std=0.0,
            reset_std=reset_std,
            reset_dtod=0.0,
            up_down=0.0,
            mult_noise=False,
            lifetime=lifetime,
        )
    if name == "scaledideal":
        # LinearStep with all dtod/write_noise/mult_noise zeroed; only gamma (× AF_RATIO)
        # and dw_min_std (× UNR) carry 6T1C nominal values scaled by sweep ratios.
        # AF=0, UNR=0 ⇒ exactly equivalent to constantstepideal.
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=w_max,
            w_min=w_min,
            gamma_up=-0.1678 * AF_RATIO,
            gamma_down=0.1410 * AF_RATIO,
            dw_min_std=0.3 * UNR,
            dw_min_dtod=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            gamma_up_dtod=0.0,
            gamma_down_dtod=0.0,
            write_noise_std=0.0,
            reset_std=reset_std,
            reset_dtod=0.0,
            up_down=0.0,
            mult_noise=False,
            lifetime=lifetime,
        )

    # Default: 6t1c (full noise)
    return LinearStepDevice(
        dw_min=dw_min,
        up_down=0.0,
        w_max=w_max,
        w_min=w_min,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=False,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=0.3,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=lifetime,
        lifetime_dtod=0.1,
        reset=0.0,
        reset_std=reset_std,
        reset_dtod=0.0,
    )


def _create_c_device(dw_min=0.001, reset_std=0.0):
    """Create device for C tile.

    Args:
        dw_min: Minimum weight update step size.
        reset_std: σ for reset operation. Default 0.0 (clean reset for transfer).

    Note: reset_dtod is hardcoded to 0.0 (PulsedDevice default) on every pulsed-device
    branch — written explicitly for readability, not configurable.
    """
    if C_DEVICE == "ideal":
        return IdealDevice()
    if C_DEVICE == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=1.0,
            w_min=-1.0,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            gamma_up_dtod=0.0,
            gamma_down_dtod=0.0,
            write_noise_std=0.0,
            reset_std=reset_std,
            reset_dtod=0.0,
            up_down=0.0,
            mult_noise=False,
        )
    if C_DEVICE == "constantstep":
        return ConstantStepDevice(
            dw_min=dw_min,
            reset_std=reset_std,
            reset_dtod=0.0,
        )
    if C_DEVICE == "constantstepideal":
        return ConstantStepDevice(
            dw_min=dw_min,
            w_max=1.0,
            w_min=-1.0,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            reset_std=reset_std,
            reset_dtod=0.0,
            up_down=0.0,
        )
    if C_DEVICE == "constantstep6t1cgamma":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=1.0,
            w_min=-1.0,
            gamma_up=-0.1678,
            gamma_down=0.1410,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            gamma_up_dtod=0.0,
            gamma_down_dtod=0.0,
            write_noise_std=0.0,
            reset_std=reset_std,
            reset_dtod=0.0,
            up_down=0.0,
            mult_noise=False,
        )
    return SoftBoundsDevice(
        dw_min=dw_min,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        reset_std=reset_std,
        reset_dtod=0.0,
        mult_noise=False,  # No multiplicative noise for C tile
    )


def create_frozen_analog_config(lrtt_config=None, out_noise=0.0):
    """Create analog config for non-LRTT linear layers (frozen analog).

    If lrtt_config is provided, derived from its C tile settings.
    Otherwise, creates a standalone config with default C tile settings.
    """
    from copy import deepcopy
    if lrtt_config is not None:
        rpu_config = SingleRPUConfig(
            device=deepcopy(lrtt_config.device.unit_cell_devices[2]),
        )
        rpu_config.mapping = deepcopy(lrtt_config.device.mapping_c)
        rpu_config.forward = deepcopy(lrtt_config.forward)
        rpu_config.backward = deepcopy(lrtt_config.backward)
    else:
        rpu_config = SingleRPUConfig(device=_create_c_device())
        rpu_config.mapping = MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=True,
            learn_out_scaling=OPT_CONFIG.get('learn_out_scaling', True),
            out_scaling_columnwise=True,
        )
        rpu_config.forward.out_noise = out_noise
        rpu_config.backward.out_noise = out_noise
        rpu_config.forward.is_perfect = IS_PERFECT
        rpu_config.backward.is_perfect = IS_PERFECT
        if NO_QUANT:
            rpu_config.forward.inp_res = -1
            rpu_config.forward.out_res = -1
            rpu_config.backward.inp_res = -1
            rpu_config.backward.out_res = -1
        if BACKWARD_OUT_BOUND != 12.0:
            rpu_config.backward.out_bound = BACKWARD_OUT_BOUND
    return rpu_config


def create_lrtt_config(rank, transfer_every, transfer_lr, fast_lr, reinit_mode,
                       c_dw_min=0.001, c_desired_bl=None, out_noise=0.0,
                       auto_scale_mode='none', correct_gradient_magnitudes=False,
                       transfer_rank_schedule='all', transfer_ranks_per_step=1,
                       scale_transfer_lr=True,
                       fi_continuous_alpha=False,
                       ab_pulse_type='default',
                       a_tau_sec=0.0, b_tau_sec=0.0,
                       a_dw_min=0.001981, b_dw_min=0.001981,
                       a_multilevel=None, b_multilevel=None,
                       a_reset_std=0.01, b_reset_std=0.01,
                       ab_desired_bl=31,
                       a_desired_bl=None, b_desired_bl=None,
                       ab_weight_scaling_omega=0.0,
                       a_weight_scaling_omega=None, b_weight_scaling_omega=None,
                       lora_alpha=1.0):
    """Create LRTT RPU configuration for analog layers.

    Per-tile parameters (each can differ between A and B):
      - tau_sec, dw_min, multilevel, reset_std: applied to LinearStepDevice instances
      - desired_bl: applied via PythonLRTTDevice.{a,b}_desired_bl override
      - weight_scaling_omega: applied via PythonLRTTDevice.{mapping_a, mapping_b}
        overrides of mapping_ab

    a_desired_bl / b_desired_bl: when None, both inherit ab_desired_bl (the rpu.update
    level value). When set, override per tile.
    a_weight_scaling_omega / b_weight_scaling_omega: when None, both inherit
    ab_weight_scaling_omega (the mapping_ab level value). When set, build mapping_a /
    mapping_b override objects.
    """
    # A and B tile devices: independently overridable via A_DEVICE / B_DEVICE.
    # When both are None, both A and B use AB_DEVICE (legacy behavior preserved).
    a_device = _create_ab_device(tau_sec=a_tau_sec, dw_min=a_dw_min, multilevel=a_multilevel,
                                 device_name=A_DEVICE, reset_std=a_reset_std)
    b_device = _create_ab_device(tau_sec=b_tau_sec, dw_min=b_dw_min, multilevel=b_multilevel,
                                 device_name=B_DEVICE, reset_std=b_reset_std)
    c_device = _create_c_device(dw_min=c_dw_min)

    # Scale B initialization to match new w_max/w_min bounds.
    # Default w_max=1.0; with multilevel, w_max = 2^(multilevel-1) * b_dw_min.
    # reinit_gain multiplies B_init (and A_init, but A is "zero" by default).
    if b_multilevel is not None and b_multilevel > 0:
        b_w_max = (2 ** b_multilevel) * b_dw_min / 2.0
        b_reinit_gain = REINIT_GAIN * b_w_max
    else:
        b_reinit_gain = REINIT_GAIN

    te = transfer_every

    # Build mapping_a / mapping_b overrides only when per-tile omega differs from shared.
    def _build_mapping(omega):
        return MappingParameter(
            weight_scaling_omega=omega,
            learn_out_scaling=False,
            max_input_size=0 if IS_PERFECT else 512,
            max_output_size=0 if IS_PERFECT else 512,
        )

    mapping_ab_obj = _build_mapping(ab_weight_scaling_omega)
    mapping_a_obj = _build_mapping(a_weight_scaling_omega) if a_weight_scaling_omega is not None else None
    mapping_b_obj = _build_mapping(b_weight_scaling_omega) if b_weight_scaling_omega is not None else None

    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=lora_alpha,
        fast_lr=fast_lr,
        reinit_gain=b_reinit_gain,
        reinit_mode=reinit_mode,
        unit_cell_devices=[a_device, b_device, c_device],
        train_c_bias=False,        # C tile bias frozen
        mapping_ab=mapping_ab_obj,
        mapping_a=mapping_a_obj,
        mapping_b=mapping_b_obj,
        mapping_c=MappingParameter(
            weight_scaling_omega=0.6,
            weight_scaling_columnwise=True,
            learn_out_scaling=OPT_CONFIG.get('learn_out_scaling', True),
            out_scaling_columnwise=True,
            max_input_size=0 if IS_PERFECT else 512,
            max_output_size=0 if IS_PERFECT else 512,
        ),
    )
    device_config.transfer_lr = transfer_lr
    device_config.units_in_mbatch = True
    device_config.transfer_method = TRANSFER_METHOD
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = FORWARD_INJECT
    device_config.no_adc_ab_projection = OPT_CONFIG.get('no_adc_ab_proj', False)
    device_config.auto_scale_mode = auto_scale_mode
    device_config.correct_gradient_magnitudes = correct_gradient_magnitudes
    device_config.transfer_rank_schedule = transfer_rank_schedule
    device_config.transfer_ranks_per_step = transfer_ranks_per_step
    device_config.scale_transfer_lr = scale_transfer_lr
    device_config.fi_continuous_alpha = fi_continuous_alpha
    device_config.ab_pulse_type = ab_pulse_type
    if c_desired_bl is not None:
        device_config.c_desired_bl = c_desired_bl
    if a_desired_bl is not None:
        device_config.a_desired_bl = a_desired_bl
    if b_desired_bl is not None:
        device_config.b_desired_bl = b_desired_bl

    # Dynamic TE
    device_config.dynamic_te = DYNAMIC_TE
    device_config.dynamic_te_power = DYNAMIC_TE_POWER
    device_config.dynamic_te_max = te * 20
    device_config.te_warmup_schedule = TE_WARMUP_SCHEDULE + [te]
    device_config.te_warmup_steps = TE_WARMUP_STEPS

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.mapping.max_input_size = 0 if IS_PERFECT else 512
    rpu_config.mapping.max_output_size = 0 if IS_PERFECT else 512

    rpu_config.update.desired_bl = ab_desired_bl

    rpu_config.forward.out_noise = out_noise
    rpu_config.backward.out_noise = out_noise
    rpu_config.forward.is_perfect = IS_PERFECT
    rpu_config.backward.is_perfect = IS_PERFECT
    if NO_QUANT:
        rpu_config.forward.inp_res = -1
        rpu_config.forward.out_res = -1
        rpu_config.backward.inp_res = -1
        rpu_config.backward.out_res = -1

    if BACKWARD_OUT_BOUND != 12.0:
        rpu_config.backward.out_bound = BACKWARD_OUT_BOUND

    return rpu_config


# =============================================================================
# Model Functions
# =============================================================================

def list_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def get_lrtt_target_module_names(lora_target):
    """Get module name patterns for LRTT conversion based on lora_target.

    Returns list of substrings identifying which linear layers should be LRTT.
    Returns [] for none mode (fully digital baseline).
    """
    if lora_target == "none":
        return []  # No LRTT (digital baseline)
    elif lora_target == "linear1":
        return ["linear1"]  # Only LRTT-able layer in 2-layer MLP (classifier excluded)
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


class MLP(nn.Module):
    """2-layer MLP for MNIST: 784 → HIDDEN_DIM → 10.

    Layer naming convention:
        - linear1: input projection (LRTT-eligible)
        - classifier: output head (always digital, controlled by HEAD_LAYER/HEAD_ANALOG)
    """
    def __init__(self, hidden=HIDDEN_DIM):
        super().__init__()
        self.linear1 = nn.Linear(784, hidden)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten 28×28 → 784
        x = self.relu(self.linear1(x))
        x = self.classifier(x)
        return self.logsoftmax(x)


def create_model(params):
    """Create MLP model for MNIST with selective LRTT analog layers.

    Architecture:
        - LRTT Target layer (based on --lora-target) → LRTT Analog
        - classifier (output head) → Digital TRAINABLE / FROZEN / Frozen Analog
          (controlled by HEAD_LAYER, HEAD_ANALOG)

    LoRA Target Options (--lora-target):
        - none: fully digital baseline
        - linear1: linear1 → LRTT Analog (default)

    LRTT layers have:
        - A/B tiles: TRAINABLE
        - C-tile: FROZEN (initial weights)
        - out_scaling: TRAINABLE
        - bias: FROZEN
    """
    from aihwkit.nn import AnalogLinear

    model = MLP(hidden=HIDDEN_DIM)

    # Get LRTT target patterns
    lrtt_patterns = get_lrtt_target_module_names(LORA_TARGET)

    def is_lrtt_target(layer_name):
        """Check if layer should be converted to LRTT Analog."""
        # classifier (output head) is always digital
        if "classifier" in layer_name:
            return False
        if not lrtt_patterns:
            return False  # none mode
        return any(p in layer_name for p in lrtt_patterns)

    # Build exclude list: all layers that should NOT be converted to LRTT
    all_linear_names = list_linear_layers(model)
    exclude_modules = []
    for name in all_linear_names:
        if not is_lrtt_target(name):
            # Use full path for exclude_modules (convert_to_analog requires exact match)
            exclude_modules.append(name)

    # Exclude classifier (output head, always digital — controlled by HEAD_LAYER/HEAD_ANALOG)
    exclude_modules.append("classifier")
    exclude_modules = list(set(exclude_modules))  # Remove duplicates

    # Step 1: Convert only LRTT target layers to LRTT Analog (skip if none mode)
    if LORA_TARGET == "none":
        # None mode: fully digital, no analog conversion
        analog_count = 0
    else:
        te = int(params["transfer_every"])
        lrtt_config = create_lrtt_config(
            rank=int(params["rank"]),
            transfer_every=te,
            transfer_lr=params["transfer_lr"],
            fast_lr=params["fast_lr"],
            reinit_mode=params["reinit_mode"],
            a_tau_sec=params["a_tau_sec"],
            b_tau_sec=params["b_tau_sec"],
            a_dw_min=params["a_dw_min"],
            b_dw_min=params["b_dw_min"],
            a_multilevel=params["a_multilevel"],
            b_multilevel=params["b_multilevel"],
            a_reset_std=params["a_reset_std"],
            b_reset_std=params["b_reset_std"],
            ab_desired_bl=params["ab_desired_bl"],
            a_desired_bl=params["a_desired_bl"],
            b_desired_bl=params["b_desired_bl"],
            c_dw_min=params["c_dw_min"],
            c_desired_bl=params["c_desired_bl"],
            out_noise=params["out_noise"],
            ab_weight_scaling_omega=params["ab_weight_scaling_omega"],
            a_weight_scaling_omega=params["a_weight_scaling_omega"],
            b_weight_scaling_omega=params["b_weight_scaling_omega"],
            auto_scale_mode=OPT_CONFIG['auto_scale_mode'],
            correct_gradient_magnitudes=OPT_CONFIG['correct_gradient_magnitudes'],
            transfer_rank_schedule=OPT_CONFIG['transfer_rank_schedule'],
            transfer_ranks_per_step=OPT_CONFIG['transfer_ranks_per_step'],
            scale_transfer_lr=OPT_CONFIG['scale_transfer_lr'],
            fi_continuous_alpha=OPT_CONFIG['fi_continuous_alpha'],
            lora_alpha=params["lora_alpha"],
            ab_pulse_type=OPT_CONFIG['ab_pulse_type'],
        )

        # Convert to analog with exclusions (only LRTT targets get converted)
        model = convert_to_analog(model, lrtt_config, exclude_modules=exclude_modules)

        # Count analog layers
        analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))

    # Step 1.5: Convert non-LRTT linear and/or classifier to frozen analog (if enabled)
    # Already-converted LRTT layers (AnalogLinear) are naturally skipped by convert_to_analog
    frozen_analog_count = 0
    any_frozen_analog = (ENCODER_ANALOG and LORA_TARGET != "linear1") or HEAD_ANALOG
    if any_frozen_analog:
        # Collect existing tile IDs (LRTT sub-tiles) before frozen conversion
        existing_tile_ids = set()
        for m in model.modules():
            if isinstance(m, AnalogLinear):
                for tile in m.analog_tiles():
                    existing_tile_ids.add(id(tile))

        frozen_config = create_frozen_analog_config(
            lrtt_config if LORA_TARGET != "none" else None,
            out_noise=params.get("out_noise", 0.0),
        )
        frozen_exclude = []
        if not HEAD_ANALOG:
            frozen_exclude.append("classifier")
        # Exclude already-LRTT layers (don't re-convert) and non-LRTT non-classifier
        # linear layers (if ENCODER_ANALOG is off — leave them digital).
        for name in all_linear_names:
            if is_lrtt_target(name):
                frozen_exclude.append(name)            # already LRTT
            elif not ENCODER_ANALOG and "classifier" not in name:
                frozen_exclude.append(name)            # non-LRTT linear, keep digital
        model = convert_to_analog(model, frozen_config, exclude_modules=frozen_exclude)
        frozen_analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear)) - analog_count

        # Hook frozen analog tile updates to no-op (prevent optimizer from modifying weights).
        # AnalogSGD/Adam calls tile.update() on ALL analog tiles unconditionally;
        # LRTT tiles are already hooked in lrtt_tile.py, but frozen analog tiles need this.
        def _frozen_noop_update(x_input, d_input, *args, **kwargs):
            return None

        # Bypass AnalogFunction.apply() for frozen tiles to avoid activation/gradient
        # saving overhead. AnalogFunction saves input 3x (autograd ctx, analog_input list,
        # analog_grad_output list) for weight updates that never happen on frozen tiles.
        # This custom Function matches LRTT C tile behavior (direct tile call).
        import types
        class _FrozenAnalogFwd(torch.autograd.Function):
            @staticmethod
            @no_grad()
            def forward(ctx, analog_tile, x_input, is_test):
                ctx.analog_tile = analog_tile
                ctx.saved_analog_tensors = [x_input]
                out = analog_tile.joint_forward(x_input, is_test, ctx)
                ctx.save_for_backward(*ctx.saved_analog_tensors)
                ctx.saved_analog_tensors = []
                return out
            @staticmethod
            @no_grad()
            def backward(ctx, grad_output):
                ctx.saved_analog_tensors = list(ctx.saved_tensors)
                grad_input = ctx.analog_tile.backward(grad_output, ctx)
                ctx.saved_analog_tensors = []
                return None, grad_input, None

        def _frozen_analog_forward(self, x_input, tensor_view=None):
            out = _FrozenAnalogFwd.apply(self, x_input, not self.training)
            if tensor_view is None:
                tensor_view = self.get_tensor_view(out.dim())
            out = self.apply_out_scaling(out, tensor_view)
            if self.digital_bias:
                return out + self.bias.view(*tensor_view)
            return out

        for mod_name, m in model.named_modules():
            if isinstance(m, AnalogLinear):
                for tile in m.analog_tiles():
                    if id(tile) not in existing_tile_ids:
                        # Head analog tiles remain trainable (weight + bias)
                        if HEAD_ANALOG and "classifier" in mod_name:
                            continue
                        tile.update = _frozen_noop_update
                        tile.forward = types.MethodType(_frozen_analog_forward, tile)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LRTT Analog layers: {analog_count}, Frozen analog layers: {frozen_analog_count}")
    print(f"  Total params: {total_params:,}, Trainable (before grad set): {trainable_before:,}")

    # Step 2: Set requires_grad
    # - LRTT layers: A/B TRAINABLE, C hook-protected (transfer-only), bias TRAINABLE (from-scratch)
    # - classifier: TRAINABLE if HEAD_LAYER=="train", else FROZEN
    # - Other linear layers (e.g., linear1 in "none" mode, future linear2/3): TRAINABLE (from-scratch)
    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            param.requires_grad = not OPT_CONFIG['no_transfer']
        elif "tile_c" in name:
            pass  # tile_c.weight hook-protected (transfer-only); bias via train_c_bias in lrtt_tile.py
        elif "out_scaling_alpha" in name:
            pass  # out_scaling: trainable per mapping_c
        elif "classifier" in name:
            param.requires_grad = (HEAD_LAYER == "train")
        else:
            param.requires_grad = True  # digital layers (incl. future linear2/3): trainable from-scratch

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable (after grad set): {trainable_after:,}")
    print(f"  LoRA target: {LORA_TARGET} -> {lrtt_patterns if lrtt_patterns else 'none (digital baseline)'}")

    try:
        return model.to(DEVICE)
    except Exception:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


# =============================================================================
# Data Functions
# =============================================================================

def load_data():
    """Load MNIST dataset and return (train_loader, val_loader)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST(root="/tmp/mnist", train=True, download=True, transform=transform)
    val_set = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=transform)

    if TRAIN_SUBSET_SIZE > 0:
        indices = torch.randperm(
            len(train_set), generator=torch.Generator().manual_seed(SEED)
        )[:TRAIN_SUBSET_SIZE].tolist()
        train_set = Subset(train_set, indices)
    if EVAL_SUBSET_SIZE > 0:
        val_set = Subset(val_set, range(min(EVAL_SUBSET_SIZE, len(val_set))))

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE // GRAD_ACCUM_STEPS, shuffle=True,
        num_workers=2, generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        val_set, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=2,
    )

    return train_loader, val_loader


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_model(model, val_loader):
    """Evaluate MLP model on MNIST. Returns (accuracy_pct, val_loss)."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.NLLLoss(reduction='sum')
    with no_grad():
        for data, target in val_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    model.train()
    acc = 100.0 * correct / max(1, total)
    avg_loss = total_loss / max(1, total)
    return acc, avg_loss


# =============================================================================
# Optuna Objective
# =============================================================================

def objective(trial, train_loader, val_loader):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 6e-3, 5e0, log=True)

    # LRTT parameters: skip sweep if --no-transfer (A/B frozen, no transfer happens)
    if OPT_CONFIG['no_transfer']:
        transfer_lr = 0.1        # fixed (not used anyway)
        transfer_every = 999999999
        rank_exp = 2             # fixed (A=0 init, no effect)
        rank = 4
        fast_lr = 1.0            # fixed (no effect)
        a_tau_sec = b_tau_sec = 0.0    # fixed
    else:
        transfer_lr = trial.suggest_float('transfer_lr', 1e-5, 3e-2, log=True)
        transfer_every = trial.suggest_int('transfer_every', 10, 10, log=True)
        rank_exp = trial.suggest_int('rank_exp', 3, 3)
        rank = 2 ** rank_exp
        fast_lr = trial.suggest_float('fast_lr', 7e-4, 2e4, log=True)
        # tau_sec → device.lifetime is per-tile splittable. Default shared (single ab variable).
        if SPLIT_AB_PARAMS.get('tau_sec', False):
            a_tau_sec = trial.suggest_float('a_tau_sec', 0, 0, log=False)
            b_tau_sec = trial.suggest_float('b_tau_sec', 0, 0, log=False)
        else:
            tau_sec = trial.suggest_float('tau_sec', 0, 0, log=False)  # 0 = no decay
            a_tau_sec = b_tau_sec = tau_sec

    # A/B device params
    # dw_min: per-tile splittable. SPLIT_AB_PARAMS['dw_min']=False sweeps a single
    # ab_dw_min and applies it to both; True sweeps a_dw_min and b_dw_min independently.
    if SPLIT_AB_PARAMS.get('dw_min', False):
        a_dw_min = trial.suggest_float('a_dw_min', 0.001981, 0.001981, log=True)
        b_dw_min = trial.suggest_float('b_dw_min', 0.001981, 0.001981, log=True)
    else:
        ab_dw_min = trial.suggest_float('ab_dw_min', 0.001981, 0.001981, log=True)  # 6t1c canonical (matches v2 selector sweep)
        a_dw_min = b_dw_min = ab_dw_min

    # desired_bl: per-tile splittable (a_desired_bl / b_desired_bl override the
    # rpu.update level desired_bl in aihwkit's create_update_params).
    if SPLIT_AB_PARAMS.get('desired_bl', False):
        a_desired_bl = trial.suggest_int('a_desired_bl', 31, 31)
        b_desired_bl = trial.suggest_int('b_desired_bl', 31, 31)
        ab_desired_bl = a_desired_bl  # placeholder for rpu.update.desired_bl base; A/B override
    else:
        ab_desired_bl = trial.suggest_int('ab_desired_bl', 31, 31)            # default: 31
        a_desired_bl = b_desired_bl = None  # None → A and B inherit ab_desired_bl base

    # multilevel: per-tile splittable. w_max-w_min = 2^multilevel * dw_min (symmetric).
    # When enabled, B init also scales accordingly. Default: disabled (w_max=1.0).
    if OPT_CONFIG.get('ab_multilevel', False):
        if SPLIT_AB_PARAMS.get('multilevel', False):
            a_multilevel = trial.suggest_int('a_multilevel', 12, 12)
            b_multilevel = trial.suggest_int('b_multilevel', 12, 12)
        else:
            ab_multilevel = trial.suggest_int('ab_multilevel', 12, 12)
            a_multilevel = b_multilevel = ab_multilevel
    else:
        a_multilevel = b_multilevel = None

    # reset_std: per-tile splittable. Default split (A=6T1C floor 0.01 fixed, B=sweep
    # for gauss_b_* random Gaussian σ). Set SPLIT_AB_PARAMS['reset_std']=False to
    # sweep a single ab_reset_std applied to both tiles (ablation use case).
    if SPLIT_AB_PARAMS.get('reset_std', True):
        a_reset_std = trial.suggest_float('a_reset_std', 1e-30, 1e-30, log=True)
        b_reset_std = trial.suggest_float('b_reset_std', 2e-5, 3e4, log=True)
    else:
        ab_reset_std = trial.suggest_float('ab_reset_std', 0.0, 0.0, log=True)
        a_reset_std = b_reset_std = ab_reset_std

    # C tile pulsed transfer params (only meaningful for onehot/direct)
    if TRANSFER_METHOD in ("onehot", "direct") and not OPT_CONFIG['no_transfer']:
        c_dw_min = trial.suggest_float('c_dw_min', 0.001, 0.001, log=True)
        c_desired_bl = trial.suggest_int('c_desired_bl', 31, 31)
    else:
        c_dw_min = 0.001   # default (unused for "set")
        c_desired_bl = None

    # IO / mapping params
    if IO_NOISE:
        out_noise = trial.suggest_float('out_noise', 0.0, 0.0)
    else:
        out_noise = 0.0
    # weight_scaling_omega: per-tile splittable via mapping_a / mapping_b override of
    # mapping_ab in aihwkit. Default shared.
    if SPLIT_AB_PARAMS.get('weight_scaling_omega', False):
        a_weight_scaling_omega = trial.suggest_float('a_weight_scaling_omega', 0.0, 0.0)
        b_weight_scaling_omega = trial.suggest_float('b_weight_scaling_omega', 0.0, 0.0)
        ab_weight_scaling_omega = a_weight_scaling_omega  # placeholder for legacy callers
    else:
        ab_weight_scaling_omega = trial.suggest_float('ab_weight_scaling_omega', 0.0, 0.0)
        a_weight_scaling_omega = b_weight_scaling_omega = ab_weight_scaling_omega

    # lora_alpha: only sweep when forward_inject is ON and fi_continuous_alpha is OFF
    if FORWARD_INJECT and not OPT_CONFIG.get('fi_continuous_alpha', False):
        lora_alpha = trial.suggest_float('lora_alpha', 0.1, 10.0, log=True)
    else:
        lora_alpha = 1.0

    # weight_decay: tune or fix to 0
    if OPT_CONFIG['tune_wd']:
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
    else:
        weight_decay = 0.0

    # momentum: 0.9 fixed by default, 0.0 with --no-momentum
    if OPT_CONFIG['tune_momentum']:
        momentum = 0.9
    else:
        momentum = 0.0

    # nesterov: True fixed by default, False with --no-nesterov
    if OPT_CONFIG['tune_nesterov'] and momentum > 0:
        nesterov = True
    else:
        nesterov = False

    # reinit_mode: use fixed value if set, otherwise tune
    if OPT_CONFIG['reinit_mode'] is not None:
        reinit_mode = OPT_CONFIG['reinit_mode']
    else:
        reinit_mode = trial.suggest_categorical('reinit_mode', ['standard', 'decay', 'hybrid'])

    # optimizer: always use config value
    optimizer_name = OPT_CONFIG['optimizer']

    params = {
        "rank": rank,
        "transfer_every": transfer_every,
        "transfer_lr": transfer_lr,
        "fast_lr": fast_lr,
        "reinit_mode": reinit_mode,
        "a_tau_sec": a_tau_sec,
        "b_tau_sec": b_tau_sec,
        "a_dw_min": a_dw_min,
        "b_dw_min": b_dw_min,
        "ab_desired_bl": ab_desired_bl,
        "a_desired_bl": a_desired_bl,
        "b_desired_bl": b_desired_bl,
        "a_multilevel": a_multilevel,
        "b_multilevel": b_multilevel,
        "a_reset_std": a_reset_std,
        "b_reset_std": b_reset_std,
        "c_dw_min": c_dw_min,
        "c_desired_bl": c_desired_bl,
        "out_noise": out_noise,
        "ab_weight_scaling_omega": ab_weight_scaling_omega,
        "a_weight_scaling_omega": a_weight_scaling_omega,
        "b_weight_scaling_omega": b_weight_scaling_omega,
        "lora_alpha": lora_alpha,
    }

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting")
    print(f"{'='*70}")
    print(f"  rank={rank}, transfer_every={transfer_every}, transfer_lr={transfer_lr:.4e}")
    print(f"  fast_lr={fast_lr:.2e}, lr={learning_rate:.2e}, wd={weight_decay:.2e}")
    print(f"  momentum={momentum:.2f}, nesterov={nesterov}, reinit_mode={reinit_mode}")
    if a_tau_sec == b_tau_sec:
        print(f"  tau_sec={a_tau_sec:.1f}, optimizer={optimizer_name}")
    else:
        print(f"  a_tau_sec={a_tau_sec:.1f}, b_tau_sec={b_tau_sec:.1f}, optimizer={optimizer_name}")
    if a_multilevel is not None or b_multilevel is not None:
        a_w = (2**a_multilevel)*a_dw_min/2 if a_multilevel else 1.0
        b_w = (2**b_multilevel)*b_dw_min/2 if b_multilevel else 1.0
        print(f"  a_dw_min={a_dw_min:.4e} (w_max=±{a_w:.4e}, multilevel={a_multilevel}), "
              f"b_dw_min={b_dw_min:.4e} (w_max=±{b_w:.4e}, multilevel={b_multilevel}), "
              f"ab_desired_bl={ab_desired_bl}")
    else:
        print(f"  a_dw_min={a_dw_min:.4e}, b_dw_min={b_dw_min:.4e}, ab_desired_bl={ab_desired_bl}, multilevel=off (w_max=±1.0)")
    print(f"  a_reset_std={a_reset_std:.4e}, b_reset_std={b_reset_std:.4e}")
    if TRANSFER_METHOD in ("onehot", "direct"):
        print(f"  c_dw_min={c_dw_min:.4e}, c_desired_bl={c_desired_bl}")
    print(f"{'='*70}")

    model = None
    try:
        manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)

        model = create_model(params)
        criterion = nn.NLLLoss()

        if LORA_TARGET == "none" and not ENCODER_ANALOG and not HEAD_ANALOG:
            # None mode (no analog tiles): use standard PyTorch optimizers
            if optimizer_name == "AnalogSGD":
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=learning_rate,
                    weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                )
        else:
            # Analog optimizers: required for LRTT tiles and frozen analog tiles
            # (AnalogSGD/Adam calls analog_ctx.reset() to prevent memory leak)
            if optimizer_name == "AnalogSGD":
                optimizer = AnalogSGD(
                    model.parameters(), lr=learning_rate,
                    weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                )
            else:
                optimizer = AnalogAdam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                )
            optimizer.regroup_param_groups()
            optimizer._grad_accum_steps = GRAD_ACCUM_STEPS
            trial.set_user_attr("grad_accum_steps", GRAD_ACCUM_STEPS)

        scheduler = StepLR(optimizer, step_size=STEP_LR_SIZE, gamma=STEP_LR_GAMMA)

        best_acc = 0.0
        epochs_without_improvement = 0
        best_train_loss = float('inf')
        train_loss_no_improvement = 0

        _nan_detected = False
        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
            optimizer.zero_grad()
            for micro_step, batch in enumerate(pbar):
                data, target = batch
                data = data.to(DEVICE)
                target = target.to(DEVICE)

                output = model(data)
                loss = criterion(output, target) / GRAD_ACCUM_STEPS
                if torch.isnan(loss):
                    tqdm.write(f"[Trial {trial.number}] NaN loss at epoch {epoch}. Aborting.")
                    _nan_detected = True
                    break
                loss.backward()

                if (micro_step + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * GRAD_ACCUM_STEPS
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")

            if _nan_detected:
                break

            train_loss = total_loss / num_batches if num_batches > 0 else 0.0
            scheduler.step()

            eval_acc, eval_loss = evaluate_model(model, val_loader)

            improved = ""
            if eval_acc > best_acc:
                best_acc = eval_acc
                epochs_without_improvement = 0
                improved = " ★"
            else:
                epochs_without_improvement += 1

            train_loss_improved = ""
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                train_loss_no_improvement = 0
                train_loss_improved = " ↓"
            else:
                train_loss_no_improvement += 1

            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"[Trial {trial.number}] Epoch {epoch:3d} | "
                  f"Acc: {eval_acc:6.2f}% | Val loss: {eval_loss:.4f} | Best acc: {best_acc:6.2f}% | "
                  f"Train loss: {train_loss:.4f}{train_loss_improved} | LR: {current_lr:.2e} | "
                  f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}{improved}")

            trial.report(best_acc, epoch)
            trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)

            if best_train_loss > TRAIN_LOSS_THRESHOLD and train_loss_no_improvement >= TRAIN_LOSS_EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Train loss early stop at epoch {epoch} "
                          f"(train_loss={train_loss:.4f} > {TRAIN_LOSS_THRESHOLD}, no improvement for {train_loss_no_improvement} epochs)")
                break

            if best_train_loss <= TRAIN_LOSS_THRESHOLD and epochs_without_improvement >= EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}")
                break

            if trial.should_prune():
                tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        print(f"\n[Trial {trial.number}] Finished - Best acc: {best_acc:.2f}%")
        print(f"{'='*70}\n")
        return best_acc

    except Exception as e:
        error_msg = str(e)[:500]
        trial.set_user_attr("error", error_msg)
        print(f"[Trial {trial.number}] Error: {error_msg}")
        raise

    finally:
        # Delete training loop vars to release model refs for C++ destructor
        try:
            del output
        except NameError:
            pass
        try:
            del loss
        except NameError:
            pass
        try:
            del pbar
        except NameError:
            pass
        try:
            del batch
        except NameError:
            pass
        try:
            del data
        except NameError:
            pass
        try:
            del target
        except NameError:
            pass
        # Delete in reverse dependency order: scheduler → optimizer → model
        # optimizer holds references to analog tiles via param_groups
        if 'scheduler' in dir():
            del scheduler
        if 'optimizer' in dir():
            del optimizer
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        tqdm.write(f"[Trial {trial.number}] GPU cache cleared")


# =============================================================================
# Visualization
# =============================================================================

def visualize_study(study, save_dir):
    """Visualize optimization history, parameter importance, and LR vs Accuracy."""
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        print("No completed trials to visualize.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    trial_numbers = [t.number for t in complete_trials]
    acc_scores = [t.value for t in complete_trials]

    # Optimization history
    axes[0].scatter(trial_numbers, acc_scores, alpha=0.6)
    axes[0].plot(trial_numbers,
                 [max(acc_scores[:i+1]) for i in range(len(acc_scores))],
                 'r-', linewidth=2, label='Best so far')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Optimization History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Parameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importances.keys())[::-1], list(importances.values())[::-1])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Parameter Importance')
    except Exception:
        axes[1].text(0.5, 0.5, 'Not enough trials', ha='center', va='center',
                     transform=axes[1].transAxes)

    # LR vs Accuracy
    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    axes[2].scatter(lrs, acc_scores, alpha=0.6)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Learning Rate')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Learning Rate vs Accuracy')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "visualization.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Visualization saved.")


def print_study_summary(study):
    """Print study summary."""
    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Study: {study.study_name}, Trials: {len(study.trials)} ({len(complete_trials)} complete)")
    if complete_trials:
        acc_scores = [t.value for t in complete_trials]
        print(f"Best acc: {max(acc_scores):.2f}%, Mean acc: {sum(acc_scores)/len(acc_scores):.2f}%")
        print(f"Best params: {study.best_params}")


# =============================================================================
# Main
# =============================================================================

def main():
    global BATCH_SIZE, GRAD_ACCUM_STEPS, N_EPOCHS, STEP_LR_SIZE, STEP_LR_GAMMA, TRANSFER_METHOD, AB_DEVICE, A_DEVICE, B_DEVICE, C_DEVICE, IO_NOISE, FORWARD_INJECT, IS_PERFECT, NO_QUANT, LORA_TARGET, HEAD_LAYER, ENCODER_ANALOG, HEAD_ANALOG, BACKWARD_OUT_BOUND, AF_RATIO, UNR, _oom_retry_pending

    parser = argparse.ArgumentParser(description="Optuna sweep for MLP MNIST LRTT")
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: auto-generated based on config)')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--optimizer', type=str, default='AnalogAdam',
                        choices=['AnalogSGD', 'AnalogAdam'],
                        help='Optimizer type (default: AnalogAdam)')
    parser.add_argument('--no-wd', action='store_true',
                        help='Disable weight decay tuning (fix to 0)')
    parser.add_argument('--no-momentum', action='store_true',
                        help='Disable momentum tuning (fix to 0, SGD only)')
    parser.add_argument('--no-nesterov', action='store_true',
                        help='Disable nesterov tuning (fix to False, SGD only)')
    parser.add_argument('--reinit-mode', type=str, default=None,
                        choices=['standard', 'decay', 'hybrid',
                                 'orthogonal_zero', 'orthogonal_decay',
                                 'gauss_b_zero', 'gauss_b_decay',
                                 'gauss_a_zero', 'gauss_a_decay',
                                 'selector_b_zero', 'selector_b_decay',
                                 'selector_a_zero', 'selector_a_decay'],
                        help='Fix reinit mode (default: tune among standard/decay/hybrid)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS,
                        help=f'Number of epochs (default: {N_EPOCHS})')
    parser.add_argument('--step-lr-size', type=int, default=STEP_LR_SIZE,
                        help=f'StepLR step size in epochs (default: {STEP_LR_SIZE})')
    parser.add_argument('--step-lr-gamma', type=float, default=STEP_LR_GAMMA,
                        help=f'StepLR gamma (default: {STEP_LR_GAMMA})')
    parser.add_argument('--transfer-method', type=str, default=TRANSFER_METHOD,
                        choices=['onehot', 'direct', 'set'],
                        help=f'Transfer method (default: {TRANSFER_METHOD})')
    parser.add_argument('--ab-device', type=str, default=AB_DEVICE,
                        choices=['6t1c', 'linearstep', 'linearstepideal', 'constantstep', 'constantstepideal', 'constantstep6t1cgamma', 'scaledideal', 'fp', 'ideal'],
                        help=f'A/B tile device type (default: {AB_DEVICE}). Used for both A and B unless --a-device or --b-device overrides.')
    parser.add_argument('--a-device', type=str, default=None,
                        choices=['6t1c', 'linearstep', 'linearstepideal', 'constantstep', 'constantstepideal', 'constantstep6t1cgamma', 'scaledideal', 'fp', 'ideal'],
                        help='Override A tile device only (default: same as --ab-device)')
    parser.add_argument('--b-device', type=str, default=None,
                        choices=['6t1c', 'linearstep', 'linearstepideal', 'constantstep', 'constantstepideal', 'constantstep6t1cgamma', 'scaledideal', 'fp', 'ideal'],
                        help='Override B tile device only (default: same as --ab-device)')
    parser.add_argument('--c-device', type=str, default=C_DEVICE,
                        choices=['softboundsideal', 'linearstepideal', 'constantstep', 'constantstepideal', 'constantstep6t1cgamma', 'ideal'],
                        help=f'C tile device type (default: {C_DEVICE})')
    parser.add_argument('--no-io-noise', action='store_true',
                        help='Disable IO out_noise (resolution kept)')
    parser.add_argument('--forward-inject', action='store_true',
                        help='Enable forward noise injection')
    parser.add_argument('--is-perfect', action='store_true',
                        help='Use ideal FP forward/backward (no ADC/DAC/noise)')
    parser.add_argument('--no-quant', action='store_true',
                        help='Disable DAC/ADC quantization (inp_res/out_res)')
    parser.add_argument('--no-transfer', action='store_true',
                        help='Disable transfer (set transfer_every to infinity)')
    parser.add_argument('--no-adc-ab-proj', action='store_true',
                        help='Use digital matmul for A/B projections (no ADC/DAC between B and A)')
    parser.add_argument('--lora-target', type=str, default=LORA_TARGET,
                        choices=['none', 'linear1'],
                        help='LoRA target: none or linear1 (default: linear1)')
    parser.add_argument('--head-layer', type=str, default=HEAD_LAYER,
                        choices=['train', 'freeze'],
                        help='Output head (classifier): train or freeze (default: train)')
    parser.add_argument('--encoder-analog', action='store_true', default=ENCODER_ANALOG,
                        help='Convert non-LRTT linear layers to frozen analog (default: digital)')
    parser.add_argument('--head-analog', action='store_true', default=HEAD_ANALOG,
                        help='Convert classifier to frozen analog (default: digital)')
    parser.add_argument('--backward-out-bound', type=float, default=BACKWARD_OUT_BOUND,
                        help=f'Backward pass output bound (default: {BACKWARD_OUT_BOUND})')
    parser.add_argument('--no-learn-out-scaling', action='store_true',
                        help='Disable trainable out_scaling on C tile')
    parser.add_argument('--auto-scale-mode', type=str, default='none',
                        choices=['none', 'shared', 'separate'],
                        help='Auto-scale mode for A/B LR normalization (default: none)')
    parser.add_argument('--correct-gradient-magnitudes', action='store_true',
                        help='Correct transfer magnitude by dividing by effective A/B LR')
    parser.add_argument('--transfer-rank-schedule', type=str, default='all',
                        choices=['all', 'round_robin'],
                        help='Transfer rank schedule (default: all)')
    parser.add_argument('--transfer-ranks-per-step', type=int, default=1,
                        help='Number of ranks per transfer step in round_robin mode (default: 1)')
    parser.add_argument('--no-scale-transfer-lr', action='store_true',
                        help='Disable scaling transfer LR by SGD LR (default: scale enabled)')
    parser.add_argument('--ab-multilevel', action='store_true',
                        help='Sweep ab_multilevel (w_max-w_min = 2^multilevel * ab_dw_min, B init scales)')
    parser.add_argument('--ab-pulse-type', type=str, default='default',
                        choices=['default', 'none', 'none_with_device', 'stochastic_compressed', 'mean_count', 'deterministic_implicit'],
                        help='Pulse type for A/B tile updates (default: use RPUConfig default)')
    parser.add_argument('--fi-continuous-alpha', action='store_true',
                        help='Use transfer LR as forward-injection α (continuity condition)')
    parser.add_argument('--af-ratio', type=float, default=AF_RATIO,
                        help='Asymmetry factor for scaledideal: gamma scaled by this ratio (default: 1.0)')
    parser.add_argument('--unr', type=float, default=UNR,
                        help='Update noise ratio for scaledideal: dw_min_std scaled by this ratio (default: 1.0)')
    args = parser.parse_args()

    # Update global config
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_STEPS = args.grad_accum_steps
    N_EPOCHS = args.epochs
    STEP_LR_SIZE = args.step_lr_size
    STEP_LR_GAMMA = args.step_lr_gamma
    TRANSFER_METHOD = args.transfer_method
    AB_DEVICE = args.ab_device
    A_DEVICE = args.a_device
    B_DEVICE = args.b_device
    C_DEVICE = args.c_device
    IO_NOISE = not args.no_io_noise
    FORWARD_INJECT = args.forward_inject
    IS_PERFECT = args.is_perfect
    NO_QUANT = args.no_quant
    LORA_TARGET = args.lora_target
    HEAD_LAYER = args.head_layer
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['reinit_mode'] = args.reinit_mode
    OPT_CONFIG['tune_wd'] = not args.no_wd
    OPT_CONFIG['tune_momentum'] = not args.no_momentum
    OPT_CONFIG['tune_nesterov'] = not args.no_nesterov
    OPT_CONFIG['no_transfer'] = args.no_transfer
    OPT_CONFIG['no_adc_ab_proj'] = args.no_adc_ab_proj
    OPT_CONFIG['learn_out_scaling'] = not args.no_learn_out_scaling
    OPT_CONFIG['auto_scale_mode'] = args.auto_scale_mode
    OPT_CONFIG['correct_gradient_magnitudes'] = args.correct_gradient_magnitudes
    OPT_CONFIG['transfer_rank_schedule'] = args.transfer_rank_schedule
    OPT_CONFIG['transfer_ranks_per_step'] = args.transfer_ranks_per_step
    OPT_CONFIG['scale_transfer_lr'] = not args.no_scale_transfer_lr
    OPT_CONFIG['fi_continuous_alpha'] = args.fi_continuous_alpha
    OPT_CONFIG['ab_pulse_type'] = args.ab_pulse_type
    OPT_CONFIG['ab_multilevel'] = args.ab_multilevel
    ENCODER_ANALOG = args.encoder_analog
    HEAD_ANALOG = args.head_analog
    BACKWARD_OUT_BOUND = args.backward_out_bound
    AF_RATIO = args.af_ratio
    UNR = args.unr

    # Auto-generate study name based on config (includes batch size)
    study_name = args.study_name or f"mlp_mnist_lrtt_bs{BATCH_SIZE}_{get_study_name_suffix()}"

    print(f"\n{'='*70}")
    print(f"  Study: {study_name}")
    print(f"  Log  : {RESULTS}/optuna_{study_name}.log")
    print(f"{'='*70}\n")

    storage = JournalStorage(JournalFileBackend(f"{RESULTS}/optuna_{study_name}.log"))

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print_study_summary(study)
        visualize_study(study, RESULTS)
        return

    # Check for OOM retry file (from previous OOM restart)
    # Multi-worker safe: each worker writes retry file with its PID, reads via glob.
    # On restart (execv→Popen), PID changes, so glob picks up any matching file.
    import glob as _glob
    retry_info = None
    retry_pattern = os.path.join(RESULTS, f"_oom_retry_{study_name}_*.json")
    retry_legacy = os.path.join(RESULTS, f"_oom_retry_{study_name}.json")
    for rf in _glob.glob(retry_pattern) + ([retry_legacy] if os.path.exists(retry_legacy) else []):
        try:
            with open(rf) as f:
                retry_info = json.load(f)
            os.remove(rf)
        except (json.JSONDecodeError, OSError):
            continue
        if retry_info:
            break
    if retry_info is not None:
        GRAD_ACCUM_STEPS = retry_info["grad_accum_steps"]
        _oom_retry_pending = True
        print(f"[OOM Retry] Retrying trial {retry_info['trial_number']}, "
              f"GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS}, micro_bs={BATCH_SIZE // GRAD_ACCUM_STEPS}")

    # Load data once (shared across all trials)
    train_loader, val_loader = load_data()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=ConfigAwareBoTorchSampler(n_startup_trials=10),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    # Warn if this is a brand-new study (likely means wrong flags were passed)
    existing_trials = study.trials
    n_complete = sum(1 for t in existing_trials if t.state == TrialState.COMPLETE)
    if not existing_trials:
        print(f"[WARNING] NEW STUDY created (no existing trials). "
              f"Verify the study name above matches your intent.\n")
    else:
        print(f"Loaded existing study: {len(existing_trials)} trials total, {n_complete} complete.\n")

    # Fix zombie RUNNING trials: mark them as FAIL so they don't block the queue.
    # In multi-worker setups, a hard crash (OOM kill, SIGKILL) can leave trials
    # stuck in RUNNING state forever. Only clean up if this is the FIRST worker
    # (no other workers are actively running this study). Check via a simple
    # file lock to avoid killing another worker's active trial.
    zombie_lock = os.path.join(RESULTS, f"_zombie_cleanup_{study_name}.lock")
    if not os.path.exists(zombie_lock):
        n_zombie = 0
        for t in existing_trials:
            if t.state == TrialState.RUNNING:
                study.tell(t.number, state=TrialState.FAIL)
                n_zombie += 1
        if n_zombie > 0:
            print(f"[Cleanup] Marked {n_zombie} zombie RUNNING trial(s) as FAIL.")
        # Create lock so subsequent workers skip cleanup
        with open(zombie_lock, 'w') as f:
            f.write(str(os.getpid()))

    # Enqueue retry trial if OOM retry pending
    if retry_info is not None:
        study.enqueue_trial(retry_info["trial_params"])

    print(f"Study: {study_name}, Device: {DEVICE}, New trials: {args.n_trials}")

    # Run trials with OOM recovery via process restart
    initial_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)

    def _restart_with_remaining(remaining):
        """execv to thin wrapper (no CUDA) that spawns child + forwards Ctrl+C."""
        new_argv = list(sys.argv)
        for i, arg in enumerate(new_argv):
            if arg == '--n-trials' and i + 1 < len(new_argv):
                new_argv[i + 1] = str(remaining)
                break
        child_cmd = [sys.executable] + new_argv
        wrapper = (
            'import subprocess,signal,sys\n'
            f'p=subprocess.Popen({child_cmd!r})\n'
            'def _fwd(s,f):\n'
            ' try: p.send_signal(s)\n'
            ' except: pass\n'
            'signal.signal(signal.SIGINT,_fwd)\n'
            'sys.exit(p.wait())\n'
        )
        os.execv(sys.executable, [sys.executable, '-c', wrapper])

    try:
        study.optimize(
            lambda trial: objective(trial, train_loader, val_loader),
            n_trials=args.n_trials,
            catch=(Exception,),
            show_progress_bar=False,
            callbacks=[_oom_restart_callback],
        )
    except _OOMRestart:
        current_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        remaining = max(1, args.n_trials - (current_complete - initial_complete))
        print(f"\n[OOM Recovery] Restarting process for {remaining} remaining trials...")
        _restart_with_remaining(remaining)
    except _OOMRetryDone:
        # Retry succeeded, restart to reset GRAD_ACCUM_STEPS to default
        current_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        remaining = args.n_trials - (current_complete - initial_complete)
        if remaining > 0:
            print(f"\n[OOM Recovery] Restarting with default GRAD_ACCUM for {remaining} remaining trials...")
            _restart_with_remaining(remaining)

    print_study_summary(study)
    visualize_study(study, RESULTS)

    # Save best params
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if complete_trials:
        best_params_file = os.path.join(RESULTS, f"best_params_{study_name}.json")
        with open(best_params_file, 'w') as f:
            json.dump({
                "best_f1": study.best_value,
                "best_params": study.best_params,
            }, f, indent=2)
        print(f"Best params saved to: {best_params_file}")

    # Save all trials
    all_trials = []
    for t in study.trials:
        all_trials.append({
            "trial": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        })
    all_trials.sort(key=lambda x: x["value"] if x["value"] is not None else -1, reverse=True)

    all_trials_file = os.path.join(RESULTS, "all_trials.json")
    with open(all_trials_file, 'w') as f:
        json.dump(all_trials, f, indent=2)
    print(f"All trials saved to: {all_trials_file}")


class _OOMRestart(BaseException):
    """Raised to trigger process restart after OOM.

    Inherits from BaseException (not Exception) so that
    study.optimize(catch=(Exception,)) does not swallow it.
    """
    pass


class _OOMRetryDone(BaseException):
    """Raised after OOM retry trial to restart with default GRAD_ACCUM_STEPS.

    Inherits from BaseException (not Exception) so that
    study.optimize(catch=(Exception,)) does not swallow it.
    """
    pass


_oom_retry_pending = False


def _oom_restart_callback(study, trial):
    """Optuna callback: on OOM, save retry file and restart process."""
    global _oom_retry_pending

    if trial.state == TrialState.FAIL:
        err = trial.user_attrs.get("error", "")
        is_oom = any(k in err.lower() for k in ("out of memory", "cublas", "cudacachingallocator"))
        is_cuda_assert = any(k in err.lower() for k in ("nvml", "internal assert failed")) and not is_oom
        if is_oom:
            # Pick the next divisor of BATCH_SIZE larger than current GRAD_ACCUM_STEPS
            # so that micro_bs = BATCH_SIZE // new_grad_accum is always exact.
            divisors = sorted(d for d in range(1, BATCH_SIZE + 1) if BATCH_SIZE % d == 0)
            larger = [d for d in divisors if d > GRAD_ACCUM_STEPS]
            if not larger:
                print(f"\n[OOM Recovery] Cannot reduce micro-batch below 1 "
                      f"(BATCH_SIZE={BATCH_SIZE}, already at max GRAD_ACCUM={GRAD_ACCUM_STEPS}). "
                      f"Skipping retry.")
                return
            new_grad_accum = larger[0]
            micro_bs = BATCH_SIZE // new_grad_accum

            retry_file = os.path.join(RESULTS, f"_oom_retry_{study.study_name}_{os.getpid()}.json")
            retry_info = {
                "trial_params": dict(trial.params),
                "trial_number": trial.number,
                "grad_accum_steps": new_grad_accum,
            }
            with open(retry_file, 'w') as f:
                json.dump(retry_info, f, indent=2)

            print(f"\n[OOM Recovery] Trial {trial.number} OOM. "
                  f"Will restart with GRAD_ACCUM_STEPS={new_grad_accum}, micro_bs={micro_bs}.")
            raise _OOMRestart()
        elif is_cuda_assert:
            # CUDA context corruption (not OOM): retry same trial in a fresh process
            # with the same ga (no bump) so the trial gets a clean CUDA context.
            retry_file = os.path.join(RESULTS, f"_oom_retry_{study.study_name}_{os.getpid()}.json")
            retry_info = {
                "trial_params": dict(trial.params),
                "trial_number": trial.number,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
            }
            with open(retry_file, 'w') as f:
                json.dump(retry_info, f, indent=2)
            print(f"\n[CUDA Recovery] Trial {trial.number} CUDA context error. "
                  f"Retrying in fresh process with ga={GRAD_ACCUM_STEPS} (no ga change).")
            raise _OOMRestart()

    # After retry trial completes, restart to reset GRAD_ACCUM_STEPS to default
    if _oom_retry_pending:
        _oom_retry_pending = False
        print(f"\n[OOM Recovery] Retry trial {trial.number} done (state={trial.state.name}). "
              f"Restarting with default GRAD_ACCUM.")
        raise _OOMRetryDone()


if __name__ == "__main__":
    main()
