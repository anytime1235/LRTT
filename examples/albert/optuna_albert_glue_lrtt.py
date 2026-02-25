# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for ALBERT + GLUE with LRTT.

Supported GLUE tasks: cola, sst2, mrpc, qqp, mnli, qnli, rte, stsb, wnli

Usage:
    python optuna_albert_glue_lrtt.py --task sst2 --n-trials 50
    python optuna_albert_glue_lrtt.py --task mrpc --n-trials 50
    python optuna_albert_glue_lrtt.py --task sst2 --visualize
    python optuna_albert_glue_lrtt.py --task sst2 --n-trials 50 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 64 --epochs 5 --warmup-steps 393 --transfer-method set --no-io-noise --lora-target qkv
    python optuna_albert_glue_lrtt.py --task cola --n-trials 150 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 16 --epochs 20 --warmup-steps 534 --transfer-method set --no-io-noise --lora-target all
    python optuna_albert_glue_lrtt.py --task stsb --n-trials 150 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 16 --epochs 20 --warmup-steps 360 --transfer-method set --no-io-noise --lora-target all
    python optuna_albert_glue_lrtt.py --task cola --n-trials 50 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 16 --epochs 20 --warmup-steps 534 --transfer-method set --no-io-noise --lora-target none --encoder-analog

   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task cola --n-trials 150 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 16 --epochs 20 --warmup-steps 534 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog  --lora-target qkvo --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task cola --n-trials 50 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 16 --epochs 20 --warmup-steps 534 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog --lora-target none --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task stsb --n-trials 150 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 16 --epochs 20 --warmup-steps 360 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog --lora-target qkvo --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task stsb --n-trials 50 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 16 --epochs 20 --warmup-steps 360 --transfer-method set --encoder-analog --embedding-analog --head-analog  --no-io-noise --lora-target none --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task sst2 --n-trials 150 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 20 --warmup-steps 2094 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog --lora-target qkvo --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task sst2 --n-trials 50 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 20 --warmup-steps 2094 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog --lora-target none --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task mnli --n-trials 150 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 128 --epochs 4 --warmup-steps 1000 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog --lora-target qkvo --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task mnli --n-trials 50 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 128 --epochs 4 --warmup-steps 1000 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog --lora-target none --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task qnli --n-trials 150 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 21 --warmup-steps 3312 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog --lora-target qkvo --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task qnli --n-trials 50 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 21 --warmup-steps 3312 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog --lora-target none --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task qqp --n-trials 150 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 128 --epochs 10 --warmup-steps 1400 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog --lora-target qkvo --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task qqp --n-trials 50 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 128 --epochs 10 --warmup-steps 1400 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog --lora-target none --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task rte --n-trials 150 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 21 --warmup-steps 80 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog --lora-target qkvo --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task rte --n-trials 50 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 21 --warmup-steps 80 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog --lora-target none --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task mrpc --n-trials 150 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 14 --warmup-steps 80 --transfer-method set --no-io-noise--encoder-analog --embedding-analog --head-analog --lora-target qkvo --no-learn-out-scaling
   HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task mrpc --n-trials 50 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 14 --warmup-steps 80 --transfer-method set --no-io-noise --encoder-analog --embedding-analog --head-analog --lora-target none --no-learn-out-scaling

HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task rte --n-trials 150 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 21 --warmup-steps 80 --transfer-method set --no-io-noise --lora-target qkvo
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task rte --n-trials 50 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 21 --warmup-steps 80 --transfer-method set --no-io-noise --lora-target qkvo --no-transfer
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task mrpc --n-trials 150 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 14 --warmup-steps 80 --transfer-method set --no-io-noise --lora-target qkvo
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task mrpc --n-trials 50 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 14 --warmup-steps 80 --transfer-method set --no-io-noise --lora-target qkvo --no-transfer
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task stsb --n-trials 150 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 16 --epochs 20 --warmup-steps 360 --transfer-method set --no-io-noise --lora-target qkvo
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task stsb --n-trials 50 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 16 --epochs 20 --warmup-steps 360 --transfer-method set --no-io-noise --lora-target qkvo --no-transfer
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task cola --n-trials 150 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 16 --epochs 20 --warmup-steps 534 --transfer-method set --no-io-noise --lora-target qkvo
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task cola --n-trials 50 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 16 --epochs 20 --warmup-steps 534 --transfer-method set --no-io-noise --lora-target qkvo --no-transfer
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task sst2 --n-trials 150 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 20 --warmup-steps 2094 --transfer-method set --no-io-noise --lora-target qkvo
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task sst2 --n-trials 50 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 20 --warmup-steps 2094 --transfer-method set --no-io-noise --lora-target qkvo --no-transfer
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task qnli --n-trials 150 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 21 --warmup-steps 3312 --transfer-method set --no-io-noise --lora-target qkvo
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task qnli --n-trials 50 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 32 --epochs 21 --warmup-steps 3312 --transfer-method set --no-io-noise --lora-target qkvo --no-transfer
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task qqp --n-trials 150 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 128 --epochs 10 --warmup-steps 1400 --transfer-method set --no-io-noise --lora-target qkvo
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task qqp --n-trials 50 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 128 --epochs 10 --warmup-steps 1400 --transfer-method set --no-io-noise --lora-target qkvo --no-transfer
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task mnli --n-trials 150 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 128 --epochs 4 --warmup-steps 1000 --transfer-method set --no-io-noise --lora-target qkvo
HF_HUB_DISABLE_XET=1 python optuna_albert_glue_lrtt.py --task mnli --n-trials 50 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 128 --epochs 4 --warmup-steps 1000 --transfer-method set --no-io-noise --lora-target qkvo --no-transfer


All flags:
    python optuna_albert_glue_lrtt.py \
        --task <str>                # GLUE task (default: sst2)
        --study-name <str>          # Study name (default: auto-generated)
        --n-trials <int>            # Number of Optuna trials (default: 50)
        --visualize                 # Visualize study results and exit
        --optimizer <str>           # AnalogSGD | AnalogAdam (default: AnalogSGD)
        --no-wd                     # Disable weight decay tuning (fix to 0)
        --no-momentum               # Disable momentum tuning (fix to 0, SGD only)
        --no-nesterov               # Disable nesterov tuning (fix to False, SGD only)
        --reinit-mode <str>         # Fix reinit mode: standard | decay | hybrid (default: tune all)
        --batch-size <int>          # Batch size (default: 64)
        --epochs <int>              # Number of epochs (default: 15)
        --warmup-steps <int>        # LR warmup steps (default: 189)
        --transfer-method <str>     # Transfer method: onehot | direct | set (default: onehot)
        --ab-device <str>           # A/B tile device: 6t1c | fp (default: 6t1c)
        --no-io-noise               # Disable IO out_noise (resolution kept)
        --lora-target <str>         # LoRA target: none | qonly | konly | vonly | qkv | qkvo | ffn | all (default: qkv)
        --head-layer <str>          # classifier: train | freeze (default: train)
        --no-transfer               # Disable LRTT transfer (A/B frozen, skip LRTT param sweep)
        --no-adc-ab-proj            # Remove ADC/DAC between A/B projections (full precision)
        --no-learn-out-scaling      # Disable trainable out_scaling on C tile
        --encoder-analog            # Non-LRTT encoder layers: frozen analog instead of digital
        --embedding-analog          # Embedding projection: frozen analog instead of digital
        --head-analog               # Classifier/qa_outputs: frozen analog instead of digital
        --backward-out-bound <float> # Backward pass output bound (default: 12.0)
        --auto-scale-mode <str>     # Auto-scale: none | shared | separate (default: none)


Inline flags (edit directly in script):
    DYNAMIC_TE = False              # Enable dynamic transfer every
    DYNAMIC_TE_POWER = 1.0          # Power for dynamic TE scaling
    TE_WARMUP_STEPS = 0            # Steps before reaching target TE
    TE_WARMUP_SCHEDULE = []         # Warmup TE schedule list
    REINIT_GAIN = 1.0               # Reinitialization gain
    DECAY_FACTOR = 1.0              # Decay factor for reinit
    TARGET_MODULES = [...]          # Modules to convert to analog
    TRAIN_SUBSET_SIZE = 0           # Training data subset (0 = full)
    EVAL_SUBSET_SIZE = 0            # Evaluation data subset (0 = full)

Enqueue example:
python3 << 'EOF'
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
study = optuna.load_study(
    study_name='albert_sst2_lrtt_bs64_sgd_hybrid_nowd_nomom_nonest_set_noio_qkv',
    storage=JournalStorage(JournalFileBackend('results/optuna_albert_sst2_lrtt/optuna_albert_sst2_lrtt_bs64_sgd_hybrid_nowd_nomom_nonest_set_noio_qkv.log')))
study.enqueue_trial({
    'learning_rate': 0.2080749864869466,
    'transfer_lr': 0.010000000000000004,
    'transfer_every': 16210,
    'rank_exp': 1,
    'fast_lr': 0.41139594231202437,
    'tau_sec': 0.0,
    'min_lr_rate': 0.0})
print('Enqueued!')
EOF

"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import math
import json
import argparse
import gc

import torch
from torch import nn, no_grad, manual_seed
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

import optuna
from optuna.trial import TrialState
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna_integration import BoTorchSampler
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lrtt_grad_accum_patch  # noqa: F401  — per-micro-batch tile.update + LRTT A/B snapshot

from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice, FloatingPointDevice
from aihwkit.simulator.configs import SingleRPUConfig

# LRTT config imports (direct imports to avoid __init__.py dependency issues)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.parameters.mapping import MappingParameter


# =============================================================================
# GLUE Task Configurations
# =============================================================================

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

TASK_TO_NUM_LABELS = {
    "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
    "mnli": 3, "qnli": 2, "rte": 2, "stsb": 1, "wnli": 2,
}

TASK_TO_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "f1",
    "qqp": "f1",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "stsb": "spearmanr",
    "wnli": "accuracy",
}

TASK_TO_MAX_SEQ_LENGTH = {
    "cola": 128, "sst2": 128, "mrpc": 128, "qqp": 128,
    "mnli": 128, "qnli": 128, "rte": 256, "stsb": 128, "wnli": 128,
}


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

# GLUE task (will be set by argparse)
TASK_NAME = "sst2"

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
RESULTS = os.path.join(os.getcwd(), "results", "optuna_albert_glue_lrtt")  # Updated per-task in main()

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "albert/albert-base-v2"
MAX_SEQ_LENGTH = 128  # Updated per-task in main()
NUM_LABELS = 2  # Will be set dynamically based on TASK_NAME

# Training defaults
N_EPOCHS = 15
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
EVAL_BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 3
VAL_LOSS_EARLY_STOP_PATIENCE = 2  # Stop if val loss doesn't improve for this many epochs
VAL_LOSS_THRESHOLD = 1.5  # Once val loss drops below this, rely on metric-based early stop only

# Scheduler
WARMUP_STEPS = 500  # warmup steps

# Dynamic TE
DYNAMIC_TE = False
DYNAMIC_TE_POWER = 1.0
TE_WARMUP_STEPS = 0
TE_WARMUP_SCHEDULE = []

# Fixed LRTT parameters
REINIT_GAIN = 1.0
DECAY_FACTOR = 1.0
TRANSFER_METHOD = "onehot"  # "onehot", "direct", or "set"
AB_DEVICE = "6t1c"  # "6t1c" or "fp"
IO_NOISE = True  # If False, disable out_noise (resolution kept)
ENCODER_ANALOG = False  # If True, non-LRTT encoder layers become frozen analog instead of digital
EMBEDDING_ANALOG = False  # If True, embedding projection → frozen analog instead of digital
HEAD_ANALOG = False  # If True, classifier → frozen analog instead of digital
BACKWARD_OUT_BOUND = 12.0  # Backward pass output bound (default 12.0)

# LoRA target options: which layers have trainable A/B tiles
# - none: no LRTT layers (fully digital baseline)
# - qkv: only query, key, value
# - ffn: attention.dense + ffn + ffn_output
# - all: all encoder linear layers
LORA_TARGET = "qkv"  # default, can be set via --lora-target
HEAD_LAYER = "train"  # default, can be set via --head-layer (train | freeze)
LORA_TARGET_MODULES = {
    "none": [],  # Empty = no layers converted to LRTT (fully digital)
    "qonly": ["query"],  # Query only (1 shared layer)
    "konly": ["key"],  # Key only (1 shared layer)
    "vonly": ["value"],  # Value only (1 shared layer)
    "qkv": ["query", "key", "value"],  # Q/K/V (3 shared layers)
    "qkvo": ["query", "key", "value", "attention.dense"],  # Q/K/V + attention output (4 shared layers)
    "ffn": ["ffn"],  # FFN layers only: ffn + ffn_output (2 in shared group)
    "all": None,  # None means all encoder layers (no filtering) (~6 shared layers)
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
    'auto_scale_mode': 'none',  # Auto-scale mode for A/B LR normalization
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

    # Add transfer method if not default
    if TRANSFER_METHOD != "onehot":
        suffix += f"_{TRANSFER_METHOD}"

    if AB_DEVICE != "6t1c":
        suffix += f"_{AB_DEVICE.replace('-', '')}"

    if not IO_NOISE:
        suffix += "_noio"

    if OPT_CONFIG['no_transfer']:
        suffix += "_notrans"

    if OPT_CONFIG.get('no_adc_ab_proj', False):
        suffix += "_noadc"

    if not OPT_CONFIG.get('learn_out_scaling', True):
        suffix += "_noos"

    if ENCODER_ANALOG:
        suffix += "_encanalog"

    if EMBEDDING_ANALOG:
        suffix += "_embedanalog"
    if HEAD_ANALOG:
        suffix += "_headanalog"

    if BACKWARD_OUT_BOUND != 12.0:
        suffix += f"_bob{BACKWARD_OUT_BOUND:g}"

    if OPT_CONFIG.get('auto_scale_mode', 'none') != 'none':
        suffix += f"_as-{OPT_CONFIG['auto_scale_mode']}"

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

def _create_ab_device(tau_sec=0.0):
    """Create A/B tile device based on AB_DEVICE setting.

    Options:
        6t1c - Full 6T1C with all noise/variation (realistic)
        fp   - FloatingPointDevice (perfect, no quantization/bounds)

    Args:
        tau_sec: Retention time constant. If 0, lifetime=0 (no decay).
    """
    if AB_DEVICE == "fp":
        return FloatingPointDevice()

    # Compute retention lifetime from tau_sec
    if tau_sec > 0:
        dt_batch_sec = 1.0
        delta = 1 - math.exp(-dt_batch_sec / tau_sec)
        lifetime = 1.0 / delta if delta > 0 else 0.0
    else:
        lifetime = 0.0

    # Default: 6t1c (full noise)
    return LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
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
        lifetime_dtod=0.0,
        reset=0.0,
        reset_dtod=0.0,
    )


def _create_c_device(dw_min=0.001):
    """Create noise-free SoftBoundsDevice for C tile."""
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
        mult_noise=False,  # No multiplicative noise for C tile
    )


def create_frozen_analog_config(lrtt_config=None, out_noise=0.0):
    """Create analog config for non-LRTT encoder layers (frozen analog).

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
        if BACKWARD_OUT_BOUND != 12.0:
            rpu_config.backward.out_bound = BACKWARD_OUT_BOUND
    return rpu_config


def create_lrtt_config(rank, transfer_every, transfer_lr, fast_lr, reinit_mode, tau_sec=0.0,
                       c_dw_min=0.001, c_desired_bl=None, out_noise=0.0, ab_weight_scaling_omega=0.0,
                       auto_scale_mode='none'):
    """Create LRTT RPU configuration for analog layers."""
    ab_device = _create_ab_device(tau_sec=tau_sec)
    c_device = _create_c_device(dw_min=c_dw_min)

    te = transfer_every
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=1.0,
        fast_lr=fast_lr,
        reinit_gain=REINIT_GAIN,
        reinit_mode=reinit_mode,
        decay_factor=DECAY_FACTOR,
        unit_cell_devices=[ab_device, ab_device, c_device],
        train_c_bias=False,        # C tile bias frozen
        mapping_ab=MappingParameter(
            weight_scaling_omega=ab_weight_scaling_omega,
            learn_out_scaling=False,
        ),
        mapping_c=MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=True,
            learn_out_scaling=OPT_CONFIG.get('learn_out_scaling', True),
            out_scaling_columnwise=True,
        ),
    )
    device_config.transfer_lr = transfer_lr
    device_config.units_in_mbatch = True
    device_config.transfer_method = TRANSFER_METHOD
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = False
    device_config.no_adc_ab_projection = OPT_CONFIG.get('no_adc_ab_proj', False)
    device_config.auto_scale_mode = auto_scale_mode
    if c_desired_bl is not None:
        device_config.c_desired_bl = c_desired_bl

    # Dynamic TE
    device_config.dynamic_te = DYNAMIC_TE
    device_config.dynamic_te_power = DYNAMIC_TE_POWER
    device_config.dynamic_te_max = te * 20
    device_config.te_warmup_schedule = TE_WARMUP_SCHEDULE + [te]
    device_config.te_warmup_steps = TE_WARMUP_STEPS

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    rpu_config.forward.out_noise = out_noise
    rpu_config.backward.out_noise = out_noise

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

    Returns list of substrings that identify which encoder layers should be LRTT.
    Returns [] for none mode (fully digital, no LRTT layers).
    """
    if lora_target == "none":
        return []  # Empty = no layers converted to LRTT (fully digital baseline)
    elif lora_target == "qonly":
        return ["query"]  # Query only (1 shared layer)
    elif lora_target == "konly":
        return ["key"]  # Key only (1 shared layer)
    elif lora_target == "vonly":
        return ["value"]  # Value only (1 shared layer)
    elif lora_target == "qkv":
        return ["query", "key", "value"]  # Q/K/V (3 shared layers)
    elif lora_target == "qkvo":
        return ["query", "key", "value", "attention.dense"]  # Q/K/V + attention output (4 shared layers)
    elif lora_target == "ffn":
        return ["ffn"]  # ffn + ffn_output (2 shared layers)
    elif lora_target == "all":
        # All encoder linear layers (exclude embeddings, classifier, embedding_hidden_mapping_in)
        return None  # None means all encoder layers (~6 shared layers)
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


def create_model(params):
    """Create ALBERT classification model with selective LRTT analog layers.

    Architecture (follows paper's approach for efficiency):
        - LRTT Target layers (based on --lora-target) → LRTT Analog
        - Non-target Encoder layers → Digital FROZEN
        - classifier → Digital TRAINABLE (weight + bias)
        - embedding_hidden_mapping_in → Digital FROZEN
        - pooler → Digital FROZEN
        - Embeddings → Digital FROZEN

    LoRA Target Options (--lora-target):
        - qkv: Q/K/V layers → LRTT Analog (72 layers)
        - ffn: projection + FFN layers → LRTT Analog (288 layers)
        - all: all encoder layers → LRTT Analog (360 layers)

    LRTT layers have:
        - A/B tiles: TRAINABLE
        - C-tile: FROZEN (pretrained weights)
        - out_scaling: TRAINABLE
        - bias: FROZEN
    """
    from aihwkit.nn import AnalogLinear

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    # Get LRTT target patterns
    lrtt_patterns = get_lrtt_target_module_names(LORA_TARGET)

    def is_lrtt_target(layer_name):
        """Check if layer should be converted to LRTT Analog."""
        # classifier is always digital
        if "classifier" in layer_name:
            return False
        # embedding_hidden_mapping_in: always digital frozen (ALBERT's embedding projection)
        if "embedding_hidden_mapping_in" in layer_name:
            return False
        # pooler: always digital frozen
        if "pooler" in layer_name:
            return False
        # Must be in encoder for other layers
        if "encoder" not in layer_name:
            return False
        # If lrtt_patterns is None (all mode), all encoder layers are targets
        if lrtt_patterns is None:
            return True
        return any(p in layer_name for p in lrtt_patterns)

    # Build exclude list: all layers that should NOT be converted to LRTT
    all_linear_names = list_linear_layers(model)
    exclude_modules = []
    for name in all_linear_names:
        if not is_lrtt_target(name):
            # Use full path for exclude_modules (convert_to_analog requires exact match)
            exclude_modules.append(name)

    # Exclude classifier, embedding_hidden_mapping_in, and pooler (always digital)
    exclude_modules.append("classifier")
    exclude_modules.append("albert.encoder.embedding_hidden_mapping_in")
    exclude_modules.append("albert.pooler")
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
            tau_sec=params["tau_sec"],
            c_dw_min=params["c_dw_min"],
            c_desired_bl=params["c_desired_bl"],
            out_noise=params["out_noise"],
            ab_weight_scaling_omega=params["ab_weight_scaling_omega"],
            auto_scale_mode=OPT_CONFIG['auto_scale_mode'],
        )

        # Convert to analog with exclusions (only LRTT targets get converted)
        model = convert_to_analog(model, lrtt_config, exclude_modules=exclude_modules)

        # Count analog layers
        analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))

    # Step 1.5: Convert remaining encoder layers to frozen analog (if enabled)
    # Already-converted LRTT layers (AnalogLinear) are naturally skipped by convert_to_analog
    frozen_analog_count = 0
    any_frozen_analog = (ENCODER_ANALOG and LORA_TARGET != "all") or EMBEDDING_ANALOG or HEAD_ANALOG
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
        frozen_exclude = ["albert.pooler"]
        if not EMBEDDING_ANALOG:
            frozen_exclude.append("albert.encoder.embedding_hidden_mapping_in")
        if not HEAD_ANALOG:
            frozen_exclude.append("classifier")
        if not ENCODER_ANALOG or LORA_TARGET == "all":
            for name in all_linear_names:
                if "encoder" in name and "embedding_hidden_mapping_in" not in name:
                    frozen_exclude.append(name)
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
    # - LRTT layers: A/B + out_scaling TRAINABLE, C + bias FROZEN
    # - Frozen analog: out_scaling TRAINABLE (same as C tile), weights FROZEN
    # - classifier: TRAINABLE if HEAD_LAYER=="train", else FROZEN
    # - embedding_hidden_mapping_in: always digital frozen
    # - pooler: always digital frozen
    # - Everything else: FROZEN
    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            # LRTT A/B tiles: TRAINABLE, but FROZEN if no_transfer (no point updating without transfer)
            param.requires_grad = not OPT_CONFIG['no_transfer']
        elif "tile_c" in name:
            pass  # Respect lrtt_tile.py settings (train_c_bias, mapping_c)
        elif "out_scaling_alpha" in name:
            pass  # Frozen analog out_scaling: TRAINABLE (same as C tile)
        elif "classifier" in name:
            # classifier: TRAINABLE or FROZEN based on setting
            param.requires_grad = (HEAD_LAYER == "train")
        elif "embedding_hidden_mapping_in" in name:
            param.requires_grad = False
        elif "pooler" in name:
            param.requires_grad = False
        elif "LayerNorm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ln_params = sum(p.numel() for n, p in model.named_parameters() if "LayerNorm" in n and p.requires_grad)
    print(f"  Trainable (after grad set): {trainable_after:,} (LayerNorm: {ln_params:,})")
    print(f"  LoRA target: {LORA_TARGET} -> {lrtt_patterns if lrtt_patterns else 'all encoder layers'}")

    try:
        return model.to(DEVICE)
    except Exception:
        # If .to(DEVICE) fails (e.g. CUBLAS/OOM), partially-transferred model
        # must be explicitly deleted to free GPU memory from tiles that DID transfer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


# =============================================================================
# Data Functions
# =============================================================================

def load_data(tokenizer):
    """Load and tokenize GLUE dataset for the specified task."""
    raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)
    sentence1_key, sentence2_key = TASK_TO_KEYS[TASK_NAME]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                padding=False,
            )
        return tokenizer(
            examples[sentence1_key], examples[sentence2_key],
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=False,
        )

    # Tokenize all splits
    cols_to_remove = [c for c in raw_datasets["train"].column_names if c != "label"]
    tokenized = raw_datasets.map(preprocess_function, batched=True, remove_columns=cols_to_remove)
    tokenized = tokenized.rename_column("label", "labels")

    # Training set
    train_dataset = tokenized["train"]
    if TRAIN_SUBSET_SIZE > 0:
        train_dataset = train_dataset.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(train_dataset)))
        )
    else:
        train_dataset = train_dataset.shuffle(seed=SEED)

    # Eval set (MNLI uses validation_matched)
    eval_key = "validation_matched" if TASK_NAME == "mnli" else "validation"
    eval_dataset = tokenized[eval_key]
    if EVAL_SUBSET_SIZE > 0:
        eval_dataset = eval_dataset.select(
            range(min(EVAL_SUBSET_SIZE, len(eval_dataset)))
        )

    collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE // GRAD_ACCUM_STEPS, shuffle=True,
        collate_fn=collator, num_workers=2,
        generator=torch.Generator().manual_seed(SEED)
    )

    eval_loader = DataLoader(
        eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=collator,
        num_workers=2
    )

    return train_loader, eval_loader


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_model(model, eval_loader):
    """Evaluate GLUE model. Returns (metric_value, val_loss)."""
    model.eval()

    is_regression = (TASK_NAME == "stsb")
    all_preds = []
    all_labels = []
    total_val_loss = 0.0
    num_val_batches = 0

    with no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            num_val_batches += 1

            if is_regression:
                preds = outputs.logits.squeeze()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.float().cpu().numpy())
            else:
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    model.train()

    val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')

    # Compute task-specific metric
    metric_name = TASK_TO_METRIC[TASK_NAME]
    if metric_name == "accuracy":
        metric_value = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100.0
    elif metric_name == "f1":
        from sklearn.metrics import f1_score
        metric_value = f1_score(all_labels, all_preds) * 100.0
    elif metric_name == "matthews_correlation":
        from sklearn.metrics import matthews_corrcoef
        metric_value = matthews_corrcoef(all_labels, all_preds) * 100.0
    elif metric_name == "spearmanr":
        from scipy.stats import spearmanr
        metric_value = spearmanr(all_preds, all_labels)[0] * 100.0

    return metric_value, val_loss


# =============================================================================
# Scheduler
# =============================================================================

def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_rate=0.0):
    """Linear schedule with warmup that decays to min_lr_rate (fraction of peak LR)."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Optuna Objective
# =============================================================================

def objective(trial, train_loader, eval_loader, tokenizer):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)

    # LRTT parameters: skip sweep if --no-transfer (A/B frozen, no transfer happens)
    if OPT_CONFIG['no_transfer']:
        transfer_lr = 0.1        # fixed (not used anyway)
        transfer_every = 999999999
        rank_exp = 2             # fixed (A=0 init, no effect)
        rank = 4
        fast_lr = 1.0            # fixed (no effect)
        tau_sec = 0.0            # fixed
    else:
        transfer_lr = trial.suggest_float('transfer_lr', 1e-5, 1e1, log=True)
        transfer_every = trial.suggest_int('transfer_every', 1, 500, log=True)
        rank_exp = trial.suggest_int('rank_exp', 0, 7)
        rank = 2 ** rank_exp
        fast_lr = trial.suggest_float('fast_lr', 1e-5, 1e1, log=True)
        tau_sec = trial.suggest_float('tau_sec', 0, 0, log=False)  # 0 = no decay

    # C tile pulsed transfer params (only meaningful for onehot/direct)
    if TRANSFER_METHOD in ("onehot", "direct") and not OPT_CONFIG['no_transfer']:
        c_dw_min = trial.suggest_float('c_dw_min', 0.001, 0.001)
        c_desired_bl = trial.suggest_int('c_desired_bl', 31, 31)
    else:
        c_dw_min = 0.001   # default (unused for "set")
        c_desired_bl = None

    # IO / mapping params
    out_noise = trial.suggest_float('out_noise', 0.0, 0.0)
    ab_weight_scaling_omega = trial.suggest_float('ab_weight_scaling_omega', 0.0, 0.0)

    min_lr_rate = trial.suggest_float('min_lr_rate', 0.0, 0.0)

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
        "tau_sec": tau_sec,
        "c_dw_min": c_dw_min,
        "c_desired_bl": c_desired_bl,
        "out_noise": out_noise,
        "ab_weight_scaling_omega": ab_weight_scaling_omega,
    }

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting")
    print(f"{'='*70}")
    print(f"  rank={rank}, transfer_every={transfer_every}, transfer_lr={transfer_lr:.4e}")
    print(f"  fast_lr={fast_lr:.2e}, lr={learning_rate:.2e}, wd={weight_decay:.2e}")
    print(f"  momentum={momentum:.2f}, nesterov={nesterov}, reinit_mode={reinit_mode}")
    print(f"  tau_sec={tau_sec:.1f}, optimizer={optimizer_name}, min_lr_rate={min_lr_rate:.4f}")
    if TRANSFER_METHOD in ("onehot", "direct") and not OPT_CONFIG['no_transfer']:
        print(f"  c_dw_min={c_dw_min:.4e}, c_desired_bl={c_desired_bl}")
    print(f"{'='*70}")

    model = None
    try:
        set_seed(SEED)

        model = create_model(params)

        if LORA_TARGET == "none" and not ENCODER_ANALOG and not EMBEDDING_ANALOG and not HEAD_ANALOG:
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

        num_training_steps = len(train_loader) * N_EPOCHS // GRAD_ACCUM_STEPS
        scheduler = get_linear_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=num_training_steps,
            min_lr_rate=min_lr_rate,
        )

        best_acc = 0.0
        epochs_without_improvement = 0
        best_val_loss = float('inf')
        val_loss_no_improvement = 0
        val_loss_crossed_threshold = False  # True once val loss drops below threshold

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
            optimizer.zero_grad()
            for micro_step, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()

                if (micro_step + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * GRAD_ACCUM_STEPS
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")

            train_loss = total_loss / num_batches if num_batches > 0 else 0.0

            eval_acc, val_loss = evaluate_model(model, eval_loader)

            improved = ""
            if eval_acc > best_acc:
                best_acc = eval_acc
                epochs_without_improvement = 0
                improved = " ★"
            else:
                epochs_without_improvement += 1

            val_loss_improved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_loss_no_improvement = 0
                val_loss_improved = " ↓"
            else:
                val_loss_no_improvement += 1

            # Reset metric patience when val loss first crosses threshold
            if not val_loss_crossed_threshold and best_val_loss <= VAL_LOSS_THRESHOLD:
                val_loss_crossed_threshold = True
                epochs_without_improvement = 0

            metric_name = TASK_TO_METRIC[TASK_NAME]
            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"[Trial {trial.number}] Epoch {epoch:3d} | "
                      f"{metric_name}: {eval_acc:6.2f}% | Best: {best_acc:6.2f}% | "
                      f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}{val_loss_improved} | LR: {current_lr:.2e} | "
                      f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}{improved}")

            trial.report(best_acc, epoch)
            trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)
            trial.set_user_attr(f"val_loss_epoch_{epoch}", val_loss)

            if not val_loss_crossed_threshold and val_loss_no_improvement >= VAL_LOSS_EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Val loss early stop at epoch {epoch} "
                          f"(val_loss={val_loss:.4f} > {VAL_LOSS_THRESHOLD}, no improvement for {val_loss_no_improvement} epochs)")
                break

            if val_loss_crossed_threshold and epochs_without_improvement >= EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}")
                break

            if trial.should_prune():
                tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        tqdm.write(f"\n[Trial {trial.number}] Finished - Best {metric_name}: {best_acc:.2f}%")
        tqdm.write(f"{'='*70}\n")
        return best_acc

    except Exception as e:
        error_msg = str(e)[:500]
        trial.set_user_attr("error", error_msg)
        tqdm.write(f"[Trial {trial.number}] Error: {error_msg}")
        raise

    finally:
        # Delete ALL local vars holding GPU refs or model refs
        # outputs/loss hold comp graph refs blocking C++ destructor
        try:
            del outputs
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
            del input_ids
        except NameError:
            pass
        try:
            del attention_mask
        except NameError:
            pass
        try:
            del labels
        except NameError:
            pass
        # Reverse dependency order: scheduler -> optimizer -> model
        try:
            del scheduler
        except NameError:
            pass
        try:
            del optimizer
        except NameError:
            pass
        if model is not None:
            del model
        gc.collect()
        gc.collect()  # Second pass for cyclic refs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# =============================================================================
# Visualization
# =============================================================================

def visualize_study(study, save_dir):
    """Visualize optimization history, parameter importance, and LR vs metric."""
    metric_name = TASK_TO_METRIC[TASK_NAME]
    metric_label = f"{metric_name} (%)"

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
    axes[0].set_ylabel(metric_label)
    axes[0].set_title(f'Optimization History ({TASK_NAME})')
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

    # LR vs metric
    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    axes[2].scatter(lrs, acc_scores, alpha=0.6)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Learning Rate')
    axes[2].set_ylabel(metric_label)
    axes[2].set_title(f'Learning Rate vs {metric_name}')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "visualization.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Visualization saved.")


def print_study_summary(study):
    """Print study summary."""
    metric_name = TASK_TO_METRIC[TASK_NAME]
    print("\n" + "=" * 60)
    print(f"STUDY SUMMARY ({TASK_NAME})")
    print("=" * 60)
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Study: {study.study_name}, Trials: {len(study.trials)} ({len(complete_trials)} complete)")
    if complete_trials:
        acc_scores = [t.value for t in complete_trials]
        print(f"Best {metric_name}: {max(acc_scores):.2f}%, Mean: {sum(acc_scores)/len(acc_scores):.2f}%")
        print(f"Best params: {study.best_params}")


# =============================================================================
# Main
# =============================================================================

def main():
    global TASK_NAME, NUM_LABELS, MAX_SEQ_LENGTH, BATCH_SIZE, GRAD_ACCUM_STEPS, N_EPOCHS, WARMUP_STEPS, TRANSFER_METHOD, AB_DEVICE, IO_NOISE, LORA_TARGET, HEAD_LAYER, ENCODER_ANALOG, EMBEDDING_ANALOG, HEAD_ANALOG, BACKWARD_OUT_BOUND, RESULTS, _oom_retry_pending

    parser = argparse.ArgumentParser(description="Optuna sweep for ALBERT GLUE LRTT")
    parser.add_argument('--task', type=str, default='sst2',
                        choices=list(TASK_TO_KEYS.keys()),
                        help='GLUE task name (default: sst2)')
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
                        choices=['standard', 'decay', 'hybrid'],
                        help='Fix reinit mode (default: tune all three)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS,
                        help=f'Number of epochs (default: {N_EPOCHS})')
    parser.add_argument('--warmup-steps', type=int, default=WARMUP_STEPS,
                        help=f'LR warmup steps (default: {WARMUP_STEPS})')
    parser.add_argument('--transfer-method', type=str, default=TRANSFER_METHOD,
                        choices=['onehot', 'direct', 'set'],
                        help=f'Transfer method (default: {TRANSFER_METHOD})')
    parser.add_argument('--ab-device', type=str, default=AB_DEVICE,
                        choices=['6t1c', 'fp'],
                        help=f'A/B tile device type (default: {AB_DEVICE})')
    parser.add_argument('--no-io-noise', action='store_true',
                        help='Disable IO out_noise (resolution kept)')
    parser.add_argument('--no-transfer', action='store_true',
                        help='Disable transfer (set transfer_every to infinity)')
    parser.add_argument('--no-adc-ab-proj', action='store_true',
                        help='Use digital matmul for A/B projections (no ADC/DAC between B and A)')
    parser.add_argument('--lora-target', type=str, default=LORA_TARGET,
                        choices=['none', 'qonly', 'konly', 'vonly', 'qkv', 'qkvo', 'ffn', 'all'],
                        help='LoRA target: none, qonly, konly, vonly, qkv, qkvo, ffn, all (default: qkv)')
    parser.add_argument('--head-layer', type=str, default=HEAD_LAYER,
                        choices=['train', 'freeze'],
                        help='classifier layer: train or freeze (default: train)')
    parser.add_argument('--encoder-analog', action='store_true', default=ENCODER_ANALOG,
                        help='Convert non-LRTT encoder layers to frozen analog (default: digital)')
    parser.add_argument('--embedding-analog', action='store_true', default=EMBEDDING_ANALOG,
                        help='Convert embedding projection to frozen analog (default: digital)')
    parser.add_argument('--head-analog', action='store_true', default=HEAD_ANALOG,
                        help='Convert classifier to frozen analog (default: digital)')
    parser.add_argument('--backward-out-bound', type=float, default=BACKWARD_OUT_BOUND,
                        help=f'Backward pass output bound (default: {BACKWARD_OUT_BOUND})')
    parser.add_argument('--no-learn-out-scaling', action='store_true',
                        help='Disable trainable out_scaling on C tile')
    parser.add_argument('--auto-scale-mode', type=str, default='none',
                        choices=['none', 'shared', 'separate'],
                        help='Auto-scale mode for A/B LR normalization (default: none)')
    args = parser.parse_args()

    # Update global config
    TASK_NAME = args.task
    NUM_LABELS = TASK_TO_NUM_LABELS[TASK_NAME]
    MAX_SEQ_LENGTH = TASK_TO_MAX_SEQ_LENGTH[TASK_NAME]
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_STEPS = args.grad_accum_steps
    N_EPOCHS = args.epochs
    WARMUP_STEPS = args.warmup_steps
    TRANSFER_METHOD = args.transfer_method
    AB_DEVICE = args.ab_device
    IO_NOISE = not args.no_io_noise
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
    ENCODER_ANALOG = args.encoder_analog
    EMBEDDING_ANALOG = args.embedding_analog
    HEAD_ANALOG = args.head_analog
    BACKWARD_OUT_BOUND = args.backward_out_bound

    # Per-task results directory
    RESULTS = os.path.join(os.getcwd(), "results", f"optuna_albert_{TASK_NAME}_lrtt")
    os.makedirs(RESULTS, exist_ok=True)

    # Auto-generate study name: albert_{TASK_NAME}_lrtt_bs{BS}_{suffix}
    study_name = args.study_name or f"albert_{TASK_NAME}_lrtt_bs{BATCH_SIZE}_{get_study_name_suffix()}"

    storage = JournalStorage(JournalFileBackend(f"{RESULTS}/optuna_{study_name}.log"))

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print_study_summary(study)
        visualize_study(study, RESULTS)
        return

    # Check for OOM retry file (from previous OOM restart)
    retry_file = os.path.join(RESULTS, f"_oom_retry_{study_name}.json")
    retry_info = None
    if os.path.exists(retry_file):
        with open(retry_file) as f:
            retry_info = json.load(f)
        os.remove(retry_file)  # Delete immediately so it won't persist across manual reruns
        GRAD_ACCUM_STEPS = retry_info["grad_accum_steps"]
        _oom_retry_pending = True
        print(f"[OOM Retry] Retrying trial {retry_info['trial_number']}, "
              f"GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS}, micro_bs={BATCH_SIZE // GRAD_ACCUM_STEPS}")

    # Load data once (shared across all trials)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=ConfigAwareBoTorchSampler(n_startup_trials=10),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    # Enqueue retry trial if OOM retry pending
    if retry_info is not None:
        study.enqueue_trial(retry_info["trial_params"])

    print(f"\nTask: {TASK_NAME}, Metric: {TASK_TO_METRIC[TASK_NAME]}")
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
            lambda trial: objective(trial, train_loader, eval_loader, tokenizer),
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
        metric_name = TASK_TO_METRIC[TASK_NAME]
        best_params_file = os.path.join(RESULTS, f"best_params_{study_name}.json")
        with open(best_params_file, 'w') as f:
            json.dump({
                "task": TASK_NAME,
                "metric": metric_name,
                f"best_{metric_name}": study.best_value,
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

    all_trials_file = os.path.join(RESULTS, f"all_trials_{TASK_NAME}.json")
    with open(all_trials_file, 'w') as f:
        json.dump(all_trials, f, indent=2)
    print(f"All trials saved to: {all_trials_file}")


class _OOMRestart(Exception):
    """Raised to trigger process restart after OOM."""
    pass


class _OOMRetryDone(Exception):
    """Raised after OOM retry trial to restart with default GRAD_ACCUM_STEPS."""
    pass


_oom_retry_pending = False



def _oom_restart_callback(study, trial):
    """Optuna callback: on OOM, save retry file and restart process."""
    global _oom_retry_pending

    if trial.state == TrialState.FAIL:
        err = trial.user_attrs.get("error", "")
        if "out of memory" in err.lower() or "cublas" in err.lower():
            new_grad_accum = GRAD_ACCUM_STEPS * 2
            micro_bs = BATCH_SIZE // new_grad_accum
            if micro_bs < 1:
                print(f"\n[OOM Recovery] Cannot reduce micro-batch below 1 "
                      f"(BATCH_SIZE={BATCH_SIZE}, GRAD_ACCUM would be {new_grad_accum}). "
                      f"Skipping retry.")
                return

            retry_file = os.path.join(RESULTS, f"_oom_retry_{study.study_name}.json")
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

    # After retry trial completes, restart to reset GRAD_ACCUM_STEPS to default
    if _oom_retry_pending:
        _oom_retry_pending = False
        print(f"\n[OOM Recovery] Retry trial {trial.number} done (state={trial.state.name}). "
              f"Restarting with default GRAD_ACCUM.")
        raise _OOMRetryDone()


if __name__ == "__main__":
    main()
