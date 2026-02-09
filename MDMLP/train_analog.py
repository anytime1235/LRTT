#!/usr/bin/env python3
"""MDMLP Analog Training Script with LRTT / TikiTaka / TTv2 support.

Based on train.py from MDMLP, modified to support analog training
using aihwkit's convert_to_analog() and AnalogSGD.

Supported algorithms (--algo):
- lrtt:     PythonLRTTRPUConfig (3-tile: A/B + C, LoRA-style)
- tikitaka: TransferCompound (2-tile: Fast + Slow, TikiTaka v1)
- ttv2:     ChoppedTransferCompound (2-tile: Fast + Slow, TikiTaka v2)

All three share the same device models:
- Fast/A/B tiles: 6T1C LinearStepDevice
- Slow/C tile: SoftBoundsDevice (noise=0)

Key changes from train.py:
- Added --analog flag and algorithm-specific CLI arguments
- convert_to_analog() with FloatingPointRPUConfig for stem/head
- AnalogSGD optimizer with regroup_param_groups
- C-only validation support (LRTT: forward_inject=False)
- AMP disabled by default in analog mode
"""
import argparse
import math
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

# aihwkit imports
from aihwkit.optim import AnalogSGD
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import FloatingPointRPUConfig, MappingParameter, UnitCellRPUConfig
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs.compounds import TransferCompound, ChoppedTransferCompound
from aihwkit.simulator.parameters.enums import NoiseManagementType, BoundManagementType
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.training import UpdateParameters

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='validation batch size override (default: None)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')


# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-repeats', type=int, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
parser.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')

# ==================== Analog Arguments ====================
parser.add_argument('--analog', action='store_true', default=False,
                    help='Enable analog mode')
parser.add_argument('--algo', type=str, default='lrtt', choices=['lrtt', 'tikitaka', 'ttv2'],
                    help='Analog algorithm: lrtt, tikitaka (v1), or ttv2 (default: lrtt)')
# Common analog parameters
parser.add_argument('--transfer-every', type=int, default=100,
                    help='Transfer frequency in steps (default: 100)')
parser.add_argument('--transfer-lr', type=float, default=1.0,
                    help='Transfer learning rate (default: 1.0)')
parser.add_argument('--validate-c-only', action='store_true', default=False,
                    help='Additionally validate using C/Slow-tile only')
# LRTT-specific parameters
parser.add_argument('--lrtt-rank', type=int, default=2,
                    help='LRTT rank (default: 2)')
parser.add_argument('--lora-alpha', type=float, default=1.0,
                    help='LoRA scaling factor for LRTT (default: 1.0)')
# TikiTaka / TTv2-specific parameters
parser.add_argument('--fast-lr', type=float, default=1.0,
                    help='Fast tile learning rate for TikiTaka/TTv2 (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.0,
                    help='Weight mixing ratio for TikiTaka/TTv2 (default: 0.0, Slow only)')
# TTv2-specific parameters
parser.add_argument('--in-chop-prob', type=float, default=0.02,
                    help='TTv2 chopped input probability (default: 0.02)')
parser.add_argument('--auto-granularity', type=float, default=300.0,
                    help='TTv2 auto granularity (default: 300.0)')
parser.add_argument('--auto-scale', action='store_true', default=True,
                    help='TTv2 enable auto scaling (default: True)')
parser.add_argument('--no-auto-scale', action='store_true', default=False,
                    help='TTv2 disable auto scaling')

# ==================== Freeze Arguments (digital mode) ====================
parser.add_argument('--freeze-mode', type=str, default='none',
                    choices=['none', 'head-only', 'stem-head'],
                    help='Freeze mode for digital probing: '
                         'none (train all), '
                         'head-only (freeze everything except head.Linear), '
                         'stem-head (freeze everything except stem.Linear + head.Linear)')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


# ==================== LRTT Configuration ====================

def create_lrtt_config(
    rank: int,
    transfer_every: int,
    transfer_lr: float,
    lora_alpha: float = 1.0,
    reinit_gain: float = 0.1,
):
    """Create LRTT config matching sweep_lrtt_squad_rank8.py (dtod0 version).

    Uses 6T1C device model: A/B tiles with LinearStepDevice, C tile with SoftBoundsDevice.

    Args:
        rank: LRTT rank dimension
        transfer_every: Transfer frequency in steps
        transfer_lr: Transfer learning rate scalar
        lora_alpha: LoRA scaling factor
        reinit_gain: Kaiming initialization gain for B matrix after transfer

    Returns:
        Configured PythonLRTTRPUConfig
    """
    # Calculate lifetime for 6T1C
    TAU_SEC = 46505.0
    dt_batch_sec = 1.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

    # A/B tiles: LinearStepDevice (6T1C)
    ab_device = LinearStepDevice(
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

    # C tile: SoftBoundsDevice (noise-free)
    c_device = SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=True,
    )

    # LRTT Device config
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        reinit_gain=reinit_gain,
        reinit_mode="hybrid",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = transfer_lr
    device_config.units_in_mbatch = True
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = False  # y = Cx only (default LRTT)

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    # Weight scaling for C tile
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


# ==================== Shared Device Factories ====================

def _create_sixt1c_device():
    """Create 6T1C LinearStepDevice (shared across LRTT/TikiTaka/TTv2)."""
    TAU_SEC = 46505.0
    dt_batch_sec = 1.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

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


def _create_softbounds_device_noisefree():
    """Create SoftBoundsDevice with all noise=0 (shared across LRTT/TikiTaka/TTv2)."""
    return SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=True,
    )


# ==================== TikiTaka v1 Configuration ====================

def create_tikitaka_config(
    transfer_every: int,
    transfer_lr: float,
    fast_lr: float = 1.0,
    gamma: float = 0.0,
):
    """Create TikiTaka v1 config (TransferCompound).

    2-tile structure: Fast (6T1C) + Slow (SoftBounds noise=0).
    Same devices as LRTT A/B and C tiles.

    Args:
        transfer_every: Transfer frequency in mini-batches
        transfer_lr: Transfer learning rate
        fast_lr: Fast tile learning rate multiplier
        gamma: Weight mixing (0=Slow only visible)

    Returns:
        Configured UnitCellRPUConfig
    """
    fast_device = _create_sixt1c_device()
    slow_device = _create_softbounds_device_noisefree()

    transfer_io = IOParameters(
        noise_management=NoiseManagementType.NONE,
        bound_management=BoundManagementType.NONE,
    )

    rpu_config = UnitCellRPUConfig(
        device=TransferCompound(
            unit_cell_devices=[fast_device, slow_device],
            transfer_every=transfer_every,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=gamma,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=True,
            transfer_forward=transfer_io,
        )
    )

    # Weight scaling (same as LRTT C tile)
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


# ==================== TikiTaka v2 Configuration ====================

def create_ttv2_config(
    transfer_every: int,
    transfer_lr: float,
    fast_lr: float = 1.0,
    gamma: float = 0.0,
    in_chop_prob: float = 0.02,
    auto_granularity: float = 300.0,
    auto_scale: bool = True,
):
    """Create TikiTaka v2 config (ChoppedTransferCompound).

    2-tile structure: Fast (6T1C) + Slow (SoftBounds noise=0).
    Same devices as LRTT A/B and C tiles.
    Key v2 features: chopped updates, auto scaling.

    Args:
        transfer_every: Transfer frequency (mat-vec units)
        transfer_lr: Transfer learning rate
        fast_lr: Fast tile learning rate multiplier
        gamma: Weight mixing (0=Slow only visible)
        in_chop_prob: Input chopper switching probability
        auto_granularity: Auto scaling granularity
        auto_scale: Enable auto scaling

    Returns:
        Configured UnitCellRPUConfig
    """
    fast_device = _create_sixt1c_device()
    slow_device = _create_softbounds_device_noisefree()

    transfer_io = IOParameters(
        noise_management=NoiseManagementType.NONE,
        bound_management=BoundManagementType.NONE,
    )

    transfer_update = UpdateParameters(
        desired_bl=1,
        update_bl_management=False,
        update_management=False,
    )

    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[fast_device, slow_device],
            transfer_every=transfer_every,
            units_in_mbatch=False,  # mat-vec units for TTv2
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=gamma,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=True,
            auto_scale=auto_scale,
            auto_granularity=auto_granularity,
            buffer_granularity=1.0,
            auto_momentum=0.99,
            in_chop_prob=in_chop_prob,
            in_chop_random=True,
            transfer_forward=transfer_io,
            transfer_update=transfer_update,
        )
    )

    # Weight scaling (same as LRTT C tile)
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


def remove_bias_from_lrtt_layers(model, exclude_names):
    """Remove bias from Linear layers that will be converted to LRTT.

    LRTT layers do not support bias. This must be called before convert_to_analog().

    Args:
        model: Digital model
        exclude_names: Set of module names that should keep bias (stem/head)
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name not in exclude_names:
            if module.bias is not None:
                module.bias = None
                _logger.info(f"  Removed bias from {name}")


def convert_model_to_analog(model, args):
    """Convert digital MDMLP model to analog.

    Supports LRTT, TikiTaka v1, and TikiTaka v2 (TTv2).

    Architecture:
    - stem.Linear, head.Linear -> FloatingPointRPUConfig (digital)
    - clayer.dense1, clayer.dense2 (x8) -> FloatingPointRPUConfig (digital)
      (min dim=3, too small for meaningful analog; only 72 params total)
    - Remaining 48 Linear layers -> algo-specific RPUConfig (analog)

    Args:
        model: Digital MDMLP model
        args: Parsed arguments

    Returns:
        Analog model
    """
    algo = getattr(args, 'algo', 'lrtt')

    # Create algorithm-specific RPU config
    if algo == 'lrtt':
        default_config = create_lrtt_config(
            rank=args.lrtt_rank,
            transfer_every=args.transfer_every,
            transfer_lr=args.transfer_lr,
            lora_alpha=args.lora_alpha,
        )
        algo_name = f"LRTT (rank={args.lrtt_rank})"
    elif algo == 'tikitaka':
        default_config = create_tikitaka_config(
            transfer_every=args.transfer_every,
            transfer_lr=args.transfer_lr,
            fast_lr=args.fast_lr,
            gamma=args.gamma,
        )
        algo_name = "TikiTaka v1"
    elif algo == 'ttv2':
        use_auto_scale = args.auto_scale and not args.no_auto_scale
        default_config = create_ttv2_config(
            transfer_every=args.transfer_every,
            transfer_lr=args.transfer_lr,
            fast_lr=args.fast_lr,
            gamma=args.gamma,
            in_chop_prob=args.in_chop_prob,
            auto_granularity=args.auto_granularity,
            auto_scale=use_auto_scale,
        )
        algo_name = "TikiTaka v2 (TTv2)"
    else:
        raise ValueError(f"Unknown algo: {algo}")

    # Modules to exclude from analog (use FloatingPoint instead):
    # - stem.Linear, head.Linear: input/output projection
    # - All clayer.dense1/dense2: min dim=3, too small for analog (72 params total)
    exclude_set = {"stem.Linear", "head.Linear"}
    for d in range(8):  # depth=8
        exclude_set.add(f"blocks.MDLayer{d}.clayer.dense1")
        exclude_set.add(f"blocks.MDLayer{d}.clayer.dense2")

    _logger.info(f"Digital layers (FloatingPointRPUConfig): {len(exclude_set)} layers")
    _logger.info(f"  stem.Linear, head.Linear + 16 clayer layers")

    # Remove bias from layers that will become LRTT (only needed for LRTT)
    if algo == 'lrtt':
        remove_bias_from_lrtt_layers(model, exclude_set)

    def specific_rpu_config_fun(module_name, module, rpu_config):
        """Assign FloatingPointRPUConfig for excluded layers."""
        if module_name in exclude_set:
            _logger.info(f"  {module_name} -> FloatingPointRPUConfig (digital)")
            return FloatingPointRPUConfig()

        _logger.info(f"  {module_name} -> {algo_name}")
        return rpu_config

    _logger.info(f"Converting model to analog ({algo_name})...")
    analog_model = convert_to_analog(
        model,
        default_config,
        specific_rpu_config_fun=specific_rpu_config_fun,
        verbose=True,
    )
    _logger.info("Analog conversion complete.")

    return analog_model


# ==================== C-only Validation ====================

def set_forward_inject(model, enabled):
    """Set forward_inject state for all LRTT layers.

    When enabled=True: y = Cx + alpha*A(Bx) (full model)
    When enabled=False: y = Cx only (C-tile inference)

    Args:
        model: Analog model with LRTT layers
        enabled: Whether to enable forward_inject

    Returns:
        list: Original states for restoration
    """
    original_states = []
    for module in model.modules():
        if hasattr(module, 'analog_tiles') and hasattr(module.analog_tiles, 'controller'):
            controller = module.analog_tiles.controller
            original_states.append((controller, controller.forward_inject_enabled))
            controller.forward_inject_enabled = enabled
    return original_states


def restore_forward_inject(original_states):
    """Restore original forward_inject states."""
    for controller, original_state in original_states:
        controller.forward_inject_enabled = original_state


# ==================== Main ====================

def main():
    setup_default_logging()
    args, args_text = _parse_args()

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # Force disable AMP in analog mode
    use_amp = None
    if args.analog:
        if args.amp or args.native_amp or args.apex_amp:
            _logger.warning("AMP is disabled in analog mode. Training in float32.")
        args.amp = False
        args.native_amp = False
        args.apex_amp = False
    else:
        # resolve AMP arguments based on PyTorch / Apex availability
        if args.amp:
            if has_native_amp:
                args.native_amp = True
            elif has_apex:
                args.apex_amp = True
        if args.apex_amp and has_apex:
            use_amp = 'apex'
        elif args.native_amp and has_native_amp:
            use_amp = 'native'
        elif args.apex_amp or args.native_amp:
            _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                            "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed, args.rank)

    if args.fuser:
        set_jit_fuser(args.fuser)

    model = create_model(
        args.model,
        in_chans=args.input_size[0],
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    # ==================== Analog Conversion ====================
    if args.analog:
        algo = getattr(args, 'algo', 'lrtt')
        _logger.info(f"Analog mode enabled: algo={algo}, transfer_every={args.transfer_every}, "
                     f"transfer_lr={args.transfer_lr}")
        if algo == 'lrtt':
            _logger.info(f"  LRTT: rank={args.lrtt_rank}, lora_alpha={args.lora_alpha}")
        elif algo == 'tikitaka':
            _logger.info(f"  TikiTaka: fast_lr={args.fast_lr}, gamma={args.gamma}")
        elif algo == 'ttv2':
            _logger.info(f"  TTv2: fast_lr={args.fast_lr}, gamma={args.gamma}, "
                         f"in_chop_prob={args.in_chop_prob}, auto_granularity={args.auto_granularity}")
        model = convert_model_to_analog(model, args)

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    # ==================== Layer Freezing (digital probing) ====================
    freeze_mode = getattr(args, 'freeze_mode', 'none')
    if freeze_mode != 'none':
        # Determine which top-level modules to keep trainable
        if freeze_mode == 'head-only':
            trainable_prefixes = ('head.Linear',)
        elif freeze_mode == 'stem-head':
            trainable_prefixes = ('stem.Linear', 'head.Linear')
        else:
            trainable_prefixes = ()

        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze target layers (weight + bias)
        unfrozen_count = 0
        unfrozen_params = 0
        for name, param in model.named_parameters():
            for prefix in trainable_prefixes:
                if name.startswith(prefix):
                    param.requires_grad = True
                    unfrozen_count += 1
                    unfrozen_params += param.numel()
                    break

        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = total_params - unfrozen_params
        _logger.info(f"Freeze mode: {freeze_mode}")
        _logger.info(f"  Trainable: {unfrozen_count} tensors, {unfrozen_params:,} params")
        _logger.info(f"  Frozen:    {frozen_params:,} params ({100*frozen_params/total_params:.1f}%)")

    # ==================== Optimizer ====================
    if args.analog:
        # Use AnalogSGD for analog mode
        optimizer = AnalogSGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        optimizer.regroup_param_groups(model)
        _logger.info(f"Using AnalogSGD optimizer (lr={args.lr}, momentum={args.momentum}, wd={args.weight_decay})")
    else:
        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval datasets
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats)
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            print(f"\nargs.yaml: \n{args_text}\n")
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            # C-only validation (analog mode)
            if args.analog and args.validate_c_only:
                c_only_metrics = validate_c_only(model, loader_eval, validate_loss_fn, args,
                                                 amp_autocast=amp_autocast)
                if args.local_rank == 0:
                    _logger.info(
                        f"C-only validation: Acc@1={c_only_metrics['top1']:.2f}  "
                        f"Acc@5={c_only_metrics['top5']:.2f}  "
                        f"Loss={c_only_metrics['loss']:.4f}")

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


def validate_c_only(model, loader, loss_fn, args, amp_autocast=suppress):
    """Validate using C-tile only (disable forward_inject for LRTT layers)."""
    original_states = set_forward_inject(model, enabled=False)
    try:
        metrics = validate(model, loader, loss_fn, args, amp_autocast=amp_autocast,
                           log_suffix=' (C-only)')
    finally:
        restore_forward_inject(original_states)
    return metrics


if __name__ == '__main__':
    main()
