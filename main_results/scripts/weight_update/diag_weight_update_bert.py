"""Diagnose analog weight-update issues during BERT-base QA fine-tuning.

Supports SingleRPU (SoftBoundsDevice) and TikiTaka (ChoppedTransferCompound v1/v2).

Usage:
  # SingleRPU, 100 steps
  python diag_weight_update_bert.py --mode single --steps 100 --dw-min 0.001

  # TikiTaka v2, 100 steps
  python diag_weight_update_bert.py --mode tiki --steps 100 --dw-min 0.001

  # Multi-seed run (Protocol B: vary both data+model seed)
  python diag_weight_update_bert.py --mode single --steps 50 --seeds "0,1,2,3,4"

  # Protocol A: fixed data seed, varying model/update seed
  python diag_weight_update_bert.py --mode single --steps 50 --seeds "0,1,2,3,4" --seed-data 42

  # Trace only specific layers/sublayers (fast runs)
  python diag_weight_update_bert.py --mode single --steps 100 --trace-layers "0,5,11" --trace-sublayers "Q,FFN1"

  # Coarse tracing (every 5 steps); grad proxy accumulated over the interval
  python diag_weight_update_bert.py --mode single --steps 100 --trace-every 5

  # Debug tile assembly order (TikiTaka multi-tile layers)
  python diag_weight_update_bert.py --mode tiki --steps 3 --debug-tiling

  # dw_min sweep
  python diag_weight_update_bert.py --mode single --dw-min-sweep 0.0005,0.001,0.002,0.005

Results: {output_dir}/{mode}/run_{hash}/ (default: ./main_results/weight_update/squad/)
Tile assembly for hidden weights follows TileModuleArray.get_weights() order:
  for each input split: cat output tiles along dim=0; then cat input splits along dim=1.
"""

# =============================================================================
# Section 2: Imports
# =============================================================================

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    set_seed,
    default_data_collator,
)
from torch.utils.data import DataLoader
from datasets import load_dataset

from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.configs import (
    SingleRPUConfig,
    UnitCellRPUConfig,
    IOParameters,
    UpdateParameters,
)
from aihwkit.simulator.configs.devices import (
    SoftBoundsDevice,
    LinearStepDevice,
)
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.utils import (
    BoundManagementType,
    NoiseManagementType,
    PulseType,
)

# =============================================================================
# Section 3: CLI Argument Parser
# =============================================================================

parser = argparse.ArgumentParser(
    description="Weight-update diagnostics for analog BERT-base QA"
)

# Mode
parser.add_argument("--mode", type=str, default="single",
                    choices=["single", "tiki"])
parser.add_argument("--use-v2", action="store_true", default=True)
parser.add_argument("--no-v2", dest="use_v2", action="store_false")

# Update parameters
parser.add_argument("--dw-min", type=float, default=0.001)
parser.add_argument("--dw-min-sweep", type=str, default=None, metavar="DW_CSV",
                    help="e.g. '0.0005,0.001,0.002'")
parser.add_argument("--desired-bl", type=int, default=31)
parser.add_argument("--update-bl-management", type=str, default=None,
                    choices=["true", "false"])
parser.add_argument("--update-management", type=str, default=None,
                    choices=["true", "false"])
parser.add_argument("--sto-round-update", action="store_true", default=False)
parser.add_argument("--pulse-type", type=str, default="STOCHASTIC_COMPRESSED")

# TikiTaka params
parser.add_argument("--transfer-every", type=int, default=1)
parser.add_argument("--transfer-lr", type=float, default=1.0)
parser.add_argument("--fast-lr", type=float, default=1.0)
parser.add_argument("--forget-buffer", action="store_true", default=True)
parser.add_argument("--no-forget-buffer", dest="forget_buffer",
                    action="store_false")
parser.add_argument("--in-chop-prob", type=float, default=0.1)
parser.add_argument("--auto-scale", action="store_true", default=True)
parser.add_argument("--no-auto-scale", dest="auto_scale",
                    action="store_false")

# Training
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--seq-len", type=int, default=384)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--seed", type=int, default=42)

# Multi-seed
parser.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds, e.g. '0,1,2,3,4'")
parser.add_argument("--seed-data", type=int, default=None,
                    help="Fixed data seed (default: same as --seed / each seed)")
parser.add_argument("--seed-model", type=int, default=None,
                    help="Fixed model seed (default: same as --seed / each seed)")

# IO isolation
parser.add_argument("--forward-perfect", action="store_true", default=False)
parser.add_argument("--backward-perfect", action="store_true", default=False)

# Tracing
parser.add_argument("--trace-every", type=int, default=1,
                    help="Record weight deltas every N steps (delta covers N steps)")
parser.add_argument("--trace-layers", type=str, default=None,
                    help="Comma-separated layer indices, e.g. '0,5,11'")
parser.add_argument("--trace-sublayers", type=str, default=None,
                    help="Comma-separated sublayers, e.g. 'Q,O,FFN1'")

# Output
parser.add_argument("--output-dir", type=str,
                    default="./main_results/weight_update/squad")
parser.add_argument("--tag", type=str, default=None)
parser.add_argument("--sample-k", type=int, default=512)
parser.add_argument("--debug-tiling", action="store_true", default=False,
                    help="Run tile assembly sanity check at startup")
parser.add_argument("--overwrite", action="store_true", default=False,
                    help="Allow overwriting existing run directories")
parser.add_argument("--exclude-ffn", action="store_true", default=False,
                    help="Exclude FFN1/FFN2 from analog conversion (keep digital)")

args = parser.parse_args()

# =============================================================================
# Section 4: Global Constants
# =============================================================================

DOC_STRIDE = 128
SEED = args.seed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

print(f"[Config] Device={DEVICE}, mode={args.mode}, steps={args.steps}, "
      f"batch_size={args.batch_size}")
print(f"[Config] dw_min={args.dw_min}, desired_bl={args.desired_bl}, "
      f"lr={args.lr}, seed={SEED}")
if args.mode == "tiki":
    print(f"[Config] TikiTaka: use_v2={args.use_v2}, "
          f"transfer_every={args.transfer_every}, "
          f"transfer_lr={args.transfer_lr}, fast_lr={args.fast_lr}")

# =============================================================================
# Section 5: Layer Name Utilities (from diag_forward_io_single_rpu.py)
# =============================================================================

# attention.output.dense must appear BEFORE output.dense to prevent collision
_LAYER_RE = re.compile(
    r"encoder\.layer\.(\d+)\."
    r"(attention\.self\.query|attention\.self\.key|attention\.self\.value"
    r"|attention\.output\.dense|intermediate\.dense|output\.dense)"
)
_SUBLAYER_MAP = {
    "attention.self.query":   "Q",
    "attention.self.key":     "K",
    "attention.self.value":   "V",
    "attention.output.dense": "O",
    "intermediate.dense":     "FFN1",
    "output.dense":           "FFN2",
}
SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]


def parse_layer_name(name: str):
    """Returns (layer_idx: int, sublayer: str) or None."""
    m = _LAYER_RE.search(name)
    if m is None:
        return None
    return int(m.group(1)), _SUBLAYER_MAP[m.group(2)]


def _get_weights_numpy(module):
    """Get effective weight matrix as CPU numpy (via module.get_weights())."""
    W, _ = module.get_weights()
    return W.detach().cpu().float().numpy()


def _get_weights_tensor(module):
    """Get effective weight matrix as GPU float tensor (no CPU transfer)."""
    W, _ = module.get_weights()
    return W.detach().float()


def _set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# =============================================================================
# Section 6: RPU Config Creation (single + tiki)
# =============================================================================

def _parse_pulse_type(name: str) -> PulseType:
    """Map CLI string to PulseType enum."""
    name_upper = name.upper().replace("-", "_")
    try:
        return PulseType[name_upper]
    except KeyError:
        valid = [pt.name for pt in PulseType]
        raise ValueError(f"Unknown pulse type '{name}'. Valid: {valid}")


def _resolve_bool_arg(cli_val, default):
    """Resolve a CLI string 'true'/'false'/None to bool."""
    if cli_val is None:
        return default
    return cli_val.lower() == "true"


def _print_resolved_update_config(rpu_config, mode):
    """Print resolved update config for debugging."""
    u = rpu_config.update
    print(f"  [Update Config] mode={mode}")
    print(f"    gradient_update: desired_bl={u.desired_bl}, pulse_type={u.pulse_type}, "
          f"sto_round={u.sto_round}, update_bl_mgmt={u.update_bl_management}, "
          f"update_mgmt={u.update_management}")
    if mode == "tiki":
        tu = rpu_config.device.transfer_update
        print(f"    transfer_update: desired_bl={tu.desired_bl}, pulse_type={tu.pulse_type}, "
              f"sto_round={tu.sto_round}, update_bl_mgmt={tu.update_bl_management}, "
              f"update_mgmt={tu.update_management}")


def create_single_config(args):
    """SingleRPUConfig with SoftBoundsDevice for diagnostics."""
    device = SoftBoundsDevice(
        dw_min=args.dw_min,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=False,
    )

    rpu = SingleRPUConfig(device=device)

    # Update params
    rpu.update.desired_bl = args.desired_bl
    rpu.update.update_bl_management = _resolve_bool_arg(
        args.update_bl_management, True)
    rpu.update.update_management = _resolve_bool_arg(
        args.update_management, True)
    rpu.update.sto_round = args.sto_round_update
    rpu.update.pulse_type = _parse_pulse_type(args.pulse_type)

    # Forward/backward
    rpu.forward.out_noise = 0.0
    rpu.backward.out_noise = 0.0
    if args.forward_perfect:
        rpu.forward.is_perfect = True
    if args.backward_perfect:
        rpu.backward.is_perfect = True

    # Mapping
    rpu.mapping.digital_bias = True
    rpu.mapping.weight_scaling_omega = 1.0
    rpu.mapping.weight_scaling_columnwise = True

    return rpu


def _create_a_device():
    """Create A tile: 6T1C LinearStepDevice (fast, noisy)."""
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
        lifetime=0.0,
        lifetime_dtod=0.0,
        reset=0.0,
        reset_dtod=0.0,
    )


def create_tiki_config(args):
    """UnitCellRPUConfig with ChoppedTransferCompound for TikiTaka."""
    a_device = _create_a_device()
    b_device = SoftBoundsDevice(
        dw_min=args.dw_min,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=False,
    )

    use_v2 = args.use_v2

    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[a_device, b_device],
            transfer_every=args.transfer_every,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=args.transfer_lr,
            fast_lr=args.fast_lr,
            scale_transfer_lr=use_v2,
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(
                desired_bl=args.desired_bl,
                update_bl_management=_resolve_bool_arg(
                    args.update_bl_management, not use_v2),
                update_management=_resolve_bool_arg(
                    args.update_management, not use_v2),
            ),
            no_buffer=(not use_v2),
            in_chop_prob=args.in_chop_prob if use_v2 else 0.0,
            out_chop_prob=0.0,
            forget_buffer=args.forget_buffer,
            auto_scale=args.auto_scale,
            auto_momentum=0.99,
        )
    )

    # Gradient update on fast tile (A) — same CLI pulse settings as single mode
    rpu_config.update.desired_bl = args.desired_bl
    rpu_config.update.pulse_type = _parse_pulse_type(args.pulse_type)
    rpu_config.update.sto_round = args.sto_round_update
    rpu_config.update.update_bl_management = _resolve_bool_arg(
        args.update_bl_management, True)
    rpu_config.update.update_management = _resolve_bool_arg(
        args.update_management, True)

    # Transfer update on slow tile (B) — also apply pulse_type and sto_round
    try:
        rpu_config.device.transfer_update.pulse_type = _parse_pulse_type(args.pulse_type)
        rpu_config.device.transfer_update.sto_round = args.sto_round_update
    except AttributeError:
        pass

    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    if args.forward_perfect:
        rpu_config.forward.is_perfect = True
    if args.backward_perfect:
        rpu_config.backward.is_perfect = True

    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True

    return rpu_config


# =============================================================================
# Section 7: Model Creation
# =============================================================================

def _encoder_linear_names(model, exclude_ffn=False):
    """All encoder Linear layer names, excluding qa_outputs and pooler.

    If exclude_ffn=True, also excludes FFN1 (intermediate.dense) and
    FFN2 (output.dense that is NOT attention.output.dense).
    """
    always_digital = ["qa_outputs", "pooler"]
    names = []
    for n, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if "encoder" not in n:
            continue
        if any(d in n for d in always_digital):
            continue
        if exclude_ffn:
            # FFN1: encoder.layer.N.intermediate.dense
            if "intermediate.dense" in n:
                continue
            # FFN2: encoder.layer.N.output.dense (but NOT attention.output.dense)
            if "output.dense" in n and "attention" not in n:
                continue
        names.append(n)
    return names


def create_model(args, model_seed=None):
    """BERT-base with all encoder linears converted to analog.

    qa_outputs stays digital (trainable). Embeddings/pooler stay digital (frozen).
    """
    seed = model_seed if model_seed is not None else SEED
    _set_all_seeds(seed)

    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

    # Reinit qa_outputs for reproducibility
    torch.manual_seed(seed)
    model.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
    model.qa_outputs.bias.data.zero_()

    # Identify layers
    enc_names = _encoder_linear_names(model, exclude_ffn=getattr(args, 'exclude_ffn', False))
    all_linear_names = [
        n for n, m in model.named_modules() if isinstance(m, nn.Linear)
    ]
    exclude = [n for n in all_linear_names if n not in enc_names]

    # Create config
    if args.mode == "single":
        rpu_config = create_single_config(args)
    else:
        rpu_config = create_tiki_config(args)

    _print_resolved_update_config(rpu_config, args.mode)

    # Single-pass conversion
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    # Gradient control
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        if isinstance(p, AnalogContext):
            p.requires_grad_(True)
    for n, p in model.named_parameters():
        if "qa_outputs" in n:
            p.requires_grad_(True)

    n_analog = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Analog tiles: {n_analog}, mode={args.mode}")

    return model.to(DEVICE)


# =============================================================================
# Section 8: Data Loading (SQuAD subset)
# =============================================================================

def load_data(tokenizer, n_step, batch_size, seq_len, data_seed=None):
    """SQuAD v1.1 -- subset for n_step batches. Seed-fixed for reproducibility."""
    seed = data_seed if data_seed is not None else SEED
    max_seq_length = seq_len

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=max_seq_length, truncation="only_second",
            stride=DOC_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        sp, ep = [], []
        for i, offset in enumerate(offset_mapping):
            ans = answers[sample_map[i]]
            if not ans["answer_start"]:
                sp.append(0); ep.append(0); continue
            sc = ans["answer_start"][0]
            ec = sc + len(ans["text"][0])
            seq = inputs.sequence_ids(i)
            idx = 0
            while seq[idx] != 1:
                idx += 1
            cs = idx
            while idx < len(seq) and seq[idx] == 1:
                idx += 1
            ce = idx - 1
            if offset[cs][0] > ec or offset[ce][1] < sc:
                sp.append(0); ep.append(0)
            else:
                idx = cs
                while idx <= ce and offset[idx][0] <= sc:
                    idx += 1
                sp.append(idx - 1)
                idx = ce
                while idx >= cs and offset[idx][1] >= ec:
                    idx -= 1
                ep.append(idx + 1)
        inputs["start_positions"] = sp
        inputs["end_positions"] = ep
        return inputs

    raw = load_dataset("squad")
    tok = raw["train"].map(
        preprocess_train, batched=True,
        remove_columns=raw["train"].column_names,
    )
    n = min(n_step * batch_size, len(tok))
    subset = tok.shuffle(seed=seed).select(range(n))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        collate_fn=default_data_collator)
    print(f"  Dataset: {n} samples -> {len(loader)} batches")
    return loader


# =============================================================================
# Section 9: SampledIndexManager + Hook Registration
# =============================================================================

class SampledIndexManager:
    """Manages sampled weight indices for efficient gradient proxy computation."""

    def __init__(self, out_features, in_features, sample_k, seed=42):
        self.out_features = out_features
        self.in_features = in_features
        n_total = out_features * in_features
        k = min(sample_k, n_total)
        self.k = k

        rng = np.random.RandomState(seed)
        flat_idx = rng.choice(n_total, size=k, replace=False)
        flat_idx.sort()
        self.flat_idx = flat_idx

        # Decompose to 2D
        self.i_indices = flat_idx // in_features    # row (output dim)
        self.j_indices = flat_idx % in_features     # col (input dim)

        # Unique indices for efficient hook gathering
        self.unique_i, self._inv_i = np.unique(self.i_indices,
                                                return_inverse=True)
        self.unique_j, self._inv_j = np.unique(self.j_indices,
                                                return_inverse=True)

        # Torch tensors for vectorized g_ij: positions mapping k samples
        # to their location in unique_i / unique_j arrays
        self.ui_positions_t = torch.tensor(self._inv_i, dtype=torch.long)
        self.uj_positions_t = torch.tensor(self._inv_j, dtype=torch.long)

        # GPU-resident flat index tensor for weight extraction
        self.flat_idx_t = torch.tensor(self.flat_idx, dtype=torch.long)

    def to(self, device):
        """Move all index tensors to the specified device."""
        self.flat_idx_t = self.flat_idx_t.to(device)
        self.ui_positions_t = self.ui_positions_t.to(device)
        self.uj_positions_t = self.uj_positions_t.to(device)
        return self


def _make_forward_pre_hook(layer_info, hook_active):
    """Create forward pre-hook to capture input activations at sampled columns."""
    j_tensor = torch.tensor(layer_info["sampler"].unique_j, dtype=torch.long)

    def hook(module, args):
        if not hook_active[0]:
            return
        x = args[0]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        layer_info["x_sampled"] = x[:, j_tensor.to(x.device)].detach()
        with torch.no_grad():
            layer_info["x_absmax"] = x.abs().max().item()

    return hook


def _make_backward_hook(layer_info, hook_active):
    """Create backward hook to capture grad_output at sampled rows."""
    i_tensor = torch.tensor(layer_info["sampler"].unique_i, dtype=torch.long)

    def hook(module, grad_input, grad_output):
        if not hook_active[0]:
            return
        d = grad_output[0]
        if d is None:
            return
        if d.dim() == 3:
            d = d.reshape(-1, d.shape[-1])
        layer_info["d_sampled"] = d[:, i_tensor.to(d.device)].detach()
        with torch.no_grad():
            layer_info["d_absmax"] = d.abs().max().item()

    return hook


def register_xd_hooks(model, sample_k, hook_active, trace_args=None):
    """Register forward pre-hooks and backward hooks on all AnalogLinear layers.

    Returns (layer_infos_dict, handle_list).
    """
    # Parse trace filters
    allowed_layers = None
    allowed_subs = None
    if trace_args is not None:
        if trace_args.trace_layers is not None:
            allowed_layers = {int(x) for x in trace_args.trace_layers.split(",")}
        if trace_args.trace_sublayers is not None:
            allowed_subs = {x.strip() for x in trace_args.trace_sublayers.split(",")}

    layer_infos = {}
    handles = []

    for name, module in model.named_modules():
        if not isinstance(module, AnalogLinear):
            continue
        parsed = parse_layer_name(name)
        if parsed is None:
            continue
        layer_idx, sublayer = parsed

        # Apply layer/sublayer filters
        if allowed_layers is not None and layer_idx not in allowed_layers:
            continue
        if allowed_subs is not None and sublayer not in allowed_subs:
            continue

        w, _ = module.get_weights()
        out_f, in_f = w.shape[0], w.shape[1]

        seed_val = 42 + int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 10000
        sampler = SampledIndexManager(out_f, in_f, sample_k, seed=seed_val)
        sampler.to(DEVICE)

        info = {
            "module": module,
            "name": name,
            "layer_idx": layer_idx,
            "sublayer": sublayer,
            "sampler": sampler,
            "out_features": out_f,
            "in_features": in_f,
            "x_sampled": None,
            "d_sampled": None,
            "x_absmax": 0.0,
            "d_absmax": 0.0,
        }
        layer_infos[name] = info

        h1 = module.register_forward_pre_hook(_make_forward_pre_hook(info, hook_active))
        h2 = module.register_full_backward_hook(_make_backward_hook(info, hook_active))
        handles.extend([h1, h2])

    print(f"  Registered hooks on {len(layer_infos)} layers, sample_k={sample_k}")
    return layer_infos, handles


# =============================================================================
# Section 10: Gradient Proxy Computation
# =============================================================================

def compute_grad_proxy(layer_info, lr, dw_min, desired_bl):
    """Compute gradient proxy metrics from hooked x/d activations.

    Must be called AFTER loss.backward() but BEFORE optimizer.step().
    Returns dict of metrics + dw_fp numpy array for cosine/slope comparison.
    """
    sampler = layer_info["sampler"]
    x_s = layer_info["x_sampled"]   # (N, |unique_j|)
    d_s = layer_info["d_sampled"]   # (N, |unique_i|)

    if x_s is None or d_s is None:
        # No data captured (e.g., layer wasn't reached)
        result = {
            "grad_absmean": float("nan"),
            "grad_deadzone_ratio": float("nan"),
            "BL_mean": float("nan"),
            "BL_p99": float("nan"),
            "BL_hit_ratio": float("nan"),
            "pulse_under_frac": float("nan"),
            "pulse_ok_frac": float("nan"),
            "pulse_over_frac": float("nan"),
            "dw_fp": None,
        }
        layer_info["x_sampled"] = None
        layer_info["d_sampled"] = None
        return result

    device = x_s.device

    # Vectorized g_ij computation: outer product at sampled positions
    # d_gathered: (N, k), x_gathered: (N, k)
    d_gathered = d_s[:, sampler.ui_positions_t.to(device)]
    x_gathered = x_s[:, sampler.uj_positions_t.to(device)]
    g_ij = (d_gathered * x_gathered).sum(dim=0)  # (k,)

    dw_fp = -lr * g_ij  # float-point expected weight update

    grad_absmean = g_ij.abs().mean().item()
    grad_deadzone_ratio = (dw_fp.abs() < dw_min).float().mean().item()

    # BL estimation per sampled pair
    # Use max across token dim for each sampled j and i
    x_j_max = x_s.abs().max(dim=0).values  # (|unique_j|,)
    d_i_max = d_s.abs().max(dim=0).values  # (|unique_i|,)

    # Gather to k positions
    x_j_gathered = x_j_max[sampler.uj_positions_t.to(device)]  # (k,)
    d_i_gathered = d_i_max[sampler.ui_positions_t.to(device)]  # (k,)

    bl_pred = torch.ceil(
        lr * x_j_gathered * d_i_gathered / (dw_min + EPS)
    )
    BL_mean = bl_pred.mean().item()
    BL_p99 = bl_pred.float().quantile(0.99).item() if bl_pred.numel() > 0 else 0.0
    BL_hit_ratio = (bl_pred >= desired_bl).float().mean().item()

    # 3-zone pulse classification (reuse gathered signals)
    p_est = lr * x_j_gathered * d_i_gathered / (dw_min + EPS)
    pulse_under_frac = (p_est < 1.0).float().mean().item()
    pulse_ok_frac = ((p_est >= 1.0) & (p_est <= desired_bl)).float().mean().item()
    pulse_over_frac = (p_est > desired_bl).float().mean().item()

    result = {
        "grad_absmean": grad_absmean,
        "grad_deadzone_ratio": grad_deadzone_ratio,
        "BL_mean": BL_mean,
        "BL_p99": BL_p99,
        "BL_hit_ratio": BL_hit_ratio,
        "pulse_under_frac": pulse_under_frac,
        "pulse_ok_frac": pulse_ok_frac,
        "pulse_over_frac": pulse_over_frac,
        "dw_fp": dw_fp.detach().float(),  # GPU tensor
    }

    # Free memory
    layer_info["x_sampled"] = None
    layer_info["d_sampled"] = None

    return result


# =============================================================================
# Section 11: WeightUpdateTracker Class
# =============================================================================

def _compute_delta_metrics(delta_np, dw_min):
    """Compute update delta metrics from numpy array."""
    abs_delta = np.abs(delta_np)
    nonzero_mask = abs_delta > 0
    return {
        "zero_ratio": float(np.mean(delta_np == 0)),
        "1lsb_ratio": float(np.mean(nonzero_mask & (abs_delta <= 1.1 * dw_min))),
        "absmean": float(np.mean(abs_delta)),
        "min_nonzero": float(np.min(abs_delta[nonzero_mask]))
        if np.any(nonzero_mask) else 0.0,
        "absmax": float(np.max(abs_delta)) if abs_delta.size > 0 else 0.0,
        "q90": float(np.quantile(abs_delta, 0.90)) if abs_delta.size > 0 else 0.0,
        "q99": float(np.quantile(abs_delta, 0.99)) if abs_delta.size > 0 else 0.0,
    }


def _cosine_sim(a_np, b_np):
    """Cosine similarity between two numpy vectors."""
    dot = np.dot(a_np, b_np)
    na = np.linalg.norm(a_np)
    nb = np.linalg.norm(b_np)
    if na < EPS or nb < EPS:
        return 0.0
    return float(dot / (na * nb))


def _lstsq_slope(x_np, y_np):
    """Least-squares slope: beta = dot(x, y) / dot(x, x)."""
    xx = np.dot(x_np, x_np)
    if xx < EPS:
        return 0.0
    return float(np.dot(x_np, y_np) / xx)


# --- GPU (torch) versions of utility functions ---

def _compute_delta_metrics_torch(delta_t, dw_min):
    """Compute update delta metrics from GPU tensor. Returns dict of floats."""
    abs_delta = delta_t.abs()
    nonzero_mask = abs_delta > 0
    n = delta_t.numel()
    return {
        "zero_ratio": float((delta_t == 0).float().sum().item() / n),
        "1lsb_ratio": float((nonzero_mask & (abs_delta <= 1.1 * dw_min)).float().sum().item() / n),
        "absmean": float(abs_delta.mean().item()),
        "min_nonzero": float(abs_delta[nonzero_mask].min().item())
        if nonzero_mask.any() else 0.0,
        "absmax": float(abs_delta.max().item()) if n > 0 else 0.0,
        "q90": float(abs_delta.float().quantile(0.90).item()) if n > 0 else 0.0,
        "q99": float(abs_delta.float().quantile(0.99).item()) if n > 0 else 0.0,
    }


def _cosine_sim_torch(a_t, b_t):
    """Cosine similarity between two GPU tensors."""
    dot = torch.dot(a_t, b_t)
    na = torch.linalg.norm(a_t)
    nb = torch.linalg.norm(b_t)
    if na.item() < EPS or nb.item() < EPS:
        return 0.0
    return float((dot / (na * nb)).item())


def _lstsq_slope_torch(x_t, y_t):
    """Least-squares slope from GPU tensors: beta = dot(x, y) / dot(x, x)."""
    xx = torch.dot(x_t, x_t)
    if xx.item() < EPS:
        return 0.0
    return float((torch.dot(x_t, y_t) / xx).item())


def _assemble_from_tiles_tensor(module, extract_fn):
    """Assemble per-tile 2D tensors into full weight matrix (GPU).

    Same layout as _assemble_from_tiles but uses torch.cat instead of np.concatenate.
    """
    analog_mod = getattr(module, 'analog_module', None)
    if analog_mod is None:
        return None

    if hasattr(analog_mod, 'array'):
        col_blocks = []
        for in_tiles in analog_mod.array:
            row_blocks = []
            for tile in in_tiles:
                d = extract_fn(tile)
                if d is None:
                    return None
                row_blocks.append(d)
            col_blocks.append(torch.cat(row_blocks, dim=0))
        return torch.cat(col_blocks, dim=1)

    tiles = list(module.analog_tiles())
    if not tiles:
        return None
    return extract_fn(tiles[0])


def _get_hidden_weights_tensor(module):
    """Get (fast_weight, slow_weight, hidden_weight) as GPU tensors for TikiTaka.

    Same logic as _get_hidden_weights but stays on GPU.
    """
    try:
        tiles = list(module.analog_tiles())
        if not tiles:
            return None, None, None

        names = tiles[0].tile.get_hidden_parameter_names()
        if "hidden_weights_0" not in names:
            return None, None, None

        a_idx = names.index("hidden_weights_0")
        b_idx = names.index("hidden_weights_1")
        buf_key = "buffered_FP_weight_0"
        has_buf = buf_key in names
        buf_idx = names.index(buf_key) if has_buf else None

        def _make_extractor(param_idx):
            def _extract(tile):
                hp = tile.tile.get_hidden_parameters()
                return hp[param_idx].detach().float()
            return _extract

        fast_w = _assemble_from_tiles_tensor(module, _make_extractor(a_idx))
        slow_w = _assemble_from_tiles_tensor(module, _make_extractor(b_idx))
        hidden_w = _assemble_from_tiles_tensor(module, _make_extractor(buf_idx)) if has_buf else None

        return fast_w, slow_w, hidden_w
    except (StopIteration, AttributeError, RuntimeError):
        return None, None, None


def _assemble_from_tiles(module, extract_fn):
    """Assemble per-tile 2D arrays into full weight matrix matching get_weights() order.

    TileModuleArray layout: array[input_split][output_split].
    Assembly: for each input split, cat output split tiles along dim=0;
              then cat input splits along dim=1.
    This matches the ordering used by module.get_weights(), so flat_idx from
    SampledIndexManager correctly indexes into the assembled result.

    Args:
        module: AnalogLinear module
        extract_fn: callable(tile) -> 2D numpy array (out_size_i, in_size_j)
    Returns:
        2D numpy array (out_size, in_size) matching get_weights() layout, or None.
    """
    analog_mod = getattr(module, 'analog_module', None)
    if analog_mod is None:
        return None

    # Multi-tile: TileModuleArray with array[in_split][out_split]
    if hasattr(analog_mod, 'array'):
        col_blocks = []
        for in_tiles in analog_mod.array:
            row_blocks = []
            for tile in in_tiles:
                d = extract_fn(tile)
                if d is None:
                    return None
                row_blocks.append(d)
            col_blocks.append(np.concatenate(row_blocks, axis=0))
        return np.concatenate(col_blocks, axis=1)

    # Single tile fallback
    tiles = list(module.analog_tiles())
    if not tiles:
        return None
    return extract_fn(tiles[0])


def _debug_check_tile_order(module, name=""):
    """Verify _assemble_from_tiles matches module.get_weights() layout.

    Only meaningful for multi-tile layers; run with --debug-tiling.
    """
    W_ref, _ = module.get_weights()
    W_ref = W_ref.detach().cpu().float().numpy()

    def _extract_weight(tile):
        w, _ = tile.get_weights()
        return w.detach().cpu().float().numpy()

    W_assembled = _assemble_from_tiles(module, _extract_weight)
    if W_assembled is None:
        print(f"  [DEBUG-TILING] {name}: could not assemble")
        return False

    if W_ref.shape != W_assembled.shape:
        print(f"  [DEBUG-TILING] {name}: SHAPE MISMATCH "
              f"ref={W_ref.shape} assembled={W_assembled.shape}")
        return False

    max_diff = float(np.max(np.abs(W_ref - W_assembled)))
    n_tiles = len(list(module.analog_tiles()))
    if max_diff > 1e-5:
        print(f"  [DEBUG-TILING] {name}: VALUE MISMATCH "
              f"max_diff={max_diff:.6e} ({n_tiles} tiles)")
        return False

    print(f"  [DEBUG-TILING] {name}: OK "
          f"({n_tiles} tiles, shape={W_ref.shape}, max_diff={max_diff:.2e})")
    return True


def _get_hidden_weights(module):
    """Get (fast_weight, slow_weight, hidden_weight) as 2D numpy for TikiTaka.

    Uses _assemble_from_tiles() to reconstruct hidden parameter matrices in the
    same layout as module.get_weights(), ensuring flat_idx consistency.

    Hidden parameter keys:
      hidden_weights_0 = A tile (fast), hidden_weights_1 = B tile (slow),
      buffered_FP_weight_0 = digital transfer buffer.

    Returns tuple of 2D numpy arrays, or (None, None, None) if not TikiTaka.
    """
    try:
        tiles = list(module.analog_tiles())
        if not tiles:
            return None, None, None

        names = tiles[0].tile.get_hidden_parameter_names()
        if "hidden_weights_0" not in names:
            return None, None, None

        a_idx = names.index("hidden_weights_0")
        b_idx = names.index("hidden_weights_1")
        buf_key = "buffered_FP_weight_0"
        has_buf = buf_key in names
        buf_idx = names.index(buf_key) if has_buf else None

        def _make_extractor(param_idx):
            def _extract(tile):
                hp = tile.tile.get_hidden_parameters()
                return hp[param_idx].detach().cpu().float().numpy()
            return _extract

        fast_w = _assemble_from_tiles(module, _make_extractor(a_idx))
        slow_w = _assemble_from_tiles(module, _make_extractor(b_idx))
        hidden_w = _assemble_from_tiles(module, _make_extractor(buf_idx)) if has_buf else None

        return fast_w, slow_w, hidden_w
    except (StopIteration, AttributeError, RuntimeError):
        return None, None, None


class WeightUpdateTracker:
    """Track per-step per-layer weight update metrics."""

    def __init__(self, model, layer_infos, args):
        self.args = args
        self.is_tiki = (args.mode == "tiki")
        self.dw_min = args.dw_min
        self.dw_min_A = 0.001981  # Fixed A-tile dw_min from LinearStepDevice
        self.desired_bl = args.desired_bl
        self.w_max = 1.0
        self.trace_every = getattr(args, 'trace_every', 1)

        self._layers = []
        for name, info in layer_infos.items():
            module = info["module"]
            sampler = info["sampler"]

            layer_entry = {
                "name": name,
                "module": module,
                "sampler": sampler,
                "layer_idx": info["layer_idx"],
                "sublayer": info["sublayer"],
                # Prev snapshot buffers (single-read pattern)
                "w_eff_prev": None,
                "w_fast_prev": None,
                "w_slow_prev": None,
                "hidden_w_full": None,
            }
            self._layers.append(layer_entry)

        self._step_rows = []
        self._dw_fp_accum = {}  # name -> accumulated dw_fp over trace interval

        # Initial snapshot (read weights once at construction time)
        self._read_current_weights()

    def _read_current_weights(self):
        """Read sampled weights for all layers, storing as 'prev' for next delta (GPU tensors)."""
        for layer in self._layers:
            module = layer["module"]
            idx_t = layer["sampler"].flat_idx_t
            w_eff = _get_weights_tensor(module).reshape(-1)
            idx_dev = idx_t.to(w_eff.device)
            layer["w_eff_prev"] = w_eff[idx_dev].to(idx_t.device).clone()
            if self.is_tiki:
                fast_w, slow_w, hidden_w = _get_hidden_weights_tensor(module)
                if fast_w is not None:
                    f_flat = fast_w.reshape(-1)
                    layer["w_fast_prev"] = f_flat[idx_dev.to(f_flat.device)].to(idx_t.device).clone()
                if slow_w is not None:
                    s_flat = slow_w.reshape(-1)
                    layer["w_slow_prev"] = s_flat[idx_dev.to(s_flat.device)].to(idx_t.device).clone()
                layer["hidden_w_full"] = hidden_w.clone() if hidden_w is not None else None

    def record_after(self, step, grad_proxy_results):
        """Compute and store metrics for all layers at this step (GPU-accelerated).

        When trace_every > 1, dw_fp is accumulated across the interval so
        alignment metrics (cosine, slope, sign_mismatch) compare N-step
        dw_eff against N-step accumulated dw_fp.
        """
        # Always accumulate dw_fp for trace-interval alignment (GPU tensors)
        for _name, _gp in grad_proxy_results.items():
            _dw_fp_step = _gp.get("dw_fp")
            if _dw_fp_step is not None:
                if _name in self._dw_fp_accum:
                    self._dw_fp_accum[_name] += _dw_fp_step
                else:
                    self._dw_fp_accum[_name] = _dw_fp_step.clone()

        if step % self.trace_every != 0:
            return  # skip recording; prev stays, dw_fp accumulates

        for layer in self._layers:
            name = layer["name"]
            module = layer["module"]
            sampler = layer["sampler"]
            idx_t = sampler.flat_idx_t

            # --- A. Effective weight delta (GPU) ---
            _w_flat = _get_weights_tensor(module).reshape(-1)
            w_eff_now = _w_flat[idx_t.to(_w_flat.device)].to(idx_t.device)
            dw_eff = w_eff_now - layer["w_eff_prev"]
            eff_metrics = _compute_delta_metrics_torch(dw_eff, self.dw_min)

            # --- A2. Saturation metrics (GPU) ---
            abs_dw_eff = dw_eff.abs()
            pulse_sat_threshold = 0.9 * self.desired_bl * self.dw_min
            pulse_sat_ratio = float((abs_dw_eff >= pulse_sat_threshold).float().mean().item())
            bound_sat_ratio = float(
                (w_eff_now.abs() >= 0.98 * self.w_max).float().mean().item())

            # --- B. Gradient proxy comparison (GPU) ---
            gp = grad_proxy_results.get(name, {})
            dw_fp = self._dw_fp_accum.get(name)

            if dw_fp is not None and dw_fp.numel() == dw_eff.numel():
                update_vs_grad_cosine = _cosine_sim_torch(dw_eff, dw_fp)
                eff_lr_slope = _lstsq_slope_torch(dw_fp, dw_eff)
            else:
                update_vs_grad_cosine = float("nan")
                eff_lr_slope = float("nan")

            # --- B2. New diagnostic metrics (GPU) ---
            if dw_fp is not None and dw_fp.numel() == dw_eff.numel():
                nonzero_mask_sm = (dw_fp != 0) | (dw_eff != 0)
                if nonzero_mask_sm.any():
                    sign_mismatch_ratio = float(
                        (dw_eff[nonzero_mask_sm].sign() != dw_fp[nonzero_mask_sm].sign()
                         ).float().mean().item())
                else:
                    sign_mismatch_ratio = 0.0
            else:
                sign_mismatch_ratio = float("nan")

            if dw_fp is not None and dw_fp.numel() == dw_eff.numel():
                scaled_fp = eff_lr_slope * dw_fp
                err_norm = torch.linalg.norm(dw_eff - scaled_fp).item()
                ref_norm = torch.linalg.norm(scaled_fp).item()
                rel_update_error = float(err_norm / (ref_norm + EPS))
            else:
                rel_update_error = float("nan")

            # BL_fp (GPU)
            bl_fp = torch.ceil(dw_eff.abs() / (self.dw_min + EPS))
            BL_fp_mean = float(bl_fp.mean().item())
            BL_fp_p99 = float(bl_fp.float().quantile(0.99).item()) if bl_fp.numel() > 0 else 0.0
            BL_fp_hit_ratio = float((bl_fp >= self.desired_bl).float().mean().item())

            # dw_p50 (GPU)
            dw_p50 = float(dw_eff.abs().float().quantile(0.50).item())

            # --- C. TikiTaka-specific (GPU) ---
            tiki_row = {}
            is_transfer = False
            dw_fast = None
            dw_slow = None
            if self.is_tiki:
                is_transfer = (
                    (step % self.args.transfer_every == 0)
                    if self.args.transfer_every > 0 else False
                )
                fast_w, slow_w, hidden_w = _get_hidden_weights_tensor(module)
                # Fast (A tile) delta
                if fast_w is not None and layer["w_fast_prev"] is not None:
                    _f_flat = fast_w.reshape(-1)
                    dw_fast = _f_flat[idx_t.to(_f_flat.device)].to(idx_t.device) - layer["w_fast_prev"]
                    fm = _compute_delta_metrics_torch(dw_fast, self.dw_min_A)
                    tiki_row["dw_fast_zero_ratio"] = fm["zero_ratio"]
                    tiki_row["dw_fast_1lsb_ratio"] = fm["1lsb_ratio"]
                    tiki_row["dw_fast_absmean"] = fm["absmean"]
                    if dw_fp is not None and dw_fp.numel() == dw_fast.numel():
                        tiki_row["dw_fast_vs_grad_cosine"] = _cosine_sim_torch(
                            dw_fast, dw_fp)
                        tiki_row["dw_fast_eff_lr_slope"] = _lstsq_slope_torch(
                            dw_fp, dw_fast)
                    else:
                        tiki_row["dw_fast_vs_grad_cosine"] = float("nan")
                        tiki_row["dw_fast_eff_lr_slope"] = float("nan")
                else:
                    for k in ["dw_fast_zero_ratio", "dw_fast_1lsb_ratio",
                              "dw_fast_absmean", "dw_fast_vs_grad_cosine",
                              "dw_fast_eff_lr_slope"]:
                        tiki_row[k] = float("nan")

                # Slow (B tile) delta
                if slow_w is not None and layer["w_slow_prev"] is not None:
                    _s_flat = slow_w.reshape(-1)
                    dw_slow = _s_flat[idx_t.to(_s_flat.device)].to(idx_t.device) - layer["w_slow_prev"]
                    sm = _compute_delta_metrics_torch(dw_slow, self.dw_min)
                    tiki_row["dw_slow_zero_ratio"] = sm["zero_ratio"]
                    tiki_row["dw_slow_1lsb_ratio"] = sm["1lsb_ratio"]
                    tiki_row["dw_slow_absmean"] = sm["absmean"]
                    if is_transfer:
                        abs_dw_slow = dw_slow.abs()
                        tiki_row["transfer_duty"] = float(
                            (dw_slow != 0).float().mean().item())
                        q99_s = float(abs_dw_slow.float().quantile(0.99).item())
                        med_s = float(abs_dw_slow.float().quantile(0.50).item())
                        tiki_row["transfer_spike"] = q99_s / (med_s + 1e-12)
                    else:
                        tiki_row["transfer_duty"] = float("nan")
                        tiki_row["transfer_spike"] = float("nan")
                else:
                    for k in ["dw_slow_zero_ratio", "dw_slow_1lsb_ratio",
                              "dw_slow_absmean"]:
                        tiki_row[k] = float("nan")
                    tiki_row["transfer_duty"] = float("nan")
                    tiki_row["transfer_spike"] = float("nan")

                # Hidden buffer stats (GPU)
                if hidden_w is not None:
                    h_flat = hidden_w.reshape(-1)
                    tiki_row["hidden_absmean"] = float(h_flat.abs().mean().item())
                    tiki_row["hidden_absmax"] = float(h_flat.abs().max().item())
                    tiki_row["hidden_below1_ratio"] = float(
                        (h_flat.abs() < self.dw_min).float().mean().item())
                    trunc = torch.trunc(h_flat)
                    trunc_nz = trunc != 0
                    tiki_row["hidden_trunc_nonzero_ratio"] = float(
                        trunc_nz.float().mean().item())
                    if trunc_nz.any():
                        tiki_row["hidden_trunc_meanabs"] = float(
                            trunc[trunc_nz].abs().mean().item())
                    else:
                        tiki_row["hidden_trunc_meanabs"] = 0.0
                    tiki_row["buffer_above_thresh_ratio"] = float(
                        (h_flat.abs() >= self.dw_min).float().mean().item())
                else:
                    for k in ["hidden_absmean", "hidden_absmax",
                              "hidden_below1_ratio",
                              "hidden_trunc_nonzero_ratio",
                              "hidden_trunc_meanabs",
                              "buffer_above_thresh_ratio"]:
                        tiki_row[k] = float("nan")

                # transfer_efficiency
                if dw_fast is not None and dw_slow is not None:
                    fast_mean = dw_fast.abs().mean().item()
                    slow_mean = dw_slow.abs().mean().item()
                    tiki_row["transfer_efficiency"] = float(
                        slow_mean / (fast_mean + EPS))
                else:
                    tiki_row["transfer_efficiency"] = float("nan")

                # Update prev for tiki (GPU tensors)
                if fast_w is not None:
                    _f2 = fast_w.reshape(-1)
                    layer["w_fast_prev"] = _f2[idx_t.to(_f2.device)].to(idx_t.device).clone()
                if slow_w is not None:
                    _s2 = slow_w.reshape(-1)
                    layer["w_slow_prev"] = _s2[idx_t.to(_s2.device)].to(idx_t.device).clone()
                layer["hidden_w_full"] = hidden_w.clone() if hidden_w is not None else None
            else:
                # Single mode: all tiki columns are nan
                for k in ["dw_fast_zero_ratio", "dw_fast_1lsb_ratio",
                          "dw_fast_absmean", "dw_slow_zero_ratio",
                          "dw_slow_1lsb_ratio", "dw_slow_absmean",
                          "dw_fast_vs_grad_cosine", "dw_fast_eff_lr_slope",
                          "hidden_absmean", "hidden_absmax",
                          "hidden_below1_ratio",
                          "hidden_trunc_nonzero_ratio",
                          "hidden_trunc_meanabs",
                          "transfer_duty", "transfer_spike",
                          "transfer_efficiency",
                          "buffer_above_thresh_ratio"]:
                    tiki_row[k] = float("nan")

            # Update prev for next step (GPU tensor)
            layer["w_eff_prev"] = w_eff_now.clone()

            # Resolve update management booleans for CSV
            if self.args.mode == "single":
                ubm = _resolve_bool_arg(self.args.update_bl_management, True)
                um = _resolve_bool_arg(self.args.update_management, True)
            else:
                ubm = _resolve_bool_arg(self.args.update_bl_management,
                                        not self.args.use_v2)
                um = _resolve_bool_arg(self.args.update_management,
                                       not self.args.use_v2)

            row = {
                "step": step,
                "mode": self.args.mode,
                "layer_idx": layer["layer_idx"],
                "sublayer": layer["sublayer"],
                "module_name": name,
                "dw_min": self.dw_min,
                "desired_bl": self.desired_bl,
                "sto_round_update": int(self.args.sto_round_update),
                "update_bl_management": int(ubm),
                "update_management": int(um),
                # Core weight delta (7)
                "dw_zero_ratio": eff_metrics["zero_ratio"],
                "dw_1lsb_ratio": eff_metrics["1lsb_ratio"],
                "dw_absmean": eff_metrics["absmean"],
                "min_nonzero_delta": eff_metrics["min_nonzero"],
                "dw_absmax": eff_metrics["absmax"],
                "dw_q90": eff_metrics["q90"],
                "dw_q99": eff_metrics["q99"],
                # Saturation (2)
                "pulse_sat_ratio": pulse_sat_ratio,
                "bound_sat_ratio": bound_sat_ratio,
                # Gradient proxy & update alignment (7)
                "grad_absmean": gp.get("grad_absmean", float("nan")),
                "grad_deadzone_ratio": gp.get("grad_deadzone_ratio",
                                               float("nan")),
                "update_vs_grad_cosine": update_vs_grad_cosine,
                "eff_lr_slope": eff_lr_slope,
                "BL_mean": gp.get("BL_mean", float("nan")),
                "BL_p99": gp.get("BL_p99", float("nan")),
                "BL_hit_ratio": gp.get("BL_hit_ratio", float("nan")),
                # 3-zone pulse classification (3)
                "pulse_under_frac": gp.get("pulse_under_frac", float("nan")),
                "pulse_ok_frac": gp.get("pulse_ok_frac", float("nan")),
                "pulse_over_frac": gp.get("pulse_over_frac", float("nan")),
                # TikiTaka transfer (1)
                "is_transfer_step": int(is_transfer) if self.is_tiki
                else float("nan"),
            }
            row.update(tiki_row)

            # New diagnostic metrics
            row["sign_mismatch_ratio"] = sign_mismatch_ratio
            row["rel_update_error"] = rel_update_error
            row["BL_fp_mean"] = BL_fp_mean
            row["BL_fp_p99"] = BL_fp_p99
            row["BL_fp_hit_ratio"] = BL_fp_hit_ratio
            row["dw_p50"] = dw_p50
            row["trace_every"] = self.trace_every

            self._step_rows.append(row)

        # Reset accumulator for next trace interval
        self._dw_fp_accum = {}

    def save_csvs(self, out_dir, tag):
        """Save step metrics and summary CSVs."""
        os.makedirs(out_dir, exist_ok=True)

        columns = [
            # Index/Config (10)
            "step", "mode", "layer_idx", "sublayer", "module_name",
            "dw_min", "desired_bl", "sto_round_update",
            "update_bl_management", "update_management",
            # Core weight delta (7)
            "dw_zero_ratio", "dw_1lsb_ratio", "dw_absmean",
            "min_nonzero_delta", "dw_absmax", "dw_q90", "dw_q99",
            # Saturation (2)
            "pulse_sat_ratio", "bound_sat_ratio",
            # Gradient proxy & update alignment (7)
            "grad_absmean", "grad_deadzone_ratio",
            "update_vs_grad_cosine", "eff_lr_slope",
            "BL_mean", "BL_p99", "BL_hit_ratio",
            # 3-zone pulse classification (3)
            "pulse_under_frac", "pulse_ok_frac", "pulse_over_frac",
            # TikiTaka fast/slow/buffer (15)
            "dw_fast_zero_ratio", "dw_fast_1lsb_ratio", "dw_fast_absmean",
            "dw_slow_zero_ratio", "dw_slow_1lsb_ratio", "dw_slow_absmean",
            "dw_fast_vs_grad_cosine", "dw_fast_eff_lr_slope",
            "hidden_absmean", "hidden_absmax", "hidden_below1_ratio",
            "hidden_trunc_nonzero_ratio", "hidden_trunc_meanabs",
            # TikiTaka transfer (5)
            "is_transfer_step", "transfer_duty", "transfer_spike",
            "transfer_efficiency", "buffer_above_thresh_ratio",
            # New diagnostic metrics
            "sign_mismatch_ratio", "rel_update_error",
            "BL_fp_mean", "BL_fp_p99", "BL_fp_hit_ratio",
            "dw_p50",
            # Config
            "trace_every",
        ]

        # Step metrics CSV
        df = pd.DataFrame(self._step_rows, columns=columns)
        step_path = os.path.join(out_dir, f"{tag}_step_metrics.csv")
        df.to_csv(step_path, index=False)
        print(f"  Saved {step_path} ({len(df)} rows)")

        # Summary CSV: groupby (layer_idx, sublayer) -> mean of numeric cols
        exclude_from_mean = {"step", "mode", "layer_idx", "sublayer",
                             "module_name", "dw_min", "desired_bl",
                             "sto_round_update", "update_bl_management",
                             "update_management"}
        numeric_cols = [c for c in columns if c not in exclude_from_mean]
        summary = df.groupby(["layer_idx", "sublayer"])[numeric_cols].mean(
            numeric_only=True)
        summary = summary.reset_index()
        sum_path = os.path.join(out_dir, f"{tag}_summary.csv")
        summary.to_csv(sum_path, index=False)
        print(f"  Saved {sum_path} ({len(summary)} rows)")

        # Print concise table
        self._print_summary_table(summary)

    def _print_summary_table(self, summary):
        """Print concise summary table to stdout."""
        print(f"\n{'=' * 100}")
        print(f"Weight Update Summary ({len(self._step_rows)} total records)")
        print(f"{'=' * 100}")
        print(f"{'Layer':>5} {'Sub':>5} "
              f"{'WZR':>8} {'W1LSB':>8} {'dwAbs':>10} "
              f"{'gDead':>8} {'cos':>8} {'slope':>8} "
              f"{'pSat':>8} {'bSat':>8}")
        print(f"{'-' * 5:>5} {'-' * 5:>5} "
              f"{'-' * 8:>8} {'-' * 8:>8} {'-' * 10:>10} "
              f"{'-' * 8:>8} {'-' * 8:>8} {'-' * 8:>8} "
              f"{'-' * 8:>8} {'-' * 8:>8}")

        for _, row in summary.iterrows():
            print(f"{int(row['layer_idx']):>5} {row['sublayer']:>5} "
                  f"{row['dw_zero_ratio']:>8.4f} "
                  f"{row['dw_1lsb_ratio']:>8.4f} "
                  f"{row['dw_absmean']:>10.6f} "
                  f"{row['grad_deadzone_ratio']:>8.4f} "
                  f"{row['update_vs_grad_cosine']:>8.4f} "
                  f"{row['eff_lr_slope']:>8.4f} "
                  f"{row['pulse_sat_ratio']:>8.4f} "
                  f"{row['bound_sat_ratio']:>8.4f}")
        print(f"{'=' * 100}")

        # TikiTaka transfer summary
        if self.is_tiki:
            df = pd.DataFrame(self._step_rows)
            ts = df[df["is_transfer_step"] == 1]
            if len(ts) > 0:
                print(f"\nTikiTaka Transfer Summary "
                      f"({len(ts)} transfer-step records)")
                print(f"{'-' * 60}")
                ts_summary = ts.groupby(
                    ["layer_idx", "sublayer"]
                )[["transfer_duty", "transfer_spike"]].mean()
                for (lidx, sub), row in ts_summary.iterrows():
                    duty_str = f"{row['transfer_duty']:.4f}"
                    spike_str = f"{row['transfer_spike']:.2f}"
                    print(f"  L{lidx:>2} {sub:>5}  "
                          f"duty={duty_str}  spike={spike_str}")
                print(f"{'-' * 60}")
        print()

    def get_summary(self):
        """Return dict of global means for sweep aggregation."""
        if not self._step_rows:
            return {}
        df = pd.DataFrame(self._step_rows)
        numeric = df.select_dtypes(include=[np.number])
        return numeric.mean().to_dict()


# =============================================================================
# Section 12: Run ID + run_one()
# =============================================================================

def _compute_run_id(run_args):
    """Compute deterministic 12-char hex run ID from config (sha1 hash).

    Excludes seed/tag/output-dir so that identical configs share the same ID.
    """
    cfg = {
        "mode": run_args.mode,
        "dw_min": run_args.dw_min,
        "desired_bl": run_args.desired_bl,
        "lr": run_args.lr,
        "steps": run_args.steps,
        "batch_size": run_args.batch_size,
        "seq_len": run_args.seq_len,
        "pulse_type": run_args.pulse_type,
        "sto_round_update": run_args.sto_round_update,
        "update_bl_management": str(run_args.update_bl_management),
        "update_management": str(run_args.update_management),
        "forward_perfect": run_args.forward_perfect,
        "backward_perfect": run_args.backward_perfect,
        "sample_k": run_args.sample_k,
        "exclude_ffn": getattr(run_args, 'exclude_ffn', False),
    }
    if run_args.mode == "tiki":
        cfg.update({
            "use_v2": run_args.use_v2,
            "transfer_every": run_args.transfer_every,
            "transfer_lr": run_args.transfer_lr,
            "fast_lr": run_args.fast_lr,
        })
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def run_one(args, dw_min_override=None, label=None, seed_override=None):
    """Run a single diagnostic session.

    Returns summary dict for sweep aggregation.
    """
    run_args = copy.deepcopy(args)
    if dw_min_override is not None:
        run_args.dw_min = dw_min_override

    run_seed = seed_override if seed_override is not None else run_args.seed
    data_seed = run_args.seed_data if run_args.seed_data is not None else run_seed
    model_seed = run_args.seed_model if run_args.seed_model is not None else run_seed

    # Collision-free directory: {output_dir}/{mode}/run_{hash}/{label_or_tag}/
    run_id = _compute_run_id(run_args)
    sub_label = label or run_args.tag or "default"
    out_dir = os.path.join(run_args.output_dir, run_args.mode,
                           f"run_{run_id}", sub_label)
    file_tag = os.path.basename(sub_label)

    # No-overwrite protection
    if os.path.exists(out_dir) and not getattr(run_args, 'overwrite', False):
        existing_csvs = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
        if existing_csvs:
            raise FileExistsError(
                f"Output directory {out_dir} already has results. "
                f"Use --overwrite to replace.")

    print(f"\n{'=' * 60}")
    print(f"[run_one] run_id={run_id}, label={sub_label}, mode={run_args.mode}, "
          f"dw_min={run_args.dw_min}, steps={run_args.steps}")
    print(f"[run_one] seeds: run={run_seed}, data={data_seed}, model={model_seed}")
    print(f"[run_one] out_dir: {out_dir}")
    print(f"{'=' * 60}")

    # JSON config dump
    config_dump = {
        "mode": run_args.mode,
        "dw_min": run_args.dw_min,
        "desired_bl": run_args.desired_bl,
        "lr": run_args.lr,
        "steps": run_args.steps,
        "batch_size": run_args.batch_size,
        "seed": run_seed,
        "seed_data": data_seed,
        "seed_model": model_seed,
        "pulse_type": run_args.pulse_type,
        "sto_round_update": run_args.sto_round_update,
        "update_bl_management": str(run_args.update_bl_management),
        "update_management": str(run_args.update_management),
        "trace_every": run_args.trace_every,
        "trace_layers": run_args.trace_layers,
        "trace_sublayers": run_args.trace_sublayers,
        "forward_perfect": run_args.forward_perfect,
        "backward_perfect": run_args.backward_perfect,
        "sample_k": run_args.sample_k,
        "exclude_ffn": getattr(run_args, 'exclude_ffn', False),
        "run_id": run_id,
    }
    if run_args.mode == "tiki":
        config_dump.update({
            "use_v2": run_args.use_v2,
            "transfer_every": run_args.transfer_every,
            "transfer_lr": run_args.transfer_lr,
            "fast_lr": run_args.fast_lr,
        })
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dump, f, indent=2)

    # Manifest at run level (one level up from sub_label)
    run_dir = os.path.dirname(out_dir)
    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = {"run_id": run_id, "mode": run_args.mode}
    manifest.update(config_dump)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    _set_all_seeds(data_seed)

    # Load tokenizer + data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loader = load_data(tokenizer, run_args.steps, run_args.batch_size,
                       run_args.seq_len, data_seed=data_seed)

    # Create model
    model = create_model(run_args, model_seed=model_seed)
    model.train()

    # Register hooks
    hook_active = [True]
    layer_infos, handles = register_xd_hooks(model, run_args.sample_k,
                                             hook_active,
                                             trace_args=run_args)

    # Print traced layer count
    n_traced = len(layer_infos)
    n_total_analog = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Tracing {n_traced}/{n_total_analog} analog layers")

    # Debug tiling sanity check
    if getattr(run_args, 'debug_tiling', False):
        print("  [DEBUG-TILING] Checking tile assembly order...")
        all_ok = True
        for lname, linfo in layer_infos.items():
            if not _debug_check_tile_order(linfo["module"], lname):
                all_ok = False
        if all_ok:
            print("  [DEBUG-TILING] All layers passed.")

    # Print trainable param count
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_analog_ctx = sum(1 for p in model.parameters()
                       if isinstance(p, AnalogContext) and p.requires_grad)
    print(f"  Trainable params: {n_trainable:,} ({n_analog_ctx} analog contexts)")

    # Create tracker
    tracker = WeightUpdateTracker(model, layer_infos, run_args)

    # Create optimizer
    optimizer = AnalogSGD(model.parameters(), lr=run_args.lr)
    optimizer.regroup_param_groups(model)

    # Training loop
    pbar = tqdm(enumerate(loader), total=min(run_args.steps, len(loader)),
                desc=f"[{file_tag}]")
    for step, batch in pbar:
        if step >= run_args.steps:
            break

        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # Forward + backward
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Compute gradient proxy (AFTER backward, BEFORE optimizer.step)
        grad_proxy_results = {}
        for name, info in layer_infos.items():
            grad_proxy_results[name] = compute_grad_proxy(
                info, run_args.lr, run_args.dw_min, run_args.desired_bl)

        # Optimizer step (analog update + ctx.reset())
        optimizer.step()

        # Record metrics after update (single read)
        tracker.record_after(step, grad_proxy_results)

        # Early warning after first few steps
        if step == min(2, run_args.steps - 1):
            recent = [r for r in tracker._step_rows if r["step"] == step]
            if recent:
                avg_wzr = np.mean([r["dw_zero_ratio"] for r in recent])
                if avg_wzr > 0.99:
                    print(f"  [WARNING] avg dw_zero_ratio={avg_wzr:.4f} at step {step}"
                          f" — updates may not be happening!")

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Cleanup hooks
    for h in handles:
        h.remove()

    # Save results
    tracker.save_csvs(out_dir, file_tag)
    summary = tracker.get_summary()

    # Cleanup
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# =============================================================================
# Section 13: run_dw_sweep() -- dw_min Sweep
# =============================================================================

def run_dw_sweep(args):
    """Run diagnostic across multiple dw_min values."""
    dw_list = [float(x.strip()) for x in args.dw_min_sweep.split(",")]
    sweep_tag = args.tag or "sweep"

    print(f"\n[Sweep] dw_min values: {dw_list}")

    sweep_rows = []
    for dw in dw_list:
        label = f"dw{dw:.4f}".replace(".", "p")
        summary = run_one(args, dw_min_override=dw, label=label)
        summary["dw_min"] = dw
        sweep_rows.append(summary)

    # Save sweep summary in mode directory
    sweep_dir = os.path.join(args.output_dir, args.mode, sweep_tag)
    os.makedirs(sweep_dir, exist_ok=True)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = os.path.join(sweep_dir, f"{sweep_tag}_sweep_summary.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\n[Sweep] Saved {sweep_path} ({len(sweep_df)} rows)")


# =============================================================================
# Section 14: Multi-seed Support
# =============================================================================

def _aggregate_seeds(args, base_dir, seed_list):
    """Aggregate per-seed step_metrics CSVs into summary statistics.

    Args:
        args: CLI args
        base_dir: directory containing seed{N}/ subdirectories
        seed_list: list of seed ints
    """
    from scipy.stats import t as t_dist

    all_dfs = []
    for s in seed_list:
        seed_dir = os.path.join(base_dir, f"seed{s}")
        csv_path = None
        if os.path.isdir(seed_dir):
            for fn in os.listdir(seed_dir):
                if fn.endswith("_step_metrics.csv"):
                    csv_path = os.path.join(seed_dir, fn)
                    break
        if csv_path is None or not os.path.exists(csv_path):
            print(f"  [WARNING] Missing CSV for seed {s} in {seed_dir}")
            continue
        df = pd.read_csv(csv_path)
        df["seed"] = s
        all_dfs.append(df)

    if not all_dfs:
        print("  [WARNING] No seed CSVs found for aggregation")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Identify numeric metric columns (exclude index/config cols)
    exclude_cols = {"step", "mode", "layer_idx", "sublayer", "module_name",
                    "dw_min", "desired_bl", "sto_round_update",
                    "update_bl_management", "update_management",
                    "seed", "trace_every"}
    metric_cols = [c for c in combined.columns
                   if c not in exclude_cols
                   and combined[c].dtype in [np.float64, np.float32, np.int64]]

    # Per-seed layer means
    seed_layer_means = combined.groupby(
        ["layer_idx", "sublayer", "seed"])[metric_cols].mean()
    seed_layer_means = seed_layer_means.reset_index()

    # Aggregate across seeds per (layer_idx, sublayer)
    n_seeds = len(seed_list)
    agg_rows = []
    for (lidx, sub), grp in seed_layer_means.groupby(["layer_idx", "sublayer"]):
        row = {"layer_idx": lidx, "sublayer": sub}
        n = len(grp)
        t_val = t_dist.ppf(0.975, max(n - 1, 1)) if n > 1 else float("nan")
        for col in metric_cols:
            vals = grp[col].dropna()
            if len(vals) > 0:
                m = float(vals.mean())
                sd = float(vals.std()) if len(vals) > 1 else 0.0
                ci_half = t_val * sd / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                row[f"{col}_mean"] = m
                row[f"{col}_std"] = sd
                row[f"{col}_ci95_lo"] = m - ci_half
                row[f"{col}_ci95_hi"] = m + ci_half
            else:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"] = float("nan")
                row[f"{col}_ci95_lo"] = float("nan")
                row[f"{col}_ci95_hi"] = float("nan")
        agg_rows.append(row)

    os.makedirs(base_dir, exist_ok=True)

    layer_df = pd.DataFrame(agg_rows)
    layer_path = os.path.join(base_dir, "aggregated_layer_summary.csv")
    layer_df.to_csv(layer_path, index=False)
    print(f"  Saved {layer_path} ({len(layer_df)} rows)")

    # Global summary: mean across all layers+seeds
    global_row = {}
    for col in metric_cols:
        vals = combined[col].dropna()
        if len(vals) > 0:
            global_row[f"{col}_mean"] = float(vals.mean())
            global_row[f"{col}_std"] = float(vals.std())
        else:
            global_row[f"{col}_mean"] = float("nan")
            global_row[f"{col}_std"] = float("nan")
    global_row["n_seeds"] = n_seeds
    global_df = pd.DataFrame([global_row])
    global_path = os.path.join(base_dir, "aggregated_global_summary.csv")
    global_df.to_csv(global_path, index=False)
    print(f"  Saved {global_path}")


def run_multi_seed(args):
    """Run diagnostic across multiple seeds and aggregate results."""
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]
    all_summaries = []

    # All seeds share the same run_id (same config)
    run_id = _compute_run_id(args)
    base_dir = os.path.join(args.output_dir, args.mode, f"run_{run_id}")

    print(f"\n[Multi-seed] Seeds: {seed_list}, run_id={run_id}")

    for seed in seed_list:
        label = f"seed{seed}"
        summary = run_one(args, label=label, seed_override=seed)
        summary["seed"] = seed
        all_summaries.append(summary)

    # Aggregate across seeds
    _aggregate_seeds(args, base_dir, seed_list)

    return all_summaries


# =============================================================================
# Section 14b: Combined dw_min sweep + multi-seed
# =============================================================================

def run_sweep_multiseed(args):
    """Run diagnostic across dw_min sweep × multiple seeds."""
    dw_list = [float(x.strip()) for x in args.dw_min_sweep.split(",")]
    seed_list = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"\n[Sweep×Seed] dw_min values: {dw_list}, seeds: {seed_list}")
    print(f"[Sweep×Seed] Total runs: {len(dw_list)} × {len(seed_list)} = {len(dw_list) * len(seed_list)}")

    sweep_rows = []
    for dw in dw_list:
        for seed in seed_list:
            label = f"dw{dw:.4f}_seed{seed}".replace(".", "p")
            summary = run_one(args, dw_min_override=dw, label=label, seed_override=seed)
            summary["dw_min"] = dw
            summary["seed"] = seed
            sweep_rows.append(summary)

    # Save combined sweep summary
    sweep_tag = args.tag or "sweep_multiseed"
    sweep_dir = os.path.join(args.output_dir, args.mode, sweep_tag)
    os.makedirs(sweep_dir, exist_ok=True)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = os.path.join(sweep_dir, f"{sweep_tag}_sweep_summary.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\n[Sweep×Seed] Saved {sweep_path} ({len(sweep_df)} rows)")


# =============================================================================
# Section 15: main()
# =============================================================================

def main():
    if args.dw_min_sweep and args.seeds:
        run_sweep_multiseed(args)
    elif args.dw_min_sweep:
        run_dw_sweep(args)
    elif args.seeds:
        run_multi_seed(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
