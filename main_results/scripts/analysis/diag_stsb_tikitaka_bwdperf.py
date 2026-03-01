# -*- coding: utf-8 -*-
"""Diagnostic: ALBERT STS-B TikiTaka backward_perfect Effectiveness.

Checks whether backward.is_perfect=True leads to effective learning by measuring
cosine similarity between gradients and weight updates (direction alignment).

Conditions:
  - ALBERT-base-v2, STS-B (regression, num_labels=1)
  - AnalogAdam lr=1.45e-3, target=attn only, non-target=digital(frozen)
  - units_in_mbatch=True, transfer_every=1
  - backward.is_perfect=True, CONVERT_NONTARGET=False

7-check verdict:
  1. A tile learning         : avg ||ΔA||_F > 1e-4
  2. B tile transfer          : avg ||ΔB||_F > 1e-4
  3. Gradient signal          : avg grad norm > 1e-6
  4. Grad-Update alignment    : avg cos(grad, ΔW) > 0
  5. A-B direction match      : avg cos(ΔA, ΔB) > 0
  6. Epoch consistency        : avg cos(ΔW_n, ΔW_{n-1}) > 0
  7. Metric improved          : final spearmanr > epoch-0 baseline

  5/7+ → PASS, 3-4 → PARTIAL, ≤2 → FAIL
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr as scipy_spearmanr

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs import (
    SingleRPUConfig, UnitCellRPUConfig, IOParameters, UpdateParameters,
)
from aihwkit.simulator.configs.compounds import TransferCompound
from aihwkit.simulator.configs.utils import BoundManagementType, NoiseManagementType
from aihwkit.optim.context import AnalogContext

os.environ["WANDB_MODE"] = "offline"

# =============================================================================
# Constants (no argparse)
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

MODEL_NAME = "albert/albert-base-v2"
TASK_NAME = "stsb"
NUM_LABELS = 1

BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
MAX_SEQ_LENGTH = 128
N_EPOCHS = 5
LR = 1.45e-3

FAST_LR = 1.0
TRANSFER_LR = 1.0
TRANSFER_EVERY = 1
UNITS_IN_MBATCH = True

BACKWARD_PERFECT = True
CONVERT_NONTARGET = False

RESULTS_DIR = "/data/results"
RESULTS_JSON = os.path.join(RESULTS_DIR, "diag_stsb_tikitaka_bwdperf.json")


# =============================================================================
# TikiTaka Device Functions
# =============================================================================

def _create_a_device():
    """A tile: 6T1C LinearStepDevice (fast, noisy)."""
    return LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=0.0, lifetime_dtod=0.0,
        reset=0.0, reset_dtod=0.0,
    )


def _create_b_device():
    """B tile: noise-free SoftBoundsDevice (slow, accurate)."""
    return SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0,
        up_down=0.0, up_down_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=False,
    )


def create_tikitaka_config():
    """TikiTaka v1 config with backward.is_perfect=True hardcoded."""
    rpu_config = UnitCellRPUConfig(
        device=TransferCompound(
            unit_cell_devices=[_create_a_device(), _create_b_device()],
            transfer_every=TRANSFER_EVERY,
            units_in_mbatch=UNITS_IN_MBATCH,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=TRANSFER_LR,
            fast_lr=FAST_LR,
            scale_transfer_lr=False,
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(),
        )
    )

    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    rpu_config.backward.is_perfect = True  # KEY SETTING

    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


# =============================================================================
# Model Creation (attn-only TikiTaka, FFN digital frozen)
# =============================================================================

def create_model():
    """ALBERT with TikiTaka on attention layers only. FFN stays digital (frozen)."""
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    # Reinitialize classifier with fixed seed
    if hasattr(model, "classifier"):
        torch.manual_seed(SEED)
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)

    always_digital = ["classifier", "albert.encoder.embedding_hidden_mapping_in"]
    all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    tikitaka_layers = [
        n for n in all_linear
        if "encoder" in n and "attention" in n
        and not any(d in n for d in always_digital)
    ]

    print(f"TikiTaka target layers: {tikitaka_layers}")

    # Single pass: convert only attn layers to TikiTaka
    tiki_config = create_tikitaka_config()
    tiki_exclude = [n for n in all_linear if n not in tikitaka_layers]
    model = convert_to_analog(model, tiki_config, exclude_modules=tiki_exclude)

    tikitaka_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"TikiTaka analog layers: {tikitaka_count}")

    # Freeze everything except: AnalogContext, classifier, LayerNorm, out_scaling
    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext):
            param.requires_grad = True
        elif "classifier" in name:
            param.requires_grad = True
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = True
        elif "out_scaling" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    return model.to(DEVICE)


# =============================================================================
# Data Loading (STS-B only)
# =============================================================================

def load_data(tokenizer):
    """Load STS-B dataset."""
    raw = load_dataset("nyu-mll/glue", TASK_NAME)

    def preprocess(examples):
        return tokenizer(
            examples["sentence1"], examples["sentence2"],
            max_length=MAX_SEQ_LENGTH, truncation=True,
        )

    remove_cols = [c for c in raw["train"].column_names if c != "label"]
    tokenized = raw.map(preprocess, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        tokenized["train"], batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, generator=torch.Generator().manual_seed(SEED),
    )
    eval_loader = DataLoader(
        tokenized["validation"], batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=collator,
    )
    return train_loader, eval_loader


# =============================================================================
# Evaluation (spearmanr + avg_loss)
# =============================================================================

def evaluate_model(model, eval_loader):
    """Returns (spearmanr, avg_loss)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    criterion = nn.MSELoss()

    with no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE).float()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()
            loss = criterion(logits, labels)

            all_preds.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item() * labels.size(0)

    model.train()
    n = len(all_labels)
    avg_loss = total_loss / n if n > 0 else 0.0
    corr = scipy_spearmanr(all_preds, all_labels)[0]
    return corr, avg_loss


# =============================================================================
# Weight / Gradient Extraction
# =============================================================================

def get_ab_weights(model):
    """Get A tile, B tile, and combined W for all TikiTaka layers.

    Each AnalogLinear may have multiple sub-tiles (e.g., 768x768 → 4x 384x384).
    We concatenate all sub-tiles into a single flattened tensor per module.
    """
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            all_A, all_B, all_W = [], [], []
            for tile in module.analog_tiles():
                if isinstance(tile.rpu_config, UnitCellRPUConfig):
                    inner = tile.tile
                    hp_names = inner.get_hidden_parameter_names()
                    hp = inner.get_hidden_parameters()
                    a_idx = hp_names.index("hidden_weights_0")
                    b_idx = hp_names.index("hidden_weights_1")
                    combined_w, _ = tile.get_weights()
                    all_A.append(hp[a_idx].clone().cpu().flatten())
                    all_B.append(hp[b_idx].clone().cpu().flatten())
                    all_W.append(combined_w.clone().cpu().flatten())
            if all_A:
                result[name] = {
                    "A": torch.cat(all_A),
                    "B": torch.cat(all_B),
                    "W": torch.cat(all_W),
                }
    return result


def get_out_scaling(model):
    """Get out_scaling values for all TikiTaka layers."""
    result = {}
    for name, param in model.named_parameters():
        if "out_scaling" in name:
            result[name] = param.detach().clone().cpu()
    return result


def setup_gradient_hooks(model):
    """Register forward + backward hooks on AnalogLinear layers.

    Forward hook captures the input tensor; backward hook captures grad_output.
    Together they let us compute the implied weight gradient:
        grad_W = grad_output^T @ input  (summed over batch*seq)
    which has the same shape as the weight matrix.

    Returns (grad_store, hooks).
    """
    grad_store = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            def make_fwd_hook(ln):
                def hook_fn(mod, inp, out):
                    if inp[0] is not None:
                        grad_store.setdefault(ln, {})["input"] = inp[0].detach().clone()
                return hook_fn

            def make_bwd_hook(ln):
                def hook_fn(mod, gi, go):
                    if go[0] is not None:
                        g = go[0].detach()
                        entry = grad_store.setdefault(ln, {})
                        entry["grad_output"] = g.clone()
                        entry["norm"] = g.norm().item()
                        entry["mean_abs"] = g.abs().mean().item()
                        # Compute implied weight gradient: grad_W = go^T @ x
                        if "input" in entry:
                            x = entry["input"]
                            # Reshape to 2D: (batch*seq, features)
                            go_2d = g.reshape(-1, g.shape[-1])   # (N, out_features)
                            x_2d = x.reshape(-1, x.shape[-1])    # (N, in_features)
                            grad_W = go_2d.t() @ x_2d            # (out_features, in_features)
                            entry["grad_W"] = grad_W.cpu()
                return hook_fn

            hooks.append(module.register_forward_hook(make_fwd_hook(name)))
            hooks.append(module.register_full_backward_hook(make_bwd_hook(name)))

    return grad_store, hooks


# =============================================================================
# Cosine Similarity Computations
# =============================================================================

def compute_cosine_sim(a, b):
    """Cosine similarity between two tensors (flattened)."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    if a_flat.norm() < 1e-12 or b_flat.norm() < 1e-12:
        return 0.0
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def accumulate_grad_W(grad_accum, grad_store, layer_names):
    """Accumulate per-step grad_W into epoch-level totals (for direction comparison).

    Also returns per-step grad norms.
    """
    step_norms = []
    for ln in layer_names:
        if ln in grad_store and "grad_W" in grad_store[ln]:
            gw = grad_store[ln]["grad_W"].flatten().float()
            if ln not in grad_accum:
                grad_accum[ln] = gw.clone()
            else:
                grad_accum[ln] += gw
            step_norms.append(grad_store[ln]["norm"])
    return step_norms


# =============================================================================
# Verdict
# =============================================================================

def compute_verdict(epoch_diagnostics, baseline_spearmanr, final_spearmanr):
    """7-check verdict. Returns (verdict_str, checks_dict)."""
    n_epochs = len(epoch_diagnostics)
    checks = {}

    # Collect per-epoch averages across layers
    all_dA_norms = []
    all_dB_norms = []
    all_grad_norms = []
    all_cos_grad_dW = []
    all_cos_dA_dB = []

    for ep_data in epoch_diagnostics:
        layer_data = ep_data["layer_diagnostics"]
        for ln, ld in layer_data.items():
            all_dA_norms.append(ld["dA_norm_epoch"])
            all_dB_norms.append(ld["dB_norm_epoch"])
            all_cos_grad_dW.append(ld["avg_cos_grad_dW"])
            all_cos_dA_dB.append(ld["cos_dA_dB_epoch"])
        all_grad_norms.append(ep_data["avg_grad_norm"])

    # Check 1: A tile learning
    avg_dA = np.mean(all_dA_norms) if all_dA_norms else 0.0
    checks["a_tile_learning"] = {"value": avg_dA, "threshold": 1e-4, "pass": avg_dA > 1e-4}

    # Check 2: B tile transfer
    avg_dB = np.mean(all_dB_norms) if all_dB_norms else 0.0
    checks["b_tile_transfer"] = {"value": avg_dB, "threshold": 1e-4, "pass": avg_dB > 1e-4}

    # Check 3: Gradient signal
    avg_grad = np.mean(all_grad_norms) if all_grad_norms else 0.0
    checks["gradient_signal"] = {"value": avg_grad, "threshold": 1e-6, "pass": avg_grad > 1e-6}

    # Check 4: Grad-Update alignment
    avg_cos_gw = np.mean(all_cos_grad_dW) if all_cos_grad_dW else 0.0
    checks["grad_update_alignment"] = {"value": avg_cos_gw, "threshold": 0.0, "pass": avg_cos_gw > 0}

    # Check 5: A-B direction match
    avg_cos_ab = np.mean(all_cos_dA_dB) if all_cos_dA_dB else 0.0
    checks["ab_direction_match"] = {"value": avg_cos_ab, "threshold": 0.0, "pass": avg_cos_ab > 0}

    # Check 6: Epoch consistency (cos(ΔW_n, ΔW_{n-1}))
    epoch_consistency_vals = []
    for ep_data in epoch_diagnostics:
        for ln, ld in ep_data["layer_diagnostics"].items():
            if "cos_dW_prev_epoch" in ld and ld["cos_dW_prev_epoch"] is not None:
                epoch_consistency_vals.append(ld["cos_dW_prev_epoch"])
    avg_epoch_cons = np.mean(epoch_consistency_vals) if epoch_consistency_vals else 0.0
    checks["epoch_consistency"] = {"value": avg_epoch_cons, "threshold": 0.0, "pass": avg_epoch_cons > 0}

    # Check 7: Metric improved
    improved = final_spearmanr > baseline_spearmanr
    checks["metric_improved"] = {
        "value": final_spearmanr, "baseline": baseline_spearmanr,
        "pass": improved,
    }

    n_pass = sum(1 for c in checks.values() if c["pass"])
    if n_pass >= 5:
        verdict = "PASS"
    elif n_pass >= 3:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return verdict, checks, n_pass


# =============================================================================
# Short name helper
# =============================================================================

def short_name(full_name):
    """Extract short layer name (query/key/value/dense)."""
    return full_name.split(".")[-1]


# =============================================================================
# Main
# =============================================================================

def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 80)
    print("DIAGNOSTIC: ALBERT STS-B TikiTaka backward_perfect Effectiveness")
    print("=" * 80)
    print(f"Model: {MODEL_NAME} | Task: {TASK_NAME} | backward.is_perfect={BACKWARD_PERFECT}")
    print(f"BSZ={BATCH_SIZE} | LR={LR} | Epochs={N_EPOCHS} | TE={TRANSFER_EVERY} | UIM={UNITS_IN_MBATCH}")
    print(f"fast_lr={FAST_LR} | transfer_lr={TRANSFER_LR}")
    print(f"Device: {DEVICE}")
    print()

    # --- Load data ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)} | Eval batches: {len(eval_loader)}")

    # --- Create model ---
    model = create_model()

    # --- Optimizer ---
    optimizer = AnalogAdam(model.parameters(), lr=LR)
    optimizer.regroup_param_groups()
    criterion = nn.MSELoss()

    # --- Gradient hooks ---
    grad_store, hooks = setup_gradient_hooks(model)

    # --- Get layer names ---
    layer_names = list(get_ab_weights(model).keys())
    snames = {ln: short_name(ln) for ln in layer_names}
    print(f"Monitored layers: {[snames[ln] for ln in layer_names]}")
    print()

    # --- Baseline evaluation (epoch 0) ---
    baseline_spearmanr, baseline_loss = evaluate_model(model, eval_loader)
    print(f"Baseline (before training): spearmanr={baseline_spearmanr:.4f} | loss={baseline_loss:.4f}")
    print()

    # --- Snapshots ---
    prev_epoch_weights = get_ab_weights(model)
    prev_epoch_out_scaling = get_out_scaling(model)

    # Storage for epoch-to-epoch ΔW (for consistency check)
    prev_epoch_dW = None  # dict: layer_name -> ΔW tensor (from previous epoch)

    epoch_diagnostics = []
    all_train_losses = []

    # --- Training loop ---
    for epoch in range(1, N_EPOCHS + 1):
        print(f"{'=' * 80}")
        print(f"=== Epoch {epoch}/{N_EPOCHS} ===")
        print(f"{'=' * 80}")

        model.train()
        epoch_losses = []
        grad_accum = {}  # accumulated grad_W per layer over the epoch
        step_grad_norms = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE).float()

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Accumulate grad_W before optimizer step (captures gradient direction)
            norms = accumulate_grad_W(grad_accum, grad_store, layer_names)
            step_grad_norms.extend(norms)
            grad_store.clear()

            optimizer.step()
            epoch_losses.append(loss.item())

        # --- End-of-epoch snapshots ---
        curr_epoch_weights = get_ab_weights(model)
        curr_out_scaling = get_out_scaling(model)

        avg_train_loss = np.mean(epoch_losses)
        all_train_losses.append(avg_train_loss)
        avg_grad_norm = np.mean(step_grad_norms) if step_grad_norms else 0.0

        # --- Evaluate ---
        eval_spearmanr, eval_loss = evaluate_model(model, eval_loader)

        print(f"Train Loss: {avg_train_loss:.4f} | Eval Spearmanr: {eval_spearmanr:.4f} | "
              f"Eval Loss: {eval_loss:.4f} | Avg Grad Norm: {avg_grad_norm:.2e}")

        # --- Per-layer diagnostics ---
        layer_diag = {}
        curr_epoch_dW = {}

        header = (f"{'Layer':<8s} | {'||ΔA||_F':>10s} | {'||ΔB||_F':>10s} | {'||ΔW||_F':>10s} | "
                  f"{'cos(g,ΔW)':>10s} | {'cos(ΔA,ΔB)':>10s} | {'out_scl_Δ':>10s}")
        print(header)
        print("-" * len(header))

        for ln in layer_names:
            sn = snames[ln]
            dA = curr_epoch_weights[ln]["A"] - prev_epoch_weights[ln]["A"]
            dB = curr_epoch_weights[ln]["B"] - prev_epoch_weights[ln]["B"]
            dW = curr_epoch_weights[ln]["W"] - prev_epoch_weights[ln]["W"]

            dA_norm = dA.norm().item()
            dB_norm = dB.norm().item()
            dW_norm = dW.norm().item()
            cos_dA_dB = compute_cosine_sim(dA, dB)

            # cos(accumulated_grad, ΔW) — direction alignment over the epoch
            avg_cos_gw = 0.0
            if ln in grad_accum:
                avg_cos_gw = compute_cosine_sim(grad_accum[ln], dW.flatten())

            # out_scaling change
            out_scale_delta = 0.0
            for os_name in curr_out_scaling:
                if sn in os_name or ln.replace(".", "_") in os_name.replace(".", "_"):
                    if os_name in prev_epoch_out_scaling:
                        out_scale_delta = (curr_out_scaling[os_name] - prev_epoch_out_scaling[os_name]).norm().item()
                    break
            # Fallback: compute over all out_scaling params matching this analog layer
            if out_scale_delta == 0.0:
                for os_name in curr_out_scaling:
                    if os_name in prev_epoch_out_scaling:
                        d = (curr_out_scaling[os_name] - prev_epoch_out_scaling[os_name]).norm().item()
                        if d > out_scale_delta:
                            out_scale_delta = d

            # Epoch-to-epoch consistency
            curr_epoch_dW[ln] = dW.clone()
            cos_dW_prev = None
            if prev_epoch_dW is not None and ln in prev_epoch_dW:
                cos_dW_prev = compute_cosine_sim(dW, prev_epoch_dW[ln])

            print(f"{sn:<8s} | {dA_norm:>10.2e} | {dB_norm:>10.2e} | {dW_norm:>10.2e} | "
                  f"{avg_cos_gw:>+10.4f} | {cos_dA_dB:>+10.4f} | {out_scale_delta:>10.2e}")

            layer_diag[ln] = {
                "short_name": sn,
                "dA_norm_epoch": dA_norm,
                "dB_norm_epoch": dB_norm,
                "dW_norm_epoch": dW_norm,
                "avg_cos_grad_dW": avg_cos_gw,
                "cos_dA_dB_epoch": cos_dA_dB,
                "out_scale_delta": out_scale_delta,
                "cos_dW_prev_epoch": cos_dW_prev,
            }

        # Epoch-to-epoch direction consistency
        if prev_epoch_dW is not None:
            print()
            print("Epoch-to-Epoch Direction Consistency:")
            parts = []
            for ln in layer_names:
                sn = snames[ln]
                val = layer_diag[ln]["cos_dW_prev_epoch"]
                if val is not None:
                    parts.append(f"{sn}={val:+.4f}")
            print("  cos(ΔW_curr, ΔW_prev): " + "  ".join(parts))

        epoch_diagnostics.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "eval_spearmanr": eval_spearmanr,
            "eval_loss": eval_loss,
            "avg_grad_norm": avg_grad_norm,
            "layer_diagnostics": layer_diag,
        })

        prev_epoch_weights = curr_epoch_weights
        prev_epoch_out_scaling = curr_out_scaling
        prev_epoch_dW = curr_epoch_dW
        print()

    # --- Final evaluation ---
    final_spearmanr, final_loss = evaluate_model(model, eval_loader)
    print(f"Final: spearmanr={final_spearmanr:.4f} | loss={final_loss:.4f}")
    print()

    # --- VERDICT ---
    verdict, checks, n_pass = compute_verdict(epoch_diagnostics, baseline_spearmanr, final_spearmanr)

    print("=" * 80)
    print("=== VERDICT ===")
    print("=" * 80)
    for ck_name, ck_data in checks.items():
        status = "PASS" if ck_data["pass"] else "FAIL"
        if "threshold" in ck_data:
            print(f"  [{status}] {ck_name}: {ck_data['value']:.6f} (threshold: {ck_data['threshold']})")
        elif "baseline" in ck_data:
            print(f"  [{status}] {ck_name}: {ck_data['value']:.4f} vs baseline {ck_data['baseline']:.4f}")
        else:
            print(f"  [{status}] {ck_name}: {ck_data['value']:.6f}")

    print()
    print(f"  Result: [{verdict}] ({n_pass}/7 checks passed)")
    print("=" * 80)

    # --- Save JSON ---
    json_result = {
        "config": {
            "model": MODEL_NAME,
            "task": TASK_NAME,
            "backward_is_perfect": BACKWARD_PERFECT,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "transfer_every": TRANSFER_EVERY,
            "units_in_mbatch": UNITS_IN_MBATCH,
            "fast_lr": FAST_LR,
            "transfer_lr": TRANSFER_LR,
            "convert_nontarget": CONVERT_NONTARGET,
        },
        "baseline_spearmanr": baseline_spearmanr,
        "final_spearmanr": final_spearmanr,
        "per_epoch": [],
        "verdict": verdict,
        "n_checks_passed": n_pass,
        "checks": {k: {kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                        for kk, vv in v.items()}
                   for k, v in checks.items()},
    }

    for ep_data in epoch_diagnostics:
        ep_json = {
            "epoch": ep_data["epoch"],
            "train_loss": ep_data["train_loss"],
            "eval_spearmanr": ep_data["eval_spearmanr"],
            "eval_loss": ep_data["eval_loss"],
            "avg_grad_norm": ep_data["avg_grad_norm"],
            "layers": {},
        }
        for ln, ld in ep_data["layer_diagnostics"].items():
            ep_json["layers"][ld["short_name"]] = {
                "dA_norm": ld["dA_norm_epoch"],
                "dB_norm": ld["dB_norm_epoch"],
                "dW_norm": ld["dW_norm_epoch"],
                "cos_grad_dW": ld["avg_cos_grad_dW"],
                "cos_dA_dB": ld["cos_dA_dB_epoch"],
                "out_scale_delta": ld["out_scale_delta"],
                "cos_dW_prev_epoch": ld["cos_dW_prev_epoch"],
            }
        json_result["per_epoch"].append(ep_json)

    # Custom encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(RESULTS_JSON, "w") as f:
        json.dump(json_result, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {RESULTS_JSON}")

    # Cleanup hooks
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
