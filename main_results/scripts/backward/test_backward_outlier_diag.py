#!/usr/bin/env python3
"""Diagnostic: Backward gradient outlier & quantization loss in LRTT LoRA.

Hooks AnalogTile.backward() to capture gradient (d_input) statistics.
Measures outlier ratio and simulates DAC quantization damage.
Then compares training loss between default vs perfect backward.
"""

import os, sys, math
import torch
import torch.nn as nn
import numpy as np

os.environ["WANDB_MODE"] = "offline"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_STEPS = 30

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.tiles.analog import AnalogTile

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, set_seed
from datasets import load_dataset

# ============================================================
# Hook AnalogTile.backward to capture d_input statistics
# ============================================================
_orig_backward = AnalogTile.backward
_cap = {"on": False, "recs": []}

def _hooked_bwd(self, d_input, ctx=None):
    if not _cap["on"]:
        return _orig_backward(self, d_input, ctx)

    # Capture gradient statistics BEFORE analog backward MVM
    with torch.no_grad():
        inp_abs = d_input.abs()
        row_max = inp_abs.max(dim=-1, keepdim=True)[0]  # per-sample max
        global_max = inp_abs.max().item()
        global_mean = inp_abs.mean().item()
        global_median = inp_abs.median().item()

        bwd_io = self.rpu_config.backward
        rec = {
            "shape": list(d_input.shape),
            "absmax": global_max,
            "mean": global_mean,
            "median": global_median,
            "std": d_input.std().item(),
            "outlier_max_mean": global_max / max(global_mean, 1e-15),
            "outlier_max_median": global_max / max(global_median, 1e-15),
            "is_perfect": bwd_io.is_perfect,
            "inp_res": bwd_io.inp_res,
            "inp_bound": bwd_io.inp_bound,
            "noise_mgmt": str(bwd_io.noise_management),
        }

        # Simulate ABS_MAX + DAC quantization damage
        if not bwd_io.is_perfect and bwd_io.inp_res > 0:
            from aihwkit.simulator.parameters.enums import NoiseManagementType
            if bwd_io.noise_management == NoiseManagementType.ABS_MAX:
                # Per-row scaling to [-1,1]
                rm = row_max.clone()
                rm[rm <= 0] = 1.0
                scaled = d_input / rm

                res = bwd_io.inp_res if bwd_io.inp_res <= 1.0 else 1.0 / bwd_io.inp_res
                n_levels = int(round(1.0 / res)) if res > 0 else 0
                half_step = res / 2.0

                # Values below half a quantization step → killed (rounded to 0)
                killed_mask = scaled.abs() < half_step
                rec["killed_pct"] = killed_mask.float().mean().item() * 100
                rec["dac_levels"] = n_levels

                # Distribution of scaled values: how concentrated near 0?
                pct_below_10pct = (scaled.abs() < 0.1).float().mean().item() * 100
                pct_below_1pct = (scaled.abs() < 0.01).float().mean().item() * 100
                rec["pct_below_10pct"] = pct_below_10pct
                rec["pct_below_1pct"] = pct_below_1pct
            else:
                rec["killed_pct"] = 0.0
                rec["dac_levels"] = 0
                rec["pct_below_10pct"] = 0.0
                rec["pct_below_1pct"] = 0.0
        else:
            rec["killed_pct"] = 0.0
            rec["dac_levels"] = 0
            rec["pct_below_10pct"] = 0.0
            rec["pct_below_1pct"] = 0.0

    _cap["recs"].append(rec)
    return _orig_backward(self, d_input, ctx)

AnalogTile.backward = _hooked_bwd


# ============================================================
# Model & Data
# ============================================================

def create_lrtt_config(backward_perfect=False):
    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3,
        write_noise_std=0.0, mean_bound_reference=True, lifetime=0.0,
    )
    c_device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=False,
    )
    device_config = PythonLRTTDevice(
        rank=16, transfer_every=10000000, lora_alpha=1.0,
        reinit_gain=1.0, reinit_mode="decay", decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.1
    device_config.units_in_mbatch = True
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = True
    device_config.combined_out_scaling = True

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    if backward_perfect:
        rpu_config.backward.is_perfect = True

    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


def create_model(backward_perfect=False):
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        "albert/albert-base-v2", num_labels=1)
    torch.manual_seed(SEED)
    nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
    nn.init.zeros_(model.classifier.bias)

    lrtt_patterns = ["attention"]
    always_digital = ["classifier", "albert.encoder.embedding_hidden_mapping_in"]
    all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    exclude = [n for n in all_linear
               if any(d in n for d in always_digital)
               or "encoder" not in n
               or not any(p in n for p in lrtt_patterns)]
    exclude += always_digital
    exclude = list(set(exclude))

    lrtt_config = create_lrtt_config(backward_perfect=backward_perfect)
    model = convert_to_analog(model, lrtt_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            param.requires_grad = True
        elif "tile_c" in name and "analog_ctx" in name:
            param.requires_grad = True
        elif "out_scaling" in name:
            param.requires_grad = True
        elif "classifier" in name:
            param.requires_grad = True
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model.to(DEVICE)


def load_stsb_data():
    tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
    raw = load_dataset("nyu-mll/glue", "stsb")
    def preprocess(ex):
        return tokenizer(ex["sentence1"], ex["sentence2"], max_length=128, truncation=True)
    remove_cols = [c for c in raw["train"].column_names if c != "label"]
    tokenized = raw.map(preprocess, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column("label", "labels")
    train = tokenized["train"].shuffle(seed=SEED).select(range(min(480, len(tokenized["train"]))))
    collator = DataCollatorWithPadding(tokenizer)
    loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True, collate_fn=collator,
                                          generator=torch.Generator().manual_seed(SEED))
    return loader


# ============================================================
# Run
# ============================================================

def run_exp(backward_perfect, loader):
    tag = "PERFECT" if backward_perfect else "DEFAULT"
    print(f"\n{'='*70}")
    print(f"  [{tag}] backward.is_perfect = {backward_perfect}")
    print(f"{'='*70}")

    set_seed(SEED)
    model = create_model(backward_perfect=backward_perfect)

    # Print backward IO config for verification
    for n, m in model.named_modules():
        if hasattr(m, 'rpu_config'):
            bwd = m.rpu_config.backward
            print(f"  Tile [{n[:50]}]: bwd.is_perfect={bwd.is_perfect}, inp_res={bwd.inp_res:.6f}, noise_mgmt={bwd.noise_management}")
            break

    optimizer = AnalogAdam(model.parameters(), lr=1.45e-3, weight_decay=0.0)
    optimizer.regroup_param_groups()

    lrtt_tile_ids = set()
    for m in model.modules():
        if hasattr(m, 'tile_a'):
            lrtt_tile_ids.update([id(m.tile_a), id(m.tile_b), id(m.tile_c)])
    lrtt_lr = 1.45e-3 * 0.1
    for group in optimizer.param_groups:
        for p in group["params"]:
            if hasattr(p, 'analog_tile') and id(p.analog_tile) in lrtt_tile_ids:
                group["lr"] = lrtt_lr
                p.analog_tile.set_learning_rate(lrtt_lr)

    model.train()
    losses = []
    all_recs = []

    for step, batch in enumerate(loader):
        if step >= N_STEPS:
            break
        inputs = {k: v.to(DEVICE) for k, v in batch.items()
                  if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}

        optimizer.zero_grad()
        _cap["on"] = False
        _cap["recs"] = []
        outputs = model(**inputs)
        loss = outputs.loss

        _cap["on"] = True
        loss.backward()
        _cap["on"] = False

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        step_recs = list(_cap["recs"])
        all_recs.extend(step_recs)

        if step % 5 == 0:
            n = len(step_recs)
            if n > 0 and not backward_perfect:
                k = np.mean([r["killed_pct"] for r in step_recs])
                o = np.mean([r["outlier_max_mean"] for r in step_recs])
                print(f"  Step {step:3d} | Loss: {loss.item():.4f} | BWD calls: {n} | "
                      f"Avg killed: {k:.1f}% | Avg outlier: {o:.1f}x")
            else:
                print(f"  Step {step:3d} | Loss: {loss.item():.4f} | BWD calls: {n}")

    # Summary
    print(f"\n--- [{tag}] Summary ---")
    print(f"  Steps: {len(losses)}")
    print(f"  Loss: start={losses[0]:.4f}, end={losses[-1]:.4f}")
    print(f"  Total backward MVM calls: {len(all_recs)}")

    if all_recs and not backward_perfect:
        outlier = [r["outlier_max_mean"] for r in all_recs]
        outlier_med = [r["outlier_max_median"] for r in all_recs]
        killed = [r["killed_pct"] for r in all_recs]
        below10 = [r["pct_below_10pct"] for r in all_recs]
        below1 = [r["pct_below_1pct"] for r in all_recs]

        print(f"\n  === Backward Gradient Outlier Analysis ===")
        print(f"  Outlier ratio (max/mean):    mean={np.mean(outlier):8.1f}, p50={np.median(outlier):8.1f}, p95={np.percentile(outlier,95):8.1f}, max={np.max(outlier):8.1f}")
        print(f"  Outlier ratio (max/median):  mean={np.mean(outlier_med):8.1f}, p50={np.median(outlier_med):8.1f}, p95={np.percentile(outlier_med,95):8.1f}, max={np.max(outlier_med):8.1f}")
        print(f"\n  === DAC Quantization Damage ===")
        print(f"  Values killed (→0) by DAC:   mean={np.mean(killed):6.1f}%, p50={np.median(killed):6.1f}%, p95={np.percentile(killed,95):6.1f}%, max={np.max(killed):6.1f}%")
        print(f"  Values < 10% of row-max:     mean={np.mean(below10):6.1f}%, p50={np.median(below10):6.1f}%")
        print(f"  Values < 1% of row-max:      mean={np.mean(below1):6.1f}%, p50={np.median(below1):6.1f}%")

        # Per-shape breakdown
        from collections import Counter
        shapes = Counter([str(r["shape"]) for r in all_recs])
        print(f"\n  === Per-shape breakdown (top 5) ===")
        for shape_str, cnt in shapes.most_common(5):
            recs_s = [r for r in all_recs if str(r["shape"]) == shape_str]
            k = np.mean([r["killed_pct"] for r in recs_s])
            o = np.mean([r["outlier_max_mean"] for r in recs_s])
            print(f"    {shape_str:30s} (n={cnt:4d}): killed={k:5.1f}%, outlier={o:6.1f}x")

    return losses, all_recs


def main():
    print("Loading STS-B data (small subset)...")
    loader = load_stsb_data()

    losses_def, recs_def = run_exp(backward_perfect=False, loader=loader)
    losses_perf, _ = run_exp(backward_perfect=True, loader=loader)

    print(f"\n{'='*70}")
    print(f"  LOSS COMPARISON: Default vs Perfect Backward ({N_STEPS} steps)")
    print(f"{'='*70}")
    print(f"  {'Step':>5}  {'Default':>10}  {'Perfect':>10}  {'Diff':>10}")
    for i in range(min(len(losses_def), len(losses_perf))):
        d = losses_perf[i] - losses_def[i]
        print(f"  {i:5d}  {losses_def[i]:10.4f}  {losses_perf[i]:10.4f}  {d:+10.4f}")

    # Final verdict
    quant_recs = [r for r in recs_def if not r["is_perfect"]]
    if quant_recs:
        avg_kill = np.mean([r["killed_pct"] for r in quant_recs])
        avg_outlier = np.mean([r["outlier_max_mean"] for r in quant_recs])
        print(f"\n  VERDICT:")
        print(f"    Avg gradient values killed by backward DAC: {avg_kill:.1f}%")
        print(f"    Avg outlier ratio: {avg_outlier:.1f}x")
        if avg_kill > 30:
            print(f"    >>> SEVERE: {avg_kill:.0f}% gradient info destroyed. backward_perfect strongly recommended.")
        elif avg_kill > 10:
            print(f"    >>> MODERATE: {avg_kill:.0f}% gradient info lost. backward_perfect should help.")
        elif avg_kill > 3:
            print(f"    >>> MILD: {avg_kill:.0f}% gradient info lost. backward_perfect may help slightly.")
        else:
            print(f"    >>> MINIMAL: {avg_kill:.0f}%. Backward quantization is not the bottleneck.")
    else:
        print(f"\n  No quantized backward MVMs found.")


if __name__ == "__main__":
    main()
