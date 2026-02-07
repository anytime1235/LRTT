#!/usr/bin/env python
"""
Diagnose Sixt1c LoRA divergence — activation/gradient logging script.

Runs 10 training steps on CoLA with Sixt1c LoRA and logs per-step:
  - lora_A / lora_B weight range (min, max)
  - lora_A / lora_B gradient norm
  - training loss

Compares two configurations:
  A) Default (weight_scaling_omega=0.0) — expected to diverge
  B) Fixed  (weight_scaling_omega=1.0, columnwise=True) — expected to converge

Usage:
    python experiments/diagnose_sixt1c.py
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.tiles.inference_torch import TorchInferenceTile
from aihwkit.simulator.configs import TorchInferenceRPUConfig
from aihwkit.simulator.presets.utils import IOParameters
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    WeightClipType,
    WeightRemapType,
    NoiseManagementType,
)
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel

from related_functions import (
    list_analog_linear_layers,
    replace_layer,
    convert_lora_layers_only_to_analog,
)

NUM_STEPS = 50
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Gradient threshold: initial gradients are naturally large due to random classifier
# head producing huge logits. Real divergence shows as gradients *increasing* over
# steps rather than decreasing. We use a very high threshold and check the trend.
GRAD_EXPLOSION_THRESHOLD = 1e10


# ---------------------------------------------------------------------------
# Config generators
# ---------------------------------------------------------------------------

def gen_pcm_config():
    """PCM RPU config for base_layer (same as run_glue.py gen_rpu_config)."""
    rpu = TorchInferenceRPUConfig()
    rpu.mapping.digital_bias = True
    rpu.mapping.weight_scaling_omega = 1.0
    rpu.mapping.weight_scaling_columnwise = True
    rpu.mapping.learn_out_scaling = True
    rpu.mapping.out_scaling_columnwise = True
    rpu.modifier.std_dev = 0.067
    rpu.modifier.type = WeightModifierType.ADD_NORMAL
    rpu.remap.type = WeightRemapType.CHANNELWISE_SYMMETRIC
    rpu.forward = IOParameters()
    rpu.forward.out_noise = 0.04
    rpu.forward.is_perfect = False
    rpu.forward.inp_res = 1 / (2**8 - 2)
    rpu.forward.out_res = 1 / (2**8 - 2)
    rpu.clip.type = WeightClipType.LAYER_GAUSSIAN
    rpu.clip.sigma = 3
    rpu.noise_model = PCMLikeNoiseModel(g_max=25.0)
    return rpu


def gen_sixt1c_config_default():
    """Default sixt1c config (omega=0.0 → no backward hook → diverges)."""
    rpu = TorchInferenceRPUConfig()
    rpu.mapping.digital_bias = True
    rpu.mapping.weight_scaling_omega = 0.0
    rpu.mapping.weight_scaling_columnwise = False
    rpu.mapping.learn_out_scaling = False
    rpu.mapping.out_scaling_columnwise = False
    rpu.modifier.std_dev = 0.0
    rpu.modifier.type = WeightModifierType.ADD_NORMAL
    rpu.remap.type = WeightRemapType.NONE
    rpu.forward = IOParameters()
    rpu.forward.out_noise = 0.0
    rpu.forward.is_perfect = False
    rpu.forward.inp_res = 1 / (2**8 - 2)
    rpu.forward.out_res = 1 / (2**8 - 2)
    rpu.clip.type = WeightClipType.NONE
    rpu.noise_model = None
    rpu.drift_compensation = None
    return rpu


def gen_sixt1c_config_fixed():
    """Fixed sixt1c config (omega=1.0 → backward hook registered → converges)."""
    rpu = TorchInferenceRPUConfig()
    rpu.mapping.digital_bias = True
    rpu.mapping.weight_scaling_omega = 1.0
    rpu.mapping.weight_scaling_columnwise = True
    rpu.mapping.learn_out_scaling = False
    rpu.mapping.out_scaling_columnwise = False
    rpu.modifier.std_dev = 0.0
    rpu.modifier.type = WeightModifierType.ADD_NORMAL
    rpu.remap.type = WeightRemapType.NONE
    rpu.forward = IOParameters()
    rpu.forward.out_noise = 0.0
    rpu.forward.is_perfect = False
    rpu.forward.inp_res = 1 / (2**8 - 2)
    rpu.forward.out_res = 1 / (2**8 - 2)
    rpu.clip.type = WeightClipType.NONE
    rpu.noise_model = None
    rpu.drift_compensation = None
    return rpu


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_sixt1c_model(sixt1c_rpu_config, label=""):
    """Build MobileBERT + LoRA + PCM base + Sixt1c LoRA layers."""
    set_seed(SEED)

    config = AutoConfig.from_pretrained(
        "google/mobilebert-uncased", num_labels=2, finetuning_task="cola"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased", config=config, ignore_mismatched_sizes=True
    )

    # Add LoRA
    peft_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=["dense", "query", "key", "value",
                        "qa_outputs", "embedding_transformation"],
    )
    digital_model = get_peft_model(model, peft_config)

    # Step 1: convert entire model to PCM analog
    pcm_rpu = gen_pcm_config()
    model = convert_to_analog(digital_model, pcm_rpu, tile_module_class=TorchInferenceTile)

    # Step 2: replace lora layers back to digital
    analog_names = list_analog_linear_layers(model)
    lora_names = [n for n in analog_names if "base_layer" not in n]
    for name in lora_names:
        replace_layer(model, digital_model, name)

    # Step 3: convert lora layers to sixt1c analog
    model = convert_lora_layers_only_to_analog(model, sixt1c_rpu_config)

    # Freeze base_layer
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear) and "base_layer" in name:
            for param in module.parameters():
                param.requires_grad = False

    model.to(DEVICE)
    return model


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def get_cola_dataloader():
    """Load CoLA training data and return a simple batch iterator."""
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
    dataset = load_dataset("nyu-mll/glue", "cola", split="train")

    def tokenize(examples):
        return tokenizer(
            examples["sentence"], padding="max_length",
            max_length=128, truncation=True,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def collect_lora_stats(model):
    """Collect weight range and grad norm for all lora_A / lora_B layers."""
    stats = {"lora_A": [], "lora_B": []}
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            key = None
            if "lora_A" in name:
                key = "lora_A"
            elif "lora_B" in name:
                key = "lora_B"
            if key is None:
                continue

            tile = module.analog_module.tile
            w = tile.get_weights()
            w_min, w_max = w.min().item(), w.max().item()

            grad_norm = float("nan")
            if tile.weight.grad is not None:
                grad_norm = tile.weight.grad.norm().item()

            stats[key].append({
                "name": name,
                "w_min": w_min,
                "w_max": w_max,
                "grad_norm": grad_norm,
            })
    return stats


def print_step_summary(step, loss, stats):
    """Print a one-line-per-layer summary."""
    print(f"\n--- Step {step} | loss={loss:.6f} ---")
    for key in ("lora_A", "lora_B"):
        if not stats[key]:
            continue
        norms = [s["grad_norm"] for s in stats[key] if not np.isnan(s["grad_norm"])]
        w_mins = [s["w_min"] for s in stats[key]]
        w_maxs = [s["w_max"] for s in stats[key]]
        avg_grad = np.mean(norms) if norms else float("nan")
        max_grad = np.max(norms) if norms else float("nan")
        print(
            f"  {key}: w=[{min(w_mins):.6f}, {max(w_maxs):.6f}]  "
            f"grad_norm avg={avg_grad:.6f} max={max_grad:.6f}"
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_diagnosis(config_name, sixt1c_rpu_config):
    """Run NUM_STEPS training steps and log diagnostics."""
    print(f"\n{'=' * 70}")
    print(f"  CONFIG: {config_name}")
    print(f"{'=' * 70}")

    model = build_sixt1c_model(sixt1c_rpu_config, label=config_name)

    optimizer = AnalogAdam(model.parameters(), lr=LEARNING_RATE)
    optimizer.regroup_param_groups(model)

    dataloader = get_cola_dataloader()
    data_iter = iter(dataloader)

    results = []

    model.train()
    for step in range(NUM_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n  *** Step {step}: loss is NaN/Inf! Stopping. ***")
            results.append({"step": step, "loss": float("nan"), "diverged": True})
            break

        loss.backward()

        # Collect stats after backward (before optimizer.step)
        stats = collect_lora_stats(model)
        print_step_summary(step, loss.item(), stats)

        # Check for gradient explosion
        all_grads = []
        for key in ("lora_A", "lora_B"):
            all_grads.extend(
                s["grad_norm"] for s in stats[key]
                if not np.isnan(s["grad_norm"])
            )
        max_grad = max(all_grads) if all_grads else float("nan")
        if np.isnan(max_grad) or max_grad > GRAD_EXPLOSION_THRESHOLD:
            print(f"\n  *** Step {step}: gradient explosion (max_grad={max_grad:.2e})! Stopping. ***")
            results.append({"step": step, "loss": loss.item(), "max_grad": max_grad, "diverged": True})
            break

        optimizer.step()
        results.append({
            "step": step,
            "loss": loss.item(),
            "max_grad": max_grad,
            "diverged": False,
        })

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY for {config_name}")
    print(f"{'=' * 70}")
    for r in results:
        status = "DIVERGED" if r.get("diverged") else "ok"
        print(f"  step={r['step']:3d}  loss={r['loss']:.6f}  max_grad={r.get('max_grad', float('nan')):.2e}  [{status}]")

    # Trend analysis
    if len(results) >= 3:
        first_3_loss = np.mean([r["loss"] for r in results[:3]])
        last_3_loss = np.mean([r["loss"] for r in results[-3:]])
        first_3_grad = np.mean([r.get("max_grad", 0) for r in results[:3] if not np.isnan(r.get("max_grad", 0))])
        last_3_grad = np.mean([r.get("max_grad", 0) for r in results[-3:] if not np.isnan(r.get("max_grad", 0))])
        print(f"\n  TREND: loss first_3_avg={first_3_loss:.4f} → last_3_avg={last_3_loss:.4f}")
        print(f"  TREND: grad first_3_avg={first_3_grad:.2e} → last_3_avg={last_3_grad:.2e}")
        if last_3_loss > first_3_loss * 2:
            print(f"  ** Loss INCREASING — divergence detected **")
        elif last_3_loss < first_3_loss * 0.5:
            print(f"  ** Loss DECREASING — training converging **")
        else:
            print(f"  ** Loss roughly stable **")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Sixt1c LoRA Divergence Diagnosis")
    print(f"  Device: {DEVICE}")
    print(f"  Steps: {NUM_STEPS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("=" * 70)

    # --- Config A: Default (expected to diverge) ---
    results_default = run_diagnosis(
        "A) Default (omega=0.0, no backward hook)",
        gen_sixt1c_config_default(),
    )

    # --- Config B: Fixed (expected to converge) ---
    results_fixed = run_diagnosis(
        "B) Fixed (omega=1.0, columnwise=True, backward hook active)",
        gen_sixt1c_config_fixed(),
    )

    # --- Comparison ---
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    def last_loss(results):
        return results[-1]["loss"] if results else float("nan")

    def diverged(results):
        return any(r.get("diverged") for r in results)

    def loss_trend(results):
        if len(results) < 3:
            return "insufficient data"
        first = np.mean([r["loss"] for r in results[:3]])
        last = np.mean([r["loss"] for r in results[-3:]])
        if np.isnan(last) or last > first * 2:
            return "DIVERGING"
        elif last < first * 0.5:
            return "CONVERGING"
        return "STABLE"

    print(f"  A) Default:  final_loss={last_loss(results_default):.6f}  diverged={diverged(results_default)}  trend={loss_trend(results_default)}")
    print(f"  B) Fixed:    final_loss={last_loss(results_fixed):.6f}  diverged={diverged(results_fixed)}  trend={loss_trend(results_fixed)}")

    trend_a = loss_trend(results_default)
    trend_b = loss_trend(results_fixed)

    if trend_a == "DIVERGING" and trend_b in ("CONVERGING", "STABLE"):
        print("\n  CONCLUSION: backward hook (weight_scaling_omega=1.0) fixes the divergence!")
    elif trend_a in ("CONVERGING", "STABLE") and trend_b in ("CONVERGING", "STABLE"):
        print("\n  NOTE: Both converging/stable. Divergence may need more steps to manifest.")
        print("  The real divergence in run_glue.py occurs ~epoch 0.37 (~100 steps with full CoLA).")
    elif diverged(results_default) and diverged(results_fixed):
        print("\n  NOTE: Both hit gradient explosion threshold. The initial loss is very large")
        print("  due to random classifier head — this is normal. Check if the loss *trends*")
        print("  differ between configs (converging vs diverging) over more steps.")
    else:
        print(f"\n  NOTE: A={trend_a}, B={trend_b}. Unexpected pattern — investigate further.")


if __name__ == "__main__":
    main()
