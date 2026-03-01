"""Diagnose: compare model output (logits) between digital vs analog non-target layers.

Tests:
  A) TikiTaka target + digital non-target (old approach)
  B) TikiTaka target + SingleRPU non-target (new approach)
  C) All digital (baseline)

If A produces reasonable logits but B doesn't, the issue is SingleRPU forward.
"""
import sys, torch, gc
sys.path.insert(0, '/data')

from torch import nn

DEVICE = torch.device("cuda")

# Fixed input
torch.manual_seed(42)
batch = {
    "input_ids": torch.randint(0, 1000, (4, 64), device=DEVICE),
    "attention_mask": torch.ones(4, 64, dtype=torch.long, device=DEVICE),
    "labels": torch.randint(0, 2, (4,), device=DEVICE),
}

params = {"transfer_every": 16, "transfer_lr": 1.0, "fast_lr": 0.026}


def check_model(model, label):
    model.eval()
    with torch.no_grad():
        out = model(**batch)
    logits = out.logits
    loss_val = nn.CrossEntropyLoss()(logits, batch['labels']).item()
    print(f"\n  [{label}]")
    print(f"    logits shape: {logits.shape}")
    print(f"    logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")
    print(f"    logits mean/std: {logits.mean().item():.4f} / {logits.std().item():.4f}")
    print(f"    loss: {loss_val:.4f}")

    # Check intermediate: get hidden states from last encoder layer
    # Just check if logits are reasonable
    return loss_val


# ============================================================
# A) TikiTaka + digital non-target (OLD approach)
# ============================================================
print("=" * 70)
print("  A) TikiTaka target + DIGITAL non-target (old approach)")
print("=" * 70)

import optuna_mobilebert_glue_tiki as mod
mod.TASK_NAME = 'mrpc'
mod.LORA_TARGET = 'qkv'
mod.HEAD_LAYER = 'train'

# Temporarily patch create_model to skip Pass 2 (SingleRPU)
import types
original_create_model = mod.create_model

def create_model_digital_nontarget(params):
    """Like original but skip SingleRPU conversion (keep non-target as digital frozen)."""
    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator.configs import SingleRPUConfig, UnitCellRPUConfig
    from aihwkit.optim.context import AnalogContext
    from transformers import AutoConfig, AutoModelForSequenceClassification

    num_labels = mod.TASK_TO_NUM_LABELS[mod.TASK_NAME]
    model_config = AutoConfig.from_pretrained(mod.MODEL_NAME, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(mod.MODEL_NAME, config=model_config)

    # Reinitialize classifier with FIXED seed
    if hasattr(model, 'classifier'):
        torch.manual_seed(mod.SEED)
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)

    always_digital = ["classifier", "embedding_transformation"]

    def is_tikitaka_target(layer_name):
        if any(d in layer_name for d in always_digital):
            return False
        if "encoder" not in layer_name:
            return False
        cat = mod._classify_encoder_layer(layer_name)
        if mod.LORA_TARGET == "qkv":
            return cat == 'attention'
        return False

    all_linear_names = mod.list_linear_layers(model)
    tikitaka_layers = [n for n in all_linear_names if is_tikitaka_target(n)]

    # Only Pass 1: TikiTaka for target layers
    from aihwkit.nn.conversion import convert_to_analog
    if tikitaka_layers:
        tiki_config = mod.create_tikitaka_config(
            transfer_every=int(params["transfer_every"]),
            transfer_lr=params["transfer_lr"],
            fast_lr=params["fast_lr"],
        )
        tiki_exclude = [n for n in all_linear_names if n not in tikitaka_layers]
        model = convert_to_analog(model, tiki_config, exclude_modules=tiki_exclude)

    # Freeze non-target digital layers
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

    return model

model_a = create_model_digital_nontarget(params)
model_a.to(DEVICE)
loss_a = check_model(model_a, "TikiTaka + Digital non-target")

# Count layers
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig, UnitCellRPUConfig
tiki_a = sum(1 for m in model_a.modules() if isinstance(m, AnalogLinear))
dig_a = sum(1 for m in model_a.modules() if isinstance(m, nn.Linear) and not isinstance(m, AnalogLinear))
print(f"    Analog: {tiki_a}, Digital: {dig_a}")

del model_a; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# B) TikiTaka + SingleRPU non-target (NEW approach)
# ============================================================
print("\n" + "=" * 70)
print("  B) TikiTaka target + SingleRPU non-target (new approach)")
print("=" * 70)

# Reload to get fresh module state
import importlib
importlib.reload(mod)
mod.TASK_NAME = 'mrpc'
mod.LORA_TARGET = 'qkv'
mod.HEAD_LAYER = 'train'

model_b = mod.create_model(params)
model_b.to(DEVICE)
loss_b = check_model(model_b, "TikiTaka + SingleRPU non-target")

tiki_b = 0
nt_b = 0
for m in model_b.modules():
    if isinstance(m, AnalogLinear):
        for tile in m.analog_tiles():
            if isinstance(tile.rpu_config, UnitCellRPUConfig):
                tiki_b += 1
            elif isinstance(tile.rpu_config, SingleRPUConfig):
                nt_b += 1
            break
print(f"    TikiTaka: {tiki_b}, SingleRPU: {nt_b}")

del model_b; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# C) All digital baseline
# ============================================================
print("\n" + "=" * 70)
print("  C) All digital baseline")
print("=" * 70)

from transformers import AutoConfig, AutoModelForSequenceClassification
num_labels = mod.TASK_TO_NUM_LABELS['mrpc']
model_config = AutoConfig.from_pretrained(mod.MODEL_NAME, num_labels=num_labels)
model_c = AutoModelForSequenceClassification.from_pretrained(mod.MODEL_NAME, config=model_config)

# Same classifier init
torch.manual_seed(mod.SEED)
nn.init.normal_(model_c.classifier.weight, mean=0.0, std=0.02)
if model_c.classifier.bias is not None:
    nn.init.zeros_(model_c.classifier.bias)

model_c.to(DEVICE)
loss_c = check_model(model_c, "All digital")

del model_c; gc.collect(); torch.cuda.empty_cache()


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  A) TikiTaka + Digital:    loss = {loss_a:.4f}")
print(f"  B) TikiTaka + SingleRPU:  loss = {loss_b:.4f}")
print(f"  C) All Digital:           loss = {loss_c:.4f}")
print(f"  Ratio B/A: {loss_b/loss_a:.1f}x")
print(f"  Ratio B/C: {loss_b/loss_c:.1f}x")
