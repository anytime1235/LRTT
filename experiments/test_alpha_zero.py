"""
Test LRTT with lora_alpha=0 to verify if explosion/NaN occurs.
"""
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "lora_training_glue"))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

from aihwkit.optim import AnalogSGD
from aihwkit.optim.context import AnalogContext

print("=" * 80)
print("TEST: lora_alpha = 0.0 (exactly zero)")
print("=" * 80)

# Setup
MODEL_NAME = "google/mobilebert-uncased"
ALPHA = 0.0  # EXACTLY ZERO
ANALOG_LR = 0.0001
CLASSIFIER_LR = 0.01

print(f"\nConfig:")
print(f"  lora_alpha: {ALPHA} (ZERO!)")
print(f"  analog_lr: {ANALOG_LR}")
print(f"  classifier_lr: {CLASSIFIER_LR}")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Convert with alpha=0
print("\nConverting to LRTT with alpha=0...")
config = create_lrtt_lora_config(rank=8, lora_alpha=ALPHA, use_floating_point=False)
model = convert_model_to_lrtt_lora(model, config, target_modules=["query", "key", "value"])
model = model.cuda()

# Setup optimizer
param_groups = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if isinstance(param, AnalogContext) or "out_scaling_alpha" in name:
        param_groups.append({"params": [param], "lr": ANALOG_LR})
    else:
        param_groups.append({"params": [param], "lr": CLASSIFIER_LR})

optimizer = AnalogSGD(param_groups, lr=ANALOG_LR)

# Get some data
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = load_dataset("glue", "sst2")
sample = dataset["train"].select(range(8))

def preprocess(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

sample = sample.map(preprocess, batched=True)
batch = {k: torch.tensor(v).cuda() for k, v in sample.to_dict().items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
batch["labels"] = torch.tensor(sample["label"]).cuda()

# Forward + backward
print("\nRunning forward + backward...")
model.train()
outputs = model(**batch)
loss = outputs.loss

print(f"  Loss: {loss.item()}")

loss.backward()

# Check gradients
print("\nChecking gradients on analog tiles:")
for name, param in model.named_parameters():
    if isinstance(param, AnalogContext):
        if hasattr(param, 'grad') and param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm = {grad_norm}")
        else:
            print(f"  {name}: grad is None")
        break  # Just check first one

# Try optimizer step
print("\nTrying optimizer step...")
try:
    optimizer.step()
    print("  ✓ Optimizer step succeeded")
except Exception as e:
    print(f"  ✗ Optimizer step failed: {e}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("If no NaN/explosion occurred, then alpha=0 is actually safe!")
print("The explosion might be from other causes (initialization, etc.)")
print("=" * 80)
