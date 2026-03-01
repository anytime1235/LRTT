#!/data/venvs/aihwkit_gpu/bin/python
# coding=utf-8
"""Quick test for high learning rate (lr=5e-1)."""

import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam

# LRTT config imports
sys.path.insert(0, '/home/jovyan/work/LRTT/src')
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

# Import config function from main script
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
from sweep_sixt1c_lora_squad_adam import (
    create_sixt1c_lora_config,
    load_squad_data,
    evaluate_squad,
)

# Test parameters
TEST_LR = 5e-3  # 0.005 - Upper bound of search range
LORA_ALPHA = 1.0
BATCH_SIZE = 256
MODEL_NAME = "google/mobilebert-uncased"
RANK = 8
REINIT_GAIN = 0.1
TARGET_MODULES = ["query", "key", "value"]
SEED = 42
NUM_BATCHES_TO_TEST = 10  # Test first 10 batches only

print("="*60)
print("HIGH LEARNING RATE TEST")
print("="*60)
print(f"Learning Rate: {TEST_LR}")
print(f"LoRA Alpha: {LORA_ALPHA}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Testing first {NUM_BATCHES_TO_TEST} batches")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# Load data
print("\nLoading data...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_loader, eval_features, eval_examples = load_squad_data(tokenizer)
print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

# Create model
print("\nCreating model...")
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# Get layers to exclude
all_linear = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
exclude.append("qa_outputs")

# Convert to analog
rpu_config = create_sixt1c_lora_config(
    rank=RANK,
    lora_alpha=LORA_ALPHA,
    reinit_gain=REINIT_GAIN,
)
model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

# Set requires_grad
for name, param in model.named_parameters():
    is_target = any(t in name for t in TARGET_MODULES)
    param.requires_grad = is_target or "qa_outputs" in name

model = model.to(device)

# Create optimizer
print(f"\nCreating optimizer with lr={TEST_LR}...")
optimizer = AnalogAdam(model.parameters(), lr=TEST_LR)
optimizer.regroup_param_groups()

# Initial evaluation
print("\nInitial evaluation...")
init_f1, init_em = evaluate_squad(model, eval_features, eval_examples, tokenizer, device)
print(f"Initial F1: {init_f1:.4f}, EM: {init_em:.4f}")

# Train for a few batches
print(f"\nTraining for {NUM_BATCHES_TO_TEST} batches...")
model.train()
batch_losses = []
nan_detected = False

try:
    for i, batch in enumerate(train_loader):
        if i >= NUM_BATCHES_TO_TEST:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ NaN/Inf detected at batch {i}!")
            print(f"   Loss: {loss.item()}")
            nan_detected = True
            break

        loss.backward()

        # Check gradients
        max_grad = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                max_grad = max(max_grad, grad_norm)
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"\n❌ NaN/Inf gradient in {name} at batch {i}!")
                    nan_detected = True
                    break

        if nan_detected:
            break

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_losses.append(loss.item())
        print(f"Batch {i}: loss={loss.item():.4f}, max_grad={max_grad:.4f}")

except Exception as e:
    print(f"\n❌ Exception occurred: {e}")
    import traceback
    traceback.print_exc()
    nan_detected = True

# Final evaluation if training succeeded
if not nan_detected and len(batch_losses) == NUM_BATCHES_TO_TEST:
    print("\n✅ Training completed without NaN/Inf")
    print(f"Loss progression: {batch_losses}")
    print(f"Average loss: {np.mean(batch_losses):.4f}")

    print("\nFinal evaluation...")
    final_f1, final_em = evaluate_squad(model, eval_features, eval_examples, tokenizer, device)
    print(f"Final F1: {final_f1:.4f}, EM: {final_em:.4f}")
    print(f"F1 change: {final_f1 - init_f1:.4f}")
else:
    print("\n❌ Training failed with NaN/Inf!")
    print(f"Completed {len(batch_losses)}/{NUM_BATCHES_TO_TEST} batches")
    if batch_losses:
        print(f"Loss progression before failure: {batch_losses}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
