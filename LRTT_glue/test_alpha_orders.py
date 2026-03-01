#!/data/venvs/aihwkit_gpu/bin/python
# coding=utf-8
"""Test LoRA Alpha at different orders of magnitude."""

import os
import sys
import torch
import torch.nn as nn

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
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

# Import functions
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
from sweep_sixt1c_lora_squad_adam import (
    create_sixt1c_lora_config,
    load_squad_data,
)

BATCH_SIZE = 256
MODEL_NAME = "google/mobilebert-uncased"
RANK = 8
REINIT_GAIN = 0.1
TARGET_MODULES = ["query", "key", "value"]
SEED = 42
NUM_BATCHES = 10
FIXED_LR = 0.001  # Fixed LR, vary alpha

# Test alpha at different orders of magnitude
ALPHA_TEST_VALUES = [
    0.01,   # 10^-2
    0.1,    # 10^-1
    0.3,    # ~10^-0.5
    1.0,    # 10^0
    3.0,    # ~10^0.5
    10.0,   # 10^1
]

print("="*80)
print("LoRA ALPHA ORDER OF MAGNITUDE TEST")
print("="*80)
print(f"Fixed LR: {FIXED_LR}")
print(f"Testing Alpha values: {ALPHA_TEST_VALUES}")
print(f"Batches per test: {NUM_BATCHES}")
print(f"Batch Size: {BATCH_SIZE}")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data once
print("\nLoading data...")
set_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_loader, _, _ = load_squad_data(tokenizer)
print(f"Train batches: {len(train_loader)}")

results = []

for alpha in ALPHA_TEST_VALUES:
    product = FIXED_LR * alpha

    print("\n" + "="*80)
    print(f"TEST: Alpha = {alpha:.2f}")
    print("="*80)
    print(f"LR: {FIXED_LR}, Alpha: {alpha}, Product: {product:.6f}")

    try:
        torch.cuda.empty_cache()
        set_seed(SEED)

        # Create model
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

        # Get exclude list
        all_linear = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
        exclude = [n for n in all_linear if not any(t in n for t in TARGET_MODULES)]
        exclude.append("qa_outputs")

        # Convert to analog
        rpu_config = create_sixt1c_lora_config(rank=RANK, lora_alpha=alpha, reinit_gain=REINIT_GAIN)
        model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

        # Set requires_grad
        for name, param in model.named_parameters():
            is_target = any(t in name for t in TARGET_MODULES)
            param.requires_grad = is_target or "qa_outputs" in name

        model = model.to(device)

        # Create optimizer
        optimizer = AnalogAdam(model.parameters(), lr=FIXED_LR)
        optimizer.regroup_param_groups()

        # Train
        model.train()
        losses = []
        grads = []
        failed = False

        for i, batch in enumerate(train_loader):
            if i >= NUM_BATCHES:
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
                print(f"  Batch {i}: ❌ NaN/Inf")
                failed = True
                break

            loss.backward()

            max_grad = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    g = param.grad.norm().item()
                    max_grad = max(max_grad, g)
                    if not torch.isfinite(param.grad).all():
                        print(f"  Batch {i}: ❌ Gradient NaN/Inf")
                        failed = True
                        break

            if failed:
                break

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())
            grads.append(max_grad)

            if i % 2 == 0:
                print(f"  Batch {i:2d}: loss={loss.item():10.2f}, grad={max_grad:10.2f}")

        if not failed and len(losses) == NUM_BATCHES:
            avg_loss = np.mean(losses)
            max_loss = np.max(losses)
            print(f"\n✅ SUCCESS")
            print(f"  Avg loss: {avg_loss:.2f}, Max loss: {max_loss:.2f}")
            print(f"  Avg grad: {np.mean(grads):.2f}, Max grad: {np.max(grads):.2f}")
            result = "✅ PASS"
        else:
            print(f"\n❌ FAILED at batch {len(losses)}")
            result = "❌ FAIL"

        results.append({
            'alpha': alpha,
            'product': product,
            'result': result,
            'batches': len(losses),
            'avg_loss': np.mean(losses) if losses else None,
        })

        del model, optimizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n❌ EXCEPTION: {str(e)[:100]}")
        results.append({
            'alpha': alpha,
            'product': product,
            'result': "❌ ERROR",
            'batches': 0,
            'avg_loss': None,
        })
        torch.cuda.empty_cache()

# Summary
print("\n" + "="*80)
print("SUMMARY: Alpha Order-of-Magnitude Test")
print("="*80)
print(f"{'Alpha':<10} {'Product':<12} {'Result':<12} {'Batches':<10} {'Avg Loss':<12}")
print("-"*80)
for r in results:
    loss_str = f"{r['avg_loss']:.2f}" if r['avg_loss'] else "N/A"
    print(f"{r['alpha']:<10.2f} {r['product']:<12.6f} {r['result']:<12} {r['batches']:<10} {loss_str:<12}")

pass_count = sum(1 for r in results if r['result'] == "✅ PASS")
print(f"\n{pass_count}/{len(results)} tests passed")

if pass_count > 0:
    safe_alphas = [r['alpha'] for r in results if r['result'] == "✅ PASS"]
    print(f"\n✅ Safe Alpha range (at LR={FIXED_LR}): {min(safe_alphas):.2f} ~ {max(safe_alphas):.2f}")
    safe_products = [r['product'] for r in results if r['result'] == "✅ PASS"]
    print(f"   Safe Product range: {min(safe_products):.6f} ~ {max(safe_products):.6f}")
print("="*80)
