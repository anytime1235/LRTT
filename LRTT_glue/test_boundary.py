#!/data/venvs/aihwkit_gpu/bin/python
# coding=utf-8
"""Test boundary cases in overlap region."""

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

# Import functions from main script
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
from sweep_sixt1c_lora_squad_adam import (
    create_sixt1c_lora_config,
    load_squad_data,
    evaluate_squad,
)

BATCH_SIZE = 256
MODEL_NAME = "google/mobilebert-uncased"
RANK = 8
REINIT_GAIN = 0.1
TARGET_MODULES = ["query", "key", "value"]
SEED = 42
NUM_BATCHES_TO_TEST = 20  # Test more batches for boundary cases

# Overlap region test cases
TEST_CASES = [
    # (lr, alpha, description)
    (0.003, 0.3, "Min overlap (Mode 1 max LR, Mode 2 min Alpha)"),
    (0.004, 0.35, "Center overlap"),
    (0.005, 0.4, "Max overlap (Mode 1 max LR, Mode 2 max Alpha)"),
    (0.003, 0.4, "Mode 1 territory (low LR, mid Alpha)"),
    (0.005, 0.3, "Mode 2 territory (mid LR, low Alpha)"),
]

print("="*80)
print("BOUNDARY REGION STABILITY TEST")
print("="*80)
print(f"Overlap Region: LR [0.003, 0.005], Alpha [0.3, 0.4]")
print(f"Testing {NUM_BATCHES_TO_TEST} batches per case")
print(f"Batch Size: {BATCH_SIZE}")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# Load data once
print("\nLoading data...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_loader, eval_features, eval_examples = load_squad_data(tokenizer)
print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

results = []

for test_lr, test_alpha, description in TEST_CASES:
    product = test_lr * test_alpha

    print("\n" + "="*80)
    print(f"TEST CASE: {description}")
    print("="*80)
    print(f"LR: {test_lr:.4f}, Alpha: {test_alpha:.2f}")
    print(f"Product (LR×Alpha): {product:.6f}")
    print(f"Safety: {'✅ SAFE' if product < 0.004 else '⚠️ BOUNDARY' if product < 0.006 else '❌ RISKY'}")
    print("-"*80)

    try:
        set_seed(SEED)

        # Create model
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

        # Get layers to exclude
        all_linear = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
        exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
        exclude.append("qa_outputs")

        # Convert to analog
        rpu_config = create_sixt1c_lora_config(
            rank=RANK,
            lora_alpha=test_alpha,
            reinit_gain=REINIT_GAIN,
        )
        model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

        # Set requires_grad
        for name, param in model.named_parameters():
            is_target = any(t in name for t in TARGET_MODULES)
            param.requires_grad = is_target or "qa_outputs" in name

        model = model.to(device)

        # Create optimizer
        optimizer = AnalogAdam(model.parameters(), lr=test_lr)
        optimizer.regroup_param_groups()

        # Train for test batches
        model.train()
        batch_losses = []
        max_grads = []
        nan_detected = False
        failed_batch = -1

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
                print(f"  Batch {i:2d}: ❌ NaN/Inf loss = {loss.item()}")
                nan_detected = True
                failed_batch = i
                break

            loss.backward()

            # Check gradients
            max_grad = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    max_grad = max(max_grad, grad_norm)
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"  Batch {i:2d}: ❌ NaN/Inf gradient in {name}")
                        nan_detected = True
                        failed_batch = i
                        break

            if nan_detected:
                break

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_losses.append(loss.item())
            max_grads.append(max_grad)

            # Print every 5 batches
            if i % 5 == 0 or i == NUM_BATCHES_TO_TEST - 1:
                print(f"  Batch {i:2d}: loss={loss.item():10.4f}, max_grad={max_grad:12.2f}")

        # Summary
        if not nan_detected and len(batch_losses) == NUM_BATCHES_TO_TEST:
            result = "✅ PASS"
            avg_loss = np.mean(batch_losses)
            max_loss = np.max(batch_losses)
            avg_grad = np.mean(max_grads)
            max_grad_overall = np.max(max_grads)

            print(f"\n✅ SUCCESS - All {NUM_BATCHES_TO_TEST} batches completed")
            print(f"  Loss: avg={avg_loss:.4f}, max={max_loss:.4f}")
            print(f"  Grad: avg={avg_grad:.2f}, max={max_grad_overall:.2f}")
        else:
            result = "❌ FAIL"
            print(f"\n❌ FAILED at batch {failed_batch}/{NUM_BATCHES_TO_TEST}")
            if batch_losses:
                print(f"  Completed batches: {len(batch_losses)}")
                print(f"  Last valid loss: {batch_losses[-1]:.4f}")

        results.append({
            'lr': test_lr,
            'alpha': test_alpha,
            'product': product,
            'description': description,
            'result': result,
            'batches_completed': len(batch_losses),
            'avg_loss': np.mean(batch_losses) if batch_losses else None,
        })

        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'lr': test_lr,
            'alpha': test_alpha,
            'product': product,
            'description': description,
            'result': "❌ EXCEPTION",
            'batches_completed': 0,
            'avg_loss': None,
        })

# Final summary
print("\n" + "="*80)
print("BOUNDARY TEST SUMMARY")
print("="*80)
print(f"{'Description':<45} {'LR':<8} {'Alpha':<7} {'Product':<10} {'Result':<10}")
print("-"*80)
for r in results:
    print(f"{r['description']:<45} {r['lr']:<8.4f} {r['alpha']:<7.2f} {r['product']:<10.6f} {r['result']:<10}")

print("\n" + "="*80)
pass_count = sum(1 for r in results if r['result'] == "✅ PASS")
fail_count = len(results) - pass_count
print(f"OVERALL: {pass_count}/{len(results)} passed, {fail_count}/{len(results)} failed")

if pass_count == len(results):
    print("✅ ALL BOUNDARY TESTS PASSED - Safe to run both modes!")
elif pass_count > 0:
    print("⚠️ SOME TESTS FAILED - Review failed cases before running")
else:
    print("❌ ALL TESTS FAILED - Do not run with current settings")
print("="*80)
