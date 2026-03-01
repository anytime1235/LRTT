#!/data/venvs/aihwkit_gpu/bin/python
# coding=utf-8
"""Test very small LoRA Alpha values."""

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

# Import functions
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
NUM_BATCHES = 20
FIXED_LR = 0.001

# Test very small alpha values
ALPHA_TEST_VALUES = [
    0.001,   # 10^-3
    0.003,   # ~10^-2.5
    0.01,    # 10^-2
    0.03,    # ~10^-1.5
    0.1,     # 10^-1
]

print("="*80)
print("SMALL LoRA ALPHA TEST")
print("="*80)
print("Question: Alpha를 더 작게 하면?")
print("  - 안정성: 더 좋아질 것")
print("  - 학습 능력: 나빠질 수 있음 (LoRA 기여도 감소)")
print("="*80)
print(f"Fixed LR: {FIXED_LR}")
print(f"Testing Alpha values: {ALPHA_TEST_VALUES}")
print(f"Batches: {NUM_BATCHES}, Batch Size: {BATCH_SIZE}")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
print("\nLoading data...")
set_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_loader, eval_features, eval_examples = load_squad_data(tokenizer)
print(f"Loaded: {len(train_loader)} train batches, {len(eval_features)} eval features")

results = []

for alpha in ALPHA_TEST_VALUES:
    product = FIXED_LR * alpha

    print("\n" + "="*80)
    print(f"TEST: Alpha = {alpha}")
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

        # Initial evaluation
        print("Initial evaluation...")
        init_f1, init_em = evaluate_squad(model, eval_features, eval_examples, tokenizer, device)
        print(f"Initial: F1={init_f1:.2f}%, EM={init_em:.2f}%")

        # Train
        print(f"Training {NUM_BATCHES} batches...")
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
                print(f"  Batch {i}: ❌ NaN/Inf loss")
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

            if i % 5 == 0:
                print(f"  Batch {i:2d}: loss={loss.item():10.2f}, grad={max_grad:12.2f}")

        # Final evaluation
        if not failed and len(losses) == NUM_BATCHES:
            print("\nFinal evaluation...")
            final_f1, final_em = evaluate_squad(model, eval_features, eval_examples, tokenizer, device)
            improvement = final_f1 - init_f1

            avg_loss = np.mean(losses)
            max_loss = np.max(losses)
            final_loss = losses[-1]

            print(f"\n✅ SUCCESS")
            print(f"  Loss: avg={avg_loss:.2f}, max={max_loss:.2f}, final={final_loss:.2f}")
            print(f"  Grad: avg={np.mean(grads):.2f}, max={np.max(grads):.2f}")
            print(f"  F1: {init_f1:.2f}% → {final_f1:.2f}% (Δ={improvement:+.2f}%)")
            print(f"  EM: {init_em:.2f}% → {final_em:.2f}%")

            result = "✅ PASS"
        else:
            print(f"\n❌ FAILED at batch {len(losses)}")
            final_f1 = None
            improvement = None
            result = "❌ FAIL"

        results.append({
            'alpha': alpha,
            'product': product,
            'result': result,
            'batches': len(losses),
            'avg_loss': np.mean(losses) if losses else None,
            'final_loss': losses[-1] if losses else None,
            'init_f1': init_f1,
            'final_f1': final_f1,
            'improvement': improvement,
        })

        del model, optimizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n❌ EXCEPTION: {str(e)[:150]}")
        results.append({
            'alpha': alpha,
            'product': product,
            'result': "❌ ERROR",
            'batches': 0,
            'avg_loss': None,
            'final_loss': None,
            'init_f1': None,
            'final_f1': None,
            'improvement': None,
        })
        torch.cuda.empty_cache()

# Summary
print("\n" + "="*80)
print("SUMMARY: Small Alpha Test (LR=0.001)")
print("="*80)
print(f"{'Alpha':<8} {'Product':<10} {'Result':<10} {'Final Loss':<12} {'Init F1':<10} {'Final F1':<10} {'ΔF1':<10}")
print("-"*80)
for r in results:
    loss_str = f"{r['final_loss']:.2f}" if r['final_loss'] else "N/A"
    init_f1_str = f"{r['init_f1']:.2f}%" if r['init_f1'] is not None else "N/A"
    final_f1_str = f"{r['final_f1']:.2f}%" if r['final_f1'] is not None else "N/A"
    imp_str = f"{r['improvement']:+.2f}%" if r['improvement'] is not None else "N/A"

    print(f"{r['alpha']:<8.3f} {r['product']:<10.6f} {r['result']:<10} {loss_str:<12} {init_f1_str:<10} {final_f1_str:<10} {imp_str:<10}")

pass_count = sum(1 for r in results if r['result'] == "✅ PASS")
print(f"\n{pass_count}/{len(results)} tests passed")

# Analysis
if pass_count > 0:
    passed = [r for r in results if r['result'] == "✅ PASS"]

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Safety
    safe_alphas = [r['alpha'] for r in passed]
    safe_products = [r['product'] for r in passed]
    print(f"✅ Safe Alpha range: {min(safe_alphas):.3f} ~ {max(safe_alphas):.3f}")
    print(f"   Safe Product range: {min(safe_products):.6f} ~ {max(safe_products):.6f}")

    # Learning effectiveness
    improvements = [r['improvement'] for r in passed if r['improvement'] is not None]
    if improvements:
        print(f"\n📈 Learning Effectiveness:")
        print(f"   F1 improvements: {min(improvements):+.2f}% ~ {max(improvements):+.2f}%")
        print(f"   Average improvement: {np.mean(improvements):+.2f}%")

        # Find best learning alpha
        best_idx = improvements.index(max(improvements))
        best_alpha = passed[best_idx]['alpha']
        best_improvement = improvements[best_idx]
        print(f"\n🏆 Best learning alpha: {best_alpha:.3f} (ΔF1={best_improvement:+.2f}%)")

    # Recommendation
    print("\n💡 Recommendation:")
    if all(imp > 0 for imp in improvements if imp is not None):
        print("   모든 alpha에서 학습 효과 확인 ✅")
        print("   → Alpha를 더 작게 해도 학습은 가능!")
    else:
        print("   일부 alpha에서 학습 효과 미미 ⚠️")

    # Check if smallest alpha still learns
    smallest_passed = min(passed, key=lambda x: x['alpha'])
    if smallest_passed['improvement'] and smallest_passed['improvement'] > 1.0:
        print(f"   Alpha {smallest_passed['alpha']:.3f}에서도 학습 효과 있음 (ΔF1={smallest_passed['improvement']:+.2f}%)")
        print("   → 더 작은 alpha도 시도 가능!")
    else:
        print(f"   Alpha {smallest_passed['alpha']:.3f}에서 학습 효과 제한적")
        print("   → 이것이 하한선일 가능성")

print("="*80)
