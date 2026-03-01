#!/usr/bin/env python
"""
Test if forward_inject=True causes gradient computation issues
"""
import sys
import torch
import time
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from transformers import AutoModelForSequenceClassification
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.optim import AnalogSGD

MODEL_NAME = "google/mobilebert-uncased"
BATCH_SIZE = 32
SEQ_LENGTH = 128

print("=" * 80)
print("TESTING FORWARD_INJECT GRADIENT ISSUE")
print("=" * 80)

device = torch.device("cuda")

# Create dummy batch
input_ids = torch.randint(0, 30522, (BATCH_SIZE, SEQ_LENGTH)).to(device)
attention_mask = torch.ones(BATCH_SIZE, SEQ_LENGTH).to(device)
labels = torch.randint(0, 2, (BATCH_SIZE,)).to(device)

def test_training_step(forward_inject: bool, model_name: str):
    """Test one training step and measure GPU usage"""
    print(f"\n{'=' * 80}")
    print(f"Testing with forward_inject={forward_inject}")
    print('=' * 80)

    # Load model
    print("[1] Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Create LRTT config
    print(f"[2] Creating LRTT config (forward_inject={forward_inject})...")
    lrtt_config = create_lrtt_lora_config(
        rank=8,
        lora_alpha=1.0,
        output_noise_level=0.0,
        use_floating_point=False,
    )

    # Manually override forward_inject for test
    lrtt_config.device.forward_inject = forward_inject
    print(f"    Config forward_inject: {lrtt_config.device.forward_inject}")

    # Convert to LRTT
    print("[3] Converting to LRTT-LoRA...")
    model = convert_model_to_lrtt_lora(model, lrtt_config, ["query", "key", "value"])
    model.to(device)

    # Create optimizer
    print("[4] Creating optimizer...")
    optimizer = AnalogSGD(model.parameters(), lr=5e-4)
    optimizer.regroup_param_groups(model)

    # Training loop
    print("[5] Running 10 training steps...")
    model.train()

    step_times = []
    for step in range(10):
        start = time.time()

        optimizer.zero_grad()

        # Forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Check for NaN
        if torch.isnan(loss):
            print(f"  Step {step}: loss=NaN ❌")
            return False, None

        # Backward
        loss.backward()

        # Check gradients
        has_grad = False
        max_grad = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None and 'query' in name:
                has_grad = True
                max_grad = max(max_grad, param.grad.abs().max().item())

        if not has_grad:
            print(f"  Step {step}: No gradients! ❌")
            return False, None

        # Optimizer step
        optimizer.step()

        elapsed = time.time() - start
        step_times.append(elapsed)

        if step % 2 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, max_grad={max_grad:.6f}, time={elapsed:.3f}s")

    avg_time = sum(step_times) / len(step_times)
    print(f"\n✓ Training successful!")
    print(f"  Average step time: {avg_time:.3f}s")
    print(f"  Final loss: {loss.item():.4f}")

    return True, avg_time

# Test both modes
print("\n" + "=" * 80)
print("TEST 1: forward_inject=FALSE (disabled)")
print("=" * 80)
success_false, time_false = test_training_step(forward_inject=False, model_name="forward_inject_false")

print("\n" + "=" * 80)
print("TEST 2: forward_inject=TRUE (enabled)")
print("=" * 80)
success_true, time_true = test_training_step(forward_inject=True, model_name="forward_inject_true")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"forward_inject=False: {'✓ SUCCESS' if success_false else '✗ FAILED'} (avg time: {time_false:.3f}s)" if success_false else "forward_inject=False: ✗ FAILED")
print(f"forward_inject=True:  {'✓ SUCCESS' if success_true else '✗ FAILED'} (avg time: {time_true:.3f}s)" if success_true else "forward_inject=True:  ✗ FAILED")

if success_false and success_true:
    speedup = time_false / time_true if time_true > 0 else 0
    print(f"\nSpeed comparison: {speedup:.2f}x")
    if speedup < 0.8:
        print("⚠️  forward_inject=True is significantly slower!")
elif success_false and not success_true:
    print("\n❌ CRITICAL: forward_inject=True causes training failure!")
    print("   Recommendation: Use forward_inject=False")
elif not success_false and success_true:
    print("\n⚠️  forward_inject=False failed but True works?")
else:
    print("\n❌ Both modes failed!")

print("=" * 80)
