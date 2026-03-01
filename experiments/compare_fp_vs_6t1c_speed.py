#!/usr/bin/env python
"""
Compare training speed: FP-LoRA vs 6T1C-LoRA
"""
import sys
import torch
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from aihwkit.optim import AnalogSGD
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

MODEL_NAME = "google/mobilebert-uncased"
TASK = "sst2"
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 128
NUM_STEPS = 50  # Test first 50 steps only

device = torch.device("cuda")

print("=" * 80)
print("FP-LORA vs 6T1C-LORA SPEED COMPARISON")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Task: {TASK}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Test steps: {NUM_STEPS}")
print("=" * 80)

# Load dataset once
print("\n[Preparing] Loading dataset...")
dataset = load_dataset("glue", TASK, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=MAX_SEQ_LENGTH)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
train_dataset = tokenized_datasets["train"]

# Create dataloader with proper padding
from torch.utils.data import DataLoader
data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=data_collator,
    shuffle=True
)

print(f"✓ Dataset loaded: {len(train_dataset)} samples")

def test_mode(mode_name, use_floating_point):
    """Test training speed for given mode"""
    print(f"\n{'=' * 80}")
    print(f"TESTING: {mode_name}")
    print(f"{'=' * 80}")

    # Load fresh model
    print(f"[1] Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Create config
    print(f"[2] Creating LRTT config...")
    lrtt_config = create_lrtt_lora_config(
        rank=8,
        lora_alpha=1.0,
        output_noise_level=0.0,
        use_floating_point=use_floating_point,
    )

    # Convert to LRTT
    print(f"[3] Converting to LRTT-LoRA...")
    model = convert_model_to_lrtt_lora(model, lrtt_config, ["query", "key", "value"])
    model.to(device)

    # Create optimizer
    print(f"[4] Creating optimizer...")
    optimizer = AnalogSGD(model.parameters(), lr=5e-4)
    optimizer.regroup_param_groups(model)

    # Training loop
    print(f"[5] Running {NUM_STEPS} training steps...")
    model.train()

    step_times = []
    total_start = time.time()

    for step, batch in enumerate(train_dataloader):
        if step >= NUM_STEPS:
            break

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        step_start = time.time()

        # Training step
        optimizer.zero_grad()

        # Add labels
        labels = batch.pop("label")
        outputs = model(**batch, labels=labels)
        loss = outputs.loss

        if torch.isnan(loss):
            print(f"  Step {step}: NaN loss! Stopping.")
            return None, None

        loss.backward()
        optimizer.step()

        step_time = time.time() - step_start
        step_times.append(step_time)

        # Print progress every 10 steps
        if step % 10 == 0:
            avg_time = sum(step_times[-10:]) / len(step_times[-10:])
            print(f"  Step {step}/{NUM_STEPS}: loss={loss.item():.4f}, step_time={step_time:.3f}s, avg_10={avg_time:.3f}s/it")

    total_time = time.time() - total_start
    avg_step_time = sum(step_times) / len(step_times)

    print(f"\n{'=' * 80}")
    print(f"{mode_name} RESULTS")
    print(f"{'=' * 80}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average step time: {avg_step_time:.3f}s/it")
    print(f"Throughput: {1/avg_step_time:.2f} it/s")
    print(f"Min step time: {min(step_times):.3f}s")
    print(f"Max step time: {max(step_times):.3f}s")
    print(f"{'=' * 80}")

    # Cleanup
    del model
    del optimizer
    torch.cuda.empty_cache()

    return avg_step_time, total_time

# Test FP-LoRA (digital mode)
fp_avg_time, fp_total_time = test_mode("FP-LORA (Digital/Floating-Point)", use_floating_point=True)

# Test 6T1C-LoRA (analog mode)
t1c_avg_time, t1c_total_time = test_mode("6T1C-LORA (Analog/6T1C)", use_floating_point=False)

# Summary
print(f"\n{'=' * 80}")
print("SPEED COMPARISON SUMMARY")
print(f"{'=' * 80}")

if fp_avg_time and t1c_avg_time:
    speedup = t1c_avg_time / fp_avg_time

    print(f"\nFP-LoRA (Digital):")
    print(f"  Average: {fp_avg_time:.3f}s/it")
    print(f"  Throughput: {1/fp_avg_time:.2f} it/s")
    print(f"  Total time: {fp_total_time:.2f}s")

    print(f"\n6T1C-LoRA (Analog):")
    print(f"  Average: {t1c_avg_time:.3f}s/it")
    print(f"  Throughput: {1/t1c_avg_time:.2f} it/s")
    print(f"  Total time: {t1c_total_time:.2f}s")

    print(f"\nSpeed ratio:")
    print(f"  6T1C / FP-LoRA: {speedup:.2f}x")

    if speedup > 1.5:
        print(f"\n⚠️  6T1C-LoRA is {speedup:.2f}x SLOWER than FP-LoRA!")
        print("  This suggests analog tile operations are the bottleneck.")
    elif speedup > 1.1:
        print(f"\n  6T1C-LoRA is {speedup:.2f}x slower (acceptable overhead)")
    else:
        print(f"\n✓ Performance is similar (within {speedup:.2f}x)")
else:
    print("One or both tests failed!")

print(f"{'=' * 80}")
