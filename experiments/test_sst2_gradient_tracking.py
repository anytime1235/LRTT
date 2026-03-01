#!/usr/bin/env python
# coding=utf-8
"""
SST-2 1 epoch test with detailed Q/K/V gradient and weight update tracking.

This script:
1. Runs SST-2 training for 1 epoch
2. Tracks gradients flowing to Q/K/V layers at each step
3. Monitors gradient clipping effects
4. Verifies A/B tile weight updates
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "offline"

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    TrainerCallback,
)
import evaluate

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.optim import AnalogSGD

print("=" * 80)
print("SST-2 1 EPOCH: Q/K/V GRADIENT & WEIGHT UPDATE TRACKING")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# =============================================================================
# Configuration
# =============================================================================
print("[1/6] Configuration...")
print("-" * 80)

TASK = "sst2"
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
LORA_ALPHA = 1.0
RANK = 8

print(f"Task: {TASK}")
print(f"Model: {MODEL_NAME}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"LoRA alpha: {LORA_ALPHA}")
print(f"Rank: {RANK}")
print()

set_seed(42)

# =============================================================================
# Load dataset
# =============================================================================
print("[2/6] Loading SST-2 dataset...")
print("-" * 80)

dataset = load_dataset("glue", TASK)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

dataset = dataset.map(preprocess, batched=True)
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")
print()

# =============================================================================
# Create model
# =============================================================================
print("[3/6] Creating LRTT-LoRA model...")
print("-" * 80)

config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

# Convert to LRTT-LoRA
lrtt_config = create_lrtt_lora_config(
    rank=RANK,
    lora_alpha=LORA_ALPHA,
    use_floating_point=False  # 6T1C mode
)

model = convert_model_to_lrtt_lora(
    model,
    lrtt_config,
    target_modules=["query", "key", "value"]
)

print("✓ Model converted to LRTT-LoRA")
print()

# =============================================================================
# Helper functions for tracking
# =============================================================================

def get_qkv_tiles(model):
    """Find all Q/K/V LRTT tiles."""
    tiles = {}
    for name, module in model.named_modules():
        if any(qkv in name for qkv in ['query', 'key', 'value']):
            if hasattr(module, 'analog_module'):
                tiles[name] = module.analog_module
    return tiles

def get_tile_weights(tile):
    """Get A, B, C weights from LRTT tile."""
    w_a, _ = tile.tile_a.get_weights()
    w_b, _ = tile.tile_b.get_weights()
    w_c, _ = tile.tile_c.get_weights()
    return {
        'A': w_a.clone().detach().cpu(),
        'B': w_b.clone().detach().cpu(),
        'C': w_c.clone().detach().cpu(),
    }

def compute_weight_stats(weights):
    """Compute statistics of weights."""
    return {
        'norm': weights.norm().item(),
        'max': weights.abs().max().item(),
        'mean': weights.abs().mean().item(),
    }

# Store initial weights
qkv_tiles = get_qkv_tiles(model)
initial_weights = {}

print("[4/6] Storing initial Q/K/V weights...")
print("-" * 80)

for name, tile in qkv_tiles.items():
    weights = get_tile_weights(tile)
    initial_weights[name] = weights

    layer_type = name.split('.')[-1] if '.' in name else name
    print(f"{layer_type}:")
    print(f"  A: {compute_weight_stats(weights['A'])}")
    print(f"  B: {compute_weight_stats(weights['B'])}")
    print(f"  C: {compute_weight_stats(weights['C'])}")

print()

# =============================================================================
# Gradient tracking callback
# =============================================================================

class GradientTracker(TrainerCallback):
    """Track gradients and weight updates during training."""

    def __init__(self, model, qkv_tiles):
        self.model = model
        self.qkv_tiles = qkv_tiles
        self.gradient_logs = []
        self.weight_logs = []
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        """Log gradients and weights after each step."""
        self.step_count += 1

        # Sample only every 10 steps to avoid overhead
        if self.step_count % 10 != 0 and self.step_count != 1:
            return

        step_log = {
            'step': state.global_step,
            'loss': state.log_history[-1].get('loss', 0) if state.log_history else 0,
            'grad_norm': state.log_history[-1].get('grad_norm', 0) if state.log_history else 0,
        }

        # Check gradients and analog contexts
        analog_grad_norms = {}
        for name, param in self.model.named_parameters():
            if any(qkv in name for qkv in ['query', 'key', 'value']):
                if 'analog_ctx' in name:
                    # Check if AnalogContext has gradient info
                    from aihwkit.optim.context import AnalogContext
                    if isinstance(param, AnalogContext) and param.has_gradient():
                        if param.analog_grad_output:
                            d_tensors = param.analog_grad_output
                            d_concat = torch.cat([d.flatten() for d in d_tensors])
                            analog_grad_norms[name] = d_concat.norm().item()

                elif param.grad is not None:
                    # Regular parameter gradients
                    analog_grad_norms[name] = param.grad.norm().item()

        step_log['analog_grads'] = analog_grad_norms
        self.gradient_logs.append(step_log)

        # Log weights at certain steps
        if self.step_count in [1, 10, 50, 100]:
            weight_log = {'step': state.global_step}
            for name, tile in self.qkv_tiles.items():
                weights = get_tile_weights(tile)
                layer_type = name.split('.')[-1] if '.' in name else name
                weight_log[layer_type] = {
                    'A': compute_weight_stats(weights['A']),
                    'B': compute_weight_stats(weights['B']),
                    'C': compute_weight_stats(weights['C']),
                }
            self.weight_logs.append(weight_log)

# =============================================================================
# Training
# =============================================================================
print("[5/6] Starting training...")
print("-" * 80)

# Optimizer
optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE)

# Training arguments
training_args = TrainingArguments(
    output_dir="/tmp/sst2_gradient_tracking",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    learning_rate=LEARNING_RATE,
    seed=42,
    max_grad_norm=1.0,  # Gradient clipping
    report_to="none",
    disable_tqdm=False,
)

# Metric
metric = evaluate.load("glue", TASK)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Create callback
gradient_tracker = GradientTracker(model, qkv_tiles)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
    callbacks=[gradient_tracker],
)

# Train
print("\nStarting training...")
train_result = trainer.train()

print("\n✓ Training completed")
print()

# Evaluate
print("Evaluating...")
eval_result = trainer.evaluate()
print(f"Eval accuracy: {eval_result['eval_accuracy']:.4f}")
print()

# =============================================================================
# Analysis
# =============================================================================
print("[6/6] Analyzing gradient flow and weight updates...")
print("=" * 80)
print()

# Print gradient logs
print("GRADIENT TRACKING (sampled steps):")
print("-" * 80)
for log in gradient_tracker.gradient_logs[:5]:  # First 5 samples
    print(f"\nStep {log['step']}:")
    print(f"  Loss: {log['loss']:.6f}")
    print(f"  Grad norm (total): {log['grad_norm']:.6f}")

    if log['analog_grads']:
        print(f"  Analog context gradients:")
        for name, grad_norm in list(log['analog_grads'].items())[:3]:  # First 3
            layer = name.split('.')[0] if '.' in name else name
            print(f"    {layer}: {grad_norm:.8f}")
    else:
        print(f"  Analog context gradients: None captured")

if len(gradient_tracker.gradient_logs) > 5:
    print(f"\n... ({len(gradient_tracker.gradient_logs) - 5} more steps logged)")

print()
print("WEIGHT UPDATE TRACKING:")
print("-" * 80)

# Get final weights
final_weights = {}
for name, tile in qkv_tiles.items():
    final_weights[name] = get_tile_weights(tile)

# Compare initial vs final
print("\nInitial vs Final weights:")
for name in list(qkv_tiles.keys())[:3]:  # Show first 3 layers
    layer_type = name.split('.')[-1] if '.' in name else name
    print(f"\n{layer_type}:")

    init = initial_weights[name]
    final = final_weights[name]

    for tile_name in ['A', 'B', 'C']:
        diff = (final[tile_name] - init[tile_name])
        change_norm = diff.norm().item()
        change_max = diff.abs().max().item()

        print(f"  {tile_name} tile:")
        print(f"    Initial norm: {init[tile_name].norm().item():.6f}")
        print(f"    Final norm:   {final[tile_name].norm().item():.6f}")
        print(f"    Change norm:  {change_norm:.6f}")
        print(f"    Max change:   {change_max:.8f}")

        if tile_name in ['A', 'B']:
            if change_max > 1e-6:
                print(f"    Status: ✅ UPDATED")
            else:
                print(f"    Status: ❌ NO CHANGE")
        else:  # C
            if change_max < 1e-6:
                print(f"    Status: ✅ FROZEN")
            else:
                print(f"    Status: ⚠️  CHANGED")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

# Check if A and B updated across all layers
all_a_updated = all(
    (final_weights[name]['A'] - initial_weights[name]['A']).abs().max().item() > 1e-6
    for name in qkv_tiles.keys()
)

all_b_updated = all(
    (final_weights[name]['B'] - initial_weights[name]['B']).abs().max().item() > 1e-6
    for name in qkv_tiles.keys()
)

all_c_frozen = all(
    (final_weights[name]['C'] - initial_weights[name]['C']).abs().max().item() < 1e-6
    for name in qkv_tiles.keys()
)

print(f"Total training steps: {gradient_tracker.step_count}")
print(f"Gradient samples logged: {len(gradient_tracker.gradient_logs)}")
print(f"Final eval accuracy: {eval_result['eval_accuracy']:.4f}")
print()

print("Q/K/V A tiles:")
if all_a_updated:
    print("  ✅ All A tiles updated successfully")
else:
    print("  ❌ Some A tiles did not update")

print("\nQ/K/V B tiles:")
if all_b_updated:
    print("  ✅ All B tiles updated successfully")
else:
    print("  ❌ Some B tiles did not update")

print("\nQ/K/V C tiles:")
if all_c_frozen:
    print("  ✅ All C tiles remained frozen")
else:
    print("  ⚠️  Some C tiles changed (should be frozen)")

print()
if all_a_updated and all_b_updated and all_c_frozen:
    print("🎉 LRTT-LoRA training working correctly on SST-2!")
else:
    print("⚠️  Issues detected in LRTT-LoRA training")

print()
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
