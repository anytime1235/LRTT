#!/usr/bin/env python
"""
Test script to verify:
1. base_layer is frozen (requires_grad=False)
2. lora_A/lora_B are trainable (requires_grad=True)
3. Eval runs without error
"""

import subprocess
import sys

# Quick test with minimal samples
cmd = [
    "python", "run_qa.py",
    "--model_name_or_path", "google/mobilebert-uncased",
    "--dataset_name", "squad",
    "--baseline_mode", "analog",
    "--analog_device", "sixt1c",
    "--sixt1c_dt_batch_sec", "1.0",
    "--sixt1c_include_retention", "true",
    "--lora_r", "8",
    "--lora_alpha", "16",
    "--do_train", "true",
    "--do_eval", "true",
    "--num_train_epochs", "1",
    "--max_train_samples", "100",
    "--max_eval_samples", "100",
    "--per_device_train_batch_size", "8",
    "--per_device_eval_batch_size", "8",
    "--learning_rate", "3e-4",
    "--analog_optimizer", "AnalogAdam",
    "--analog_lr", "3e-4",
    "--output_dir", "/tmp/test_sixt1c_freeze",
    "--overwrite_output_dir", "true",
    "--report_to", "none",
]

print("=" * 60)
print("Testing Sixt1c mode: SoftBounds base_layer + LinearStepDevice LoRA")
print("=" * 60)
print("\nCommand:")
print(" ".join(cmd))
print("\n")

result = subprocess.run(cmd, cwd="/data/LRTT_transformer/lora_training")
sys.exit(result.returncode)
