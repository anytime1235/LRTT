"""
Test LRTT with different analog LR values.

Configuration:
- out_scaling_alpha: uses analog_lr (grouped with analog layers)
- classifier_lr: 0.01 (fixed)
- lora_alpha: 1.0 (fixed)
- Test analog_lr: [0.1, 0.01, 0.001]
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lora_training_glue"))

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from aihwkit.optim import AnalogSGD
from aihwkit.optim.context import AnalogContext

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

# Fixed hyperparameters
CLASSIFIER_LR = 0.01
LORA_ALPHA = 1.0
RANK = 8
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
MAX_STEPS = 10

# Test different analog LR values
ANALOG_LR_VALUES = [0.0005, 0.0001]


def run_test(analog_lr: float):
    """Run single test with given analog_lr."""
    print("\n" + "=" * 80)
    print(f"TEST: analog_lr={analog_lr}, classifier_lr={CLASSIFIER_LR}, lora_alpha={LORA_ALPHA}")
    print("=" * 80)

    # 1. Load model and tokenizer
    print("\n[1/6] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )
    print(f"✓ Model loaded: {MODEL_NAME}")

    # 2. Convert to LRTT-LoRA
    print("\n[2/6] Converting to LRTT-LoRA...")
    lrtt_config = create_lrtt_lora_config(
        rank=RANK,
        lora_alpha=LORA_ALPHA,
        output_noise_level=0.0,
        use_floating_point=False,
    )
    model = convert_model_to_lrtt_lora(
        model,
        lrtt_config,
        target_modules=["query", "key", "value"],
    )
    model = model.cuda()
    print("✓ Model converted and moved to GPU")

    # 3. Prepare data
    print("\n[3/6] Loading dataset...")
    dataset = load_dataset("glue", TASK_NAME)
    train_dataset = dataset["train"].select(range(32))

    def preprocess_function(examples):
        result = tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        result["labels"] = examples["label"]
        return result

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
    )
    train_dataset.set_format(type="torch")
    print(f"✓ Dataset loaded: {len(train_dataset)} samples")

    # 4. Setup optimizer with separate LRs
    print("\n[4/6] Setting up optimizer...")
    print(f"  analog_lr (analog_ctx + out_scaling_alpha): {analog_lr}")
    print(f"  classifier_lr (classifier weights/bias): {CLASSIFIER_LR}")

    param_groups = []
    n_analog = 0
    n_out_scale = 0
    n_classifier = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if isinstance(param, AnalogContext):
            # Analog context: use analog_lr
            param_groups.append({
                "params": [param],
                "lr": analog_lr,
            })
            n_analog += 1
        elif "out_scaling_alpha" in name:
            # out_scaling_alpha: also use analog_lr (grouped with analog)
            param_groups.append({
                "params": [param],
                "lr": analog_lr,
            })
            n_out_scale += 1
        else:
            # Digital parameters (classifier): use classifier_lr
            param_groups.append({
                "params": [param],
                "lr": CLASSIFIER_LR,
            })
            n_classifier += 1

    print(f"  Analog params: {n_analog}, OutScale params: {n_out_scale}, Classifier params: {n_classifier}")

    # CRITICAL: Use analog_lr as default (for regroup_param_groups)
    optimizer = AnalogSGD(param_groups, lr=analog_lr)
    print(f"✓ Optimizer created (default lr={analog_lr})")

    # 5. Training setup
    print("\n[5/6] Setting up training...")
    training_args = TrainingArguments(
        output_dir=f"/tmp/lrtt_lr_test_analog{analog_lr}",
        per_device_train_batch_size=8,
        max_steps=MAX_STEPS,
        logging_steps=1,
        save_steps=1000,
        save_total_limit=1,
        report_to="none",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
    )
    print("✓ Trainer configured")

    # 6. Run training
    print(f"\n[6/6] Running training ({MAX_STEPS} steps)...")
    print("-" * 80)

    try:
        trainer.train()
        print("\n" + "-" * 80)
        print("✓ Training completed successfully!")
        success = True
    except Exception as e:
        print("\n" + "-" * 80)
        print(f"✗ Training failed with error:")
        print(f"  {type(e).__name__}: {e}")
        success = False

    return success


def main():
    print("=" * 80)
    print("LRTT LEARNING RATE SWEEP")
    print("=" * 80)
    print(f"\nFixed parameters:")
    print(f"  classifier_lr: {CLASSIFIER_LR}")
    print(f"  lora_alpha: {LORA_ALPHA}")
    print(f"  rank: {RANK}")
    print(f"  max_steps: {MAX_STEPS}")
    print(f"\nTesting analog_lr values: {ANALOG_LR_VALUES}")
    print(f"  out_scaling_alpha: uses analog_lr (grouped with analog layers)")
    print()

    results = {}

    for analog_lr in ANALOG_LR_VALUES:
        success = run_test(analog_lr)
        results[analog_lr] = "SUCCESS" if success else "FAILED"

        # Clean up GPU memory
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for analog_lr, status in results.items():
        print(f"  analog_lr={analog_lr:5.3f}: {status}")
    print("=" * 80)


if __name__ == "__main__":
    main()
