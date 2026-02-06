#!/home/jovyan/work/ml/.venv310/bin/python
# coding=utf-8
"""Compare LRTT vs TTv2 accuracy on MobileBERT SST2 query layer.

Trains LRTT with Bayesian-optimized parameters and compares accuracy with
TTv2 best result from wandb sweep (tikitaka-v2-refined-sweep, trial_37).

TTv2 Reference:
- Accuracy: 75.57%
- target_modules: ['query']
- transfer_every: 82
- transfer_lr: 3.72
- learning_rate: 0.000967
- fast_lr: 0.847
- in_chop_prob: 0.037

LRTT Bayesian Best (for delta_C matching):
- learning_rate: 0.00157
- transfer_lr: 0.0248
- transfer_every: 100 (fixed)
- cumulative_delta_C: 36.95 (TTv2 대비 1.24x)
"""

import os
import sys
import json
import datetime
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import evaluate

# aihwkit imports
from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import SoftBoundsDevice, LinearStepDevice

# LRTT imports
sys.path.insert(0, '/home/jovyan/work/LRTT/src')
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

from config import (
    MODEL_NAME, TASK_NAME, MAX_SEQ_LENGTH, BATCH_SIZE, SEED,
    TARGET_LAYER_NAME, SOFTBOUNDS_CONFIG, LINEARSTEP_AB_CONFIG,
    OUTPUT_DIR,
)
from layer_utils import list_linear_layers, print_trainable_params


# =============================================================================
# Reference Data
# =============================================================================

TTv2_BEST_RESULT = {
    'accuracy': 0.7557,  # 75.57%
    'source': 'wandb tikitaka-v2-refined-sweep trial_37',
    'config': {
        'target_modules': ['query'],
        'transfer_every': 82,
        'transfer_lr': 3.72,
        'learning_rate': 0.000967,
        'fast_lr': 0.847,
        'in_chop_prob': 0.037,
    }
}

LRTT_BAYESIAN_BEST = {
    'learning_rate': 0.00157,
    'transfer_lr': 0.0248,
    'transfer_every': 100,
    'rank': 8,
    'cumulative_delta_C': 36.95,
    'delta_C_ratio_to_ttv2': 1.24,
}

# Alternative LRTT config using TTv2-like parameters
LRTT_TTv2_LIKE = {
    'learning_rate': 0.000967,  # Same as TTv2
    'transfer_lr': 1.0,  # Higher transfer_lr like TTv2 (scaled for LRTT)
    'transfer_every': 100,
    'rank': 8,
}


# =============================================================================
# LRTT Model Creation
# =============================================================================

def create_lrtt_rpu_config(
    transfer_every: int = 100,
    transfer_lr: float = 0.0248,
    rank: int = 8,
    lora_alpha: float = 1.0,
    noise_free: bool = False,
) -> PythonLRTTRPUConfig:
    """Create LRTT RPU config with 6T1C A/B tiles + SoftBounds C tile.

    Uses sixt1c_ab_softbounds configuration matching the Bayesian sweep.

    Args:
        noise_free: If True, set all noise parameters (dtod, std) to 0 for A/B tiles.
    """
    # Noise parameters to zero out when noise_free=True
    if noise_free:
        dw_min_dtod = 0.0
        up_down_dtod = 0.0
        w_max_dtod = 0.0
        w_min_dtod = 0.0
        gamma_up_dtod = 0.0
        gamma_down_dtod = 0.0
        dw_min_std = 0.0
        write_noise_std = 0.0
        lifetime_dtod = 0.0
        reset_dtod = 0.0
    else:
        dw_min_dtod = LINEARSTEP_AB_CONFIG['dw_min_dtod']
        up_down_dtod = LINEARSTEP_AB_CONFIG['up_down_dtod']
        w_max_dtod = LINEARSTEP_AB_CONFIG['w_max_dtod']
        w_min_dtod = LINEARSTEP_AB_CONFIG['w_min_dtod']
        gamma_up_dtod = LINEARSTEP_AB_CONFIG['gamma_up_dtod']
        gamma_down_dtod = LINEARSTEP_AB_CONFIG['gamma_down_dtod']
        dw_min_std = LINEARSTEP_AB_CONFIG['dw_min_std']
        write_noise_std = LINEARSTEP_AB_CONFIG['write_noise_std']
        lifetime_dtod = LINEARSTEP_AB_CONFIG['lifetime_dtod']
        reset_dtod = LINEARSTEP_AB_CONFIG['reset_dtod']

    # A/B tiles: LinearStepDevice (6T1C)
    ab_device = LinearStepDevice(
        dw_min=LINEARSTEP_AB_CONFIG['dw_min'],
        up_down=LINEARSTEP_AB_CONFIG['up_down'],
        w_max=LINEARSTEP_AB_CONFIG['w_max'],
        w_min=LINEARSTEP_AB_CONFIG['w_min'],
        gamma_up=LINEARSTEP_AB_CONFIG['gamma_up'],
        gamma_down=LINEARSTEP_AB_CONFIG['gamma_down'],
        mult_noise=LINEARSTEP_AB_CONFIG['mult_noise'],
        dw_min_dtod=dw_min_dtod,
        up_down_dtod=up_down_dtod,
        w_max_dtod=w_max_dtod,
        w_min_dtod=w_min_dtod,
        gamma_up_dtod=gamma_up_dtod,
        gamma_down_dtod=gamma_down_dtod,
        dw_min_std=dw_min_std,
        write_noise_std=write_noise_std,
        mean_bound_reference=LINEARSTEP_AB_CONFIG['mean_bound_reference'],
        lifetime=LINEARSTEP_AB_CONFIG['lifetime'],
        lifetime_dtod=lifetime_dtod,
        reset=LINEARSTEP_AB_CONFIG['reset'],
        reset_dtod=reset_dtod,
    )

    # C tile: SoftBoundsDevice (noise=0)
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        reinit_gain=0.1,
        reinit_mode="decay",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = transfer_lr
    device_config.forward_inject = False  # 6T1C uses forward_inject=False
    device_config.transfer_method = "onehot"  # row-by-row transfer
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    return PythonLRTTRPUConfig(device=device_config)


def create_lrtt_model(
    learning_rate: float = 0.00157,
    transfer_lr: float = 0.0248,
    transfer_every: int = 100,
    rank: int = 8,
    lora_alpha: float = 1.0,
    target_modules: list = None,
    noise_free: bool = False,
) -> nn.Module:
    """Create LRTT model for MobileBERT with specified target layers.

    Args:
        learning_rate: Not used here, just for config reference
        transfer_lr: LRTT transfer learning rate
        transfer_every: Steps between transfers
        rank: LoRA rank
        lora_alpha: LoRA alpha scaling
        target_modules: List of target module names (e.g., ["query"])
        noise_free: If True, set all noise parameters to 0 for A/B tiles
    """
    if target_modules is None:
        target_modules = ["query"]

    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )

    # Get all linear layers
    all_linear = list_linear_layers(model)

    # Exclude all layers except targets
    exclude = [name for name in all_linear
               if not any(t in name for t in target_modules)]
    exclude.append("classifier")  # Don't convert classifier to analog

    # Create LRTT config
    rpu_config = create_lrtt_rpu_config(
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,
        rank=rank,
        lora_alpha=lora_alpha,
        noise_free=noise_free,
    )

    # Convert target layers to LRTT
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    # Freeze all except target layers and classifier
    for name, param in model.named_parameters():
        is_target = any(t in name for t in target_modules)
        if is_target or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation data.

    Returns:
        (accuracy, average_loss)
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    model.train()
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0

    return accuracy, avg_loss


# =============================================================================
# Training
# =============================================================================

def train_and_evaluate_lrtt(
    learning_rate: float = 0.00157,
    transfer_lr: float = 0.0248,
    transfer_every: int = 100,
    rank: int = 8,
    num_epochs: int = 1,
    warmup_steps: int = 0,
    target_modules: list = None,
    output_dir: Optional[str] = None,
    noise_free: bool = False,
) -> Dict[str, Any]:
    """Train LRTT model and return accuracy metrics."""

    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f'lrtt_accuracy_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(SEED)

    # Load data
    print("\nLoading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
        )

    tokenized = raw_datasets.map(preprocess, batched=True)
    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Create LRTT model
    print("\n" + "=" * 70)
    print("Creating LRTT model with Bayesian-optimized parameters")
    print("=" * 70)
    print(f"  learning_rate: {learning_rate}")
    print(f"  transfer_lr: {transfer_lr}")
    print(f"  transfer_every: {transfer_every}")
    print(f"  rank: {rank}")

    if target_modules is None:
        target_modules = ["query"]
    print(f"  target_modules: {target_modules}")
    print(f"  noise_free: {noise_free}")

    model = create_lrtt_model(
        learning_rate=learning_rate,
        transfer_lr=transfer_lr,
        transfer_every=transfer_every,
        rank=rank,
        target_modules=target_modules,
        noise_free=noise_free,
    )
    model.to(device)
    print_trainable_params(model)

    # Create optimizer
    optimizer = AnalogAdam(model.parameters(), lr=learning_rate)

    # Create scheduler with warmup
    num_training_steps = (len(train_dataset) // BATCH_SIZE) * num_epochs
    if warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        print(f"  warmup_steps: {warmup_steps}")
        print(f"  num_training_steps: {num_training_steps}")
    else:
        scheduler = None

    # Metric for evaluation
    metric = evaluate.load("glue", TASK_NAME)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="wandb",
        seed=SEED,
        remove_unused_columns=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Train
    print("\n" + "=" * 70)
    print(f"Training LRTT for {num_epochs} epoch(s)...")
    print("=" * 70)

    trainer.train()

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final evaluation...")
    print("=" * 70)

    eval_results = trainer.evaluate()
    lrtt_accuracy = eval_results.get('eval_accuracy', 0.0)

    print(f"\nLRTT Final Accuracy: {lrtt_accuracy:.4f} ({lrtt_accuracy*100:.2f}%)")

    return {
        'accuracy': lrtt_accuracy,
        'eval_results': eval_results,
        'config': {
            'learning_rate': learning_rate,
            'transfer_lr': transfer_lr,
            'transfer_every': transfer_every,
            'rank': rank,
            'num_epochs': num_epochs,
            'warmup_steps': warmup_steps,
            'target_modules': target_modules,
            'noise_free': noise_free,
        }
    }


# =============================================================================
# Main Comparison
# =============================================================================

def compare_ttv2_lrtt_accuracy(
    num_epochs: int = 1,
    warmup_steps: int = 0,
    output_dir: Optional[str] = None,
    use_ttv2_like_params: bool = False,
    target_modules: list = None,
    noise_free: bool = False,
) -> Dict[str, Any]:
    """Compare LRTT accuracy with TTv2 reference.

    Args:
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        output_dir: Output directory for results
        use_ttv2_like_params: If True, use TTv2-like parameters instead of Bayesian-optimized
        target_modules: List of target module names (e.g., ["query"])
        noise_free: If True, set all noise parameters to 0 for A/B tiles (sixt1c)
    """

    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_DIR, f'ttv2_lrtt_comparison_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("LRTT vs TTv2 ACCURACY COMPARISON")
    print("=" * 80)

    print("\n[TTv2 Reference (from wandb)]")
    print(f"  Source: {TTv2_BEST_RESULT['source']}")
    print(f"  Accuracy: {TTv2_BEST_RESULT['accuracy']:.4f} ({TTv2_BEST_RESULT['accuracy']*100:.2f}%)")
    print(f"  Config: {TTv2_BEST_RESULT['config']}")

    # Choose LRTT parameters
    if use_ttv2_like_params:
        lrtt_params = LRTT_TTv2_LIKE
        params_name = "TTv2-like"
        print("\n[Using TTv2-like LRTT Parameters]")
    else:
        lrtt_params = LRTT_BAYESIAN_BEST
        params_name = "Bayesian-optimized"
        print("\n[Using Bayesian-optimized LRTT Parameters]")

    print(f"  learning_rate: {lrtt_params['learning_rate']}")
    print(f"  transfer_lr: {lrtt_params['transfer_lr']}")
    print(f"  transfer_every: {lrtt_params['transfer_every']}")
    print(f"  rank: {lrtt_params['rank']}")
    if 'delta_C_ratio_to_ttv2' in lrtt_params:
        print(f"  Expected delta_C ratio to TTv2: {lrtt_params['delta_C_ratio_to_ttv2']}x")

    # Train LRTT
    if noise_free:
        print("\n[NOISE-FREE MODE ENABLED]")
        print("  All noise parameters (dtod, std) set to 0 for A/B tiles")

    lrtt_results = train_and_evaluate_lrtt(
        learning_rate=lrtt_params['learning_rate'],
        transfer_lr=lrtt_params['transfer_lr'],
        transfer_every=lrtt_params['transfer_every'],
        rank=lrtt_params['rank'],
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        target_modules=target_modules,
        output_dir=os.path.join(output_dir, 'lrtt_training'),
        noise_free=noise_free,
    )

    lrtt_accuracy = lrtt_results['accuracy']
    ttv2_accuracy = TTv2_BEST_RESULT['accuracy']

    # Compute comparison metrics
    accuracy_diff = lrtt_accuracy - ttv2_accuracy
    relative_to_ttv2 = lrtt_accuracy / ttv2_accuracy if ttv2_accuracy > 0 else 0

    # Prepare results
    comparison_results = {
        'ttv2': {
            'accuracy': ttv2_accuracy,
            'source': TTv2_BEST_RESULT['source'],
            'config': TTv2_BEST_RESULT['config'],
        },
        'lrtt': {
            'accuracy': lrtt_accuracy,
            'params_type': params_name,
            'config': lrtt_results['config'],
            'eval_results': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                           for k, v in lrtt_results['eval_results'].items()},
        },
        'comparison': {
            'accuracy_difference': accuracy_diff,
            'relative_to_ttv2': relative_to_ttv2,
            'lrtt_better': lrtt_accuracy > ttv2_accuracy,
        },
        'metadata': {
            'num_epochs': num_epochs,
            'warmup_steps': warmup_steps,
            'use_ttv2_like_params': use_ttv2_like_params,
            'noise_free': noise_free,
            'timestamp': datetime.datetime.now().isoformat(),
        }
    }

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'Method':<20} {'Accuracy':>12} {'vs TTv2':>12}")
    print("-" * 50)
    print(f"{'TTv2':<20} {ttv2_accuracy:>12.4f} {'(baseline)':>12}")
    print(f"{'LRTT (' + params_name + ')':<20} {lrtt_accuracy:>12.4f} {accuracy_diff:>+12.4f}")

    print(f"\nLRTT Accuracy Relative to TTv2: {relative_to_ttv2:.4f}x ({relative_to_ttv2*100:.2f}%)")

    # Interpretation
    print("\n" + "-" * 80)
    print("[INTERPRETATION]")
    if accuracy_diff >= 0:
        print(f"  LRTT achieves {accuracy_diff*100:+.2f}%p higher accuracy than TTv2.")
    else:
        print(f"  LRTT achieves {accuracy_diff*100:.2f}%p lower accuracy than TTv2.")

    if abs(accuracy_diff) < 0.02:
        print("  The accuracy difference is within 2%p, suggesting comparable performance.")
    elif abs(accuracy_diff) < 0.05:
        print("  The accuracy difference is within 5%p, suggesting similar performance.")
    else:
        print("  The accuracy difference is significant. Further tuning may be needed.")

    # Save results
    results_path = os.path.join(output_dir, f'lrtt_ttv2_accuracy_comparison_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return comparison_results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare LRTT vs TTv2 accuracy")
    parser.add_argument("--num_epochs", type=int, default=1,
                       help="Number of training epochs (default: 1)")
    parser.add_argument("--warmup_steps", type=int, default=0,
                       help="Number of warmup steps (default: 0)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--ttv2_like", action="store_true",
                       help="Use TTv2-like parameters instead of Bayesian-optimized")
    parser.add_argument("--target_modules", nargs="+", default=None,
                       help="Target modules to convert (e.g., query key value)")
    parser.add_argument("--noise_free", action="store_true",
                       help="Set all noise parameters to 0 for A/B tiles (sixt1c)")

    args = parser.parse_args()

    results = compare_ttv2_lrtt_accuracy(
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        use_ttv2_like_params=args.ttv2_like,
        target_modules=args.target_modules,
        noise_free=args.noise_free,
    )

    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
