"""
Optuna hyperparameter sweep for LRTT-LoRA on SST-2.

Search space:
- classifier_lr: 0.01 (FIXED)
- analog_lr: [1e-4, 1e-2] (log-uniform)
- lora_alpha: [0.01, 100] (log-uniform)
- out_scaling_alpha: uses analog_lr (grouped with analog layers)

Model: MobileBERT
Task: SST-2 (GLUE)
Trials: 30
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "lora_training_glue"))

import torch
import numpy as np
import optuna
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)

from aihwkit.optim import AnalogSGD
from aihwkit.optim.context import AnalogContext

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
RANK = 8
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3
MAX_STEPS = -1  # Use epochs instead
WARMUP_RATIO = 0.1
SEED = 42

# Fixed hyperparameters
CLASSIFIER_LR = 0.01  # FIXED

# Search space
N_TRIALS = 30
ANALOG_LR_MIN = 1e-4
ANALOG_LR_MAX = 1e-2
LORA_ALPHA_MIN = 0.01
LORA_ALPHA_MAX = 100.0

# Output
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(__file__).parent / f"optuna_sst2_lrtt_{TIMESTAMP}"
LOG_FILE = OUTPUT_DIR / "optuna_sweep.log"


def setup_logging():
    """Setup logging to both file and console."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def preprocess_dataset(tokenizer, max_length=128):
    """Load and preprocess SST-2 dataset."""
    dataset = load_dataset("glue", TASK_NAME)

    def preprocess_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    eval_dataset = dataset["validation"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    return train_dataset, eval_dataset


def compute_metrics(eval_pred):
    """Compute accuracy metric."""
    metric = evaluate.load("glue", TASK_NAME)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def objective(trial):
    """Optuna objective function."""
    # Sample hyperparameters
    analog_lr = trial.suggest_float("analog_lr", ANALOG_LR_MIN, ANALOG_LR_MAX, log=True)
    lora_alpha = trial.suggest_float("lora_alpha", LORA_ALPHA_MIN, LORA_ALPHA_MAX, log=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"Trial {trial.number}")
    logger.info(f"{'='*80}")
    logger.info(f"Hyperparameters:")
    logger.info(f"  analog_lr: {analog_lr:.6f}")
    logger.info(f"  classifier_lr: {CLASSIFIER_LR} (fixed)")
    logger.info(f"  lora_alpha: {lora_alpha:.6f}")
    logger.info(f"  rank: {RANK}")

    # Set seed
    set_seed(SEED)

    try:
        # 1. Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # 2. Load and preprocess data
        train_dataset, eval_dataset = preprocess_dataset(tokenizer, MAX_SEQ_LENGTH)

        # 3. Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )

        # 4. Convert to LRTT-LoRA
        lrtt_config = create_lrtt_lora_config(
            rank=RANK,
            lora_alpha=lora_alpha,
            output_noise_level=0.0,
            use_floating_point=False,
        )

        model = convert_model_to_lrtt_lora(
            model,
            lrtt_config,
            target_modules=["query", "key", "value"],
        )

        # 5. Setup optimizer with separate LRs
        param_groups = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if isinstance(param, AnalogContext):
                # Analog context: use analog_lr
                param_groups.append({
                    "params": [param],
                    "lr": analog_lr,
                })
            elif "out_scaling_alpha" in name:
                # out_scaling_alpha: also use analog_lr (grouped with analog)
                param_groups.append({
                    "params": [param],
                    "lr": analog_lr,
                })
            else:
                # Digital parameters (classifier): use CLASSIFIER_LR
                param_groups.append({
                    "params": [param],
                    "lr": CLASSIFIER_LR,
                })

        # Create optimizer with analog_lr as default
        optimizer = AnalogSGD(param_groups, lr=analog_lr)

        # 6. Training arguments
        trial_output_dir = OUTPUT_DIR / f"trial_{trial.number}"

        training_args = TrainingArguments(
            output_dir=str(trial_output_dir),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=analog_lr,  # For logging (actual LR is set in optimizer)
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS,
            max_steps=MAX_STEPS,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=0.0,
            logging_dir=str(trial_output_dir / "logs"),
            logging_steps=50,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="none",
            seed=SEED,
            max_grad_norm=1.0,
            disable_tqdm=False,
        )

        # 7. Create Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None),
        )

        # 8. Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # 9. Evaluate
        eval_result = trainer.evaluate()

        accuracy = eval_result["eval_accuracy"]
        loss = eval_result["eval_loss"]

        logger.info(f"Trial {trial.number} completed:")
        logger.info(f"  eval_accuracy: {accuracy:.4f}")
        logger.info(f"  eval_loss: {loss:.4f}")

        # Clean up
        del model
        del trainer
        torch.cuda.empty_cache()

        return accuracy

    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Clean up on error
        torch.cuda.empty_cache()

        # Return a very low score so Optuna knows this trial failed
        return 0.0


def main():
    """Main function to run Optuna sweep."""
    logger.info("="*80)
    logger.info("LRTT-LORA OPTUNA SWEEP - SST-2")
    logger.info("="*80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Task: {TASK_NAME}")
    logger.info(f"  Rank: {RANK}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Max seq length: {MAX_SEQ_LENGTH}")
    logger.info(f"  Seed: {SEED}")
    logger.info(f"\nSearch space:")
    logger.info(f"  classifier_lr: {CLASSIFIER_LR} (FIXED)")
    logger.info(f"  analog_lr: [{ANALOG_LR_MIN}, {ANALOG_LR_MAX}] (log-uniform)")
    logger.info(f"  lora_alpha: [{LORA_ALPHA_MIN}, {LORA_ALPHA_MAX}] (log-uniform)")
    logger.info(f"  out_scaling_alpha: uses analog_lr")
    logger.info(f"\nOptuna:")
    logger.info(f"  Trials: {N_TRIALS}")
    logger.info(f"  Output: {OUTPUT_DIR}")
    logger.info(f"  Log: {LOG_FILE}")
    logger.info("")

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"lrtt_lora_sst2_{TIMESTAMP}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100),
    )

    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)

    # Print results
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETED")
    logger.info("="*80)
    logger.info(f"\nBest trial:")
    logger.info(f"  Number: {study.best_trial.number}")
    logger.info(f"  Accuracy: {study.best_trial.value:.4f}")
    logger.info(f"  Params:")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save best params
    best_params_file = OUTPUT_DIR / "best_params.txt"
    with open(best_params_file, "w") as f:
        f.write(f"Best Trial: {study.best_trial.number}\n")
        f.write(f"Best Accuracy: {study.best_trial.value:.4f}\n")
        f.write(f"\nBest Parameters:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")

    logger.info(f"\nBest parameters saved to: {best_params_file}")
    logger.info(f"Full log saved to: {LOG_FILE}")

    # Print top 5 trials
    logger.info("\nTop 5 trials:")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value else 0.0, reverse=True)[:5]
    for i, trial in enumerate(top_trials, 1):
        logger.info(f"  {i}. Trial {trial.number}: accuracy={trial.value:.4f}")
        logger.info(f"     analog_lr={trial.params['analog_lr']:.6f}, lora_alpha={trial.params['lora_alpha']:.6f}")


if __name__ == "__main__":
    main()
