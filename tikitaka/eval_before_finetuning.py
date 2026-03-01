#!/usr/bin/env python
"""Evaluate pretrained MobileBERT (after analog QKV conversion) BEFORE fine-tuning."""

import os, sys, json
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/data/LRTT_transformer/src')

from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    default_data_collator, set_seed,
)
from datasets import load_dataset
from torch.utils.data import DataLoader

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import (
    UnitCellRPUConfig, IOParameters, UpdateParameters,
    NoiseManagementType, BoundManagementType,
)
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsReferenceDevice

# --- Same config as sweep_tikitaka_batch256.py ---
MODEL_NAME = "google/mobilebert-uncased"
TARGET_MODULES = ["query", "key", "value", "classifier"]
MAX_SEQ_LENGTH = 128
EVAL_BATCH_SIZE = 64
SEED = 42

TASK_TO_KEYS = {
    "cola": ("sentence", None), "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"), "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"), "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None), "stsb": ("sentence1", "sentence2"),
}
TASK_TO_NUM_LABELS = {
    "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
    "mnli": 3, "qnli": 2, "rte": 2, "stsb": 1,
}
TASK_TO_METRIC = {
    "cola": "matthews_correlation", "sst2": "accuracy", "mrpc": "f1",
    "qqp": "f1", "mnli": "accuracy", "qnli": "accuracy",
    "rte": "accuracy", "stsb": "spearmanr",
}
GLUE_TASKS = ["rte", "mrpc", "stsb", "cola", "sst2", "qnli", "qqp", "mnli"]

# Warm-start params (just needed to build the analog config, doesn't affect eval)
WARM_PARAMS = {
    "transfer_every": 475, "transfer_lr": 0.207, "fast_lr": 3.0,
    "auto_granularity": 169.0, "in_chop_prob": 0.061,
}


def create_tikitaka_v2_config():
    sixt1c_device = LinearStepDevice(
        dw_min=0.001981, gamma_up=-0.1678, gamma_down=0.1410,
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3,
        write_noise_std=0.0, mult_noise=True, mean_bound_reference=True, lifetime=0.0,
    )
    softbounds_device = SoftBoundsReferenceDevice(
        w_max=1.0, w_min=-1.0, dw_min=0.001, dw_min_std=0.0,
        write_noise_std=0.0, diffusion=0.0, dw_min_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0, up_down=0.0, up_down_dtod=0.0,
        lifetime=0.0, lifetime_dtod=0.0, slope_up_dtod=0.0, slope_down_dtod=0.0,
    )
    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[sixt1c_device, softbounds_device],
            transfer_every=WARM_PARAMS["transfer_every"],
            units_in_mbatch=False, n_reads_per_transfer=1, transfer_columns=True,
            gamma=0.0, transfer_lr=WARM_PARAMS["transfer_lr"],
            fast_lr=WARM_PARAMS["fast_lr"], scale_transfer_lr=True,
            auto_scale=True, auto_granularity=WARM_PARAMS["auto_granularity"],
            buffer_granularity=1.0, auto_momentum=0.99,
            in_chop_prob=WARM_PARAMS["in_chop_prob"], in_chop_random=True,
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False,
            ),
        )
    )
    rpu_config.forward = IOParameters(
        is_perfect=False, inp_noise=0.0, out_noise=0.0, out_noise_std=0.0,
        w_noise=0.0, noise_management=NoiseManagementType.ABS_MAX,
        bound_management=BoundManagementType.ITERATIVE,
    )
    rpu_config.backward = IOParameters(
        is_perfect=False, inp_noise=0.0, out_noise=0.0, out_noise_std=0.0,
        w_noise=0.0, noise_management=NoiseManagementType.ABS_MAX,
        bound_management=BoundManagementType.ITERATIVE,
    )
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


def create_model(task_name, device):
    num_labels = TASK_TO_NUM_LABELS[task_name]
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    classifier_trainable = any("classifier" in t for t in TARGET_MODULES)
    if hasattr(model, 'classifier') and not classifier_trainable:
        torch.manual_seed(SEED)
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)
    elif classifier_trainable:
        print(f"  [INFO] Classifier TRAINABLE — using pretrained init")

    all_linear = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("classifier")

    rpu_config = create_tikitaka_v2_config()
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        is_head = "classifier" in name
        if is_head:
            param.requires_grad = is_target
        elif "bias" in name:
            param.requires_grad = False
        else:
            param.requires_grad = is_target

    return model.to(device)


def load_eval_data(task_name, tokenizer):
    raw = load_dataset("nyu-mll/glue", task_name)
    s1_key, s2_key = TASK_TO_KEYS[task_name]

    def preprocess(examples):
        if s2_key is None:
            return tokenizer(examples[s1_key], padding="max_length",
                             max_length=MAX_SEQ_LENGTH, truncation=True)
        return tokenizer(examples[s1_key], examples[s2_key],
                         padding="max_length", max_length=MAX_SEQ_LENGTH, truncation=True)

    tokenized = raw.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    eval_key = "validation_matched" if task_name == "mnli" else "validation"
    return DataLoader(tokenized[eval_key], batch_size=EVAL_BATCH_SIZE,
                      shuffle=False, collate_fn=default_data_collator)


def evaluate(model, eval_loader, task_name, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    is_regression = task_name == "stsb"
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            if is_regression:
                labels = labels.float()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze() if is_regression else outputs.logits
            loss = criterion(logits, labels)

            if is_regression:
                all_preds.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item() * labels.size(0)

    n = len(all_labels)
    if is_regression:
        from scipy.stats import spearmanr
        metric = spearmanr(all_preds, all_labels)[0]
    elif task_name in ["mrpc", "qqp"]:
        from sklearn.metrics import f1_score
        metric = f1_score(all_labels, all_preds)
    elif task_name == "cola":
        from sklearn.metrics import matthews_corrcoef
        metric = matthews_corrcoef(all_labels, all_preds)
    else:
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        metric = correct / n if n > 0 else 0.0

    return metric, total_loss / n if n > 0 else 0.0


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    output_dir = "/data/results/before_finetuning"
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    print("=" * 60)
    print("Before Fine-tuning Evaluation (Analog QKV Mapped)")
    print(f"Model: {MODEL_NAME}")
    print(f"Target modules: {TARGET_MODULES}")
    print("=" * 60)

    for task in GLUE_TASKS:
        print(f"\n--- {task.upper()} ---")
        eval_loader = load_eval_data(task, tokenizer)
        set_seed(SEED)  # reset seed before model creation (same as run_trial)
        model = create_model(task, device)
        metric_val, eval_loss = evaluate(model, eval_loader, task, device)
        metric_name = TASK_TO_METRIC[task]

        results[task] = {
            "metric_name": metric_name,
            "metric_value": float(metric_val) if np.isfinite(metric_val) else str(metric_val),
            "eval_loss": float(eval_loss),
            "num_labels": TASK_TO_NUM_LABELS[task],
        }
        print(f"  {metric_name}: {metric_val:.4f}, loss: {eval_loss:.4f}")

        del model
        torch.cuda.empty_cache()

    out_file = os.path.join(output_dir, "before_finetuning_results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for task, r in results.items():
        print(f"  {task:6s} | {r['metric_name']:>22s} = {r['metric_value']}")
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
