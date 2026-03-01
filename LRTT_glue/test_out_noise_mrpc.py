"""Compare MRPC F1 with out_noise=0.06 (default) vs out_noise=0.0.

Sweep across alpha values with fixed lr=0.002, QKV target.
"""
import os
import sys
sys.path.insert(0, '/data/LRTT_transformer/src')

import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    default_data_collator, set_seed,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

MODEL_NAME = "google/mobilebert-uncased"
RANK = 8
NUM_EPOCHS = 3
BATCH_SIZE = 256
MAX_SEQ_LENGTH = 128
SEED = 42
WARMUP_STEPS = 0
MIN_LR_RATIO = 0.05

LR = 0.002
TARGET_MODULES = ["query", "key", "value"]


def create_config(lora_alpha, out_noise_val):
    TAU_SEC = 46505.0
    delta = 1 - math.exp(-1.0 / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3,
        write_noise_std=0.0, mean_bound_reference=True,
        lifetime=lifetime, lifetime_dtod=0.0, reset=0.0, reset_dtod=0.0,
    )
    c_device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=True,
    )
    device_config = PythonLRTTDevice(
        rank=RANK, transfer_every=1000000,
        lora_alpha=lora_alpha, reinit_gain=0.1,
        reinit_mode="hybrid", decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.001
    device_config.units_in_mbatch = True
    device_config.forward_inject = True
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.backward.out_noise = out_noise_val
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


def create_model(lora_alpha, out_noise_val, device_hw):
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    exclude = [n for n in all_linear if not any(t in n for t in TARGET_MODULES)]
    exclude.append("classifier")

    rpu_config = create_config(lora_alpha, out_noise_val)
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES) and "bias" not in name
        param.requires_grad = is_target or "classifier" in name

    return model.to(device_hw)


def evaluate(model, eval_loader, device_hw):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in eval_loader:
            ids = batch['input_ids'].to(device_hw)
            mask = batch['attention_mask'].to(device_hw)
            labels = batch['labels'].to(device_hw)
            outputs = model(input_ids=ids, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    model.train()
    f1 = f1_score(all_labels, all_preds)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = total_loss / len(all_labels)
    return f1, acc, avg_loss


def run_single(lora_alpha, lr, out_noise_val, device_hw, train_loader, eval_loader):
    label = f"alpha={lora_alpha}, lr={lr}, out_noise={out_noise_val}"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  QKV, epochs={NUM_EPOCHS}")
    print(f"{'='*60}")

    set_seed(SEED)
    model = create_model(lora_alpha, out_noise_val, device_hw)

    optimizer = AnalogAdam(model.parameters(), lr=lr)
    optimizer.regroup_param_groups()

    num_steps = len(train_loader) * NUM_EPOCHS
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step) / float(max(1, WARMUP_STEPS))
        progress = float(step - WARMUP_STEPS) / float(max(1, num_steps - WARMUP_STEPS))
        return max(MIN_LR_RATIO, 1.0 - (1.0 - MIN_LR_RATIO) * progress)

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss()

    # Initial eval
    f1, acc, loss = evaluate(model, eval_loader, device_hw)
    print(f"  Epoch 0: F1={f1:.4f}  Acc={acc:.4f}  Loss={loss:.4f}")

    best_f1 = f1
    model.train()

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}", leave=False)
        for batch in pbar:
            ids = batch['input_ids'].to(device_hw)
            mask = batch['attention_mask'].to(device_hw)
            labels = batch['labels'].to(device_hw)

            optimizer.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        f1, acc, eval_loss = evaluate(model, eval_loader, device_hw)
        train_avg = total_loss / n_batches
        print(f"  Epoch {epoch}: F1={f1:.4f}  Acc={acc:.4f}  TrainLoss={train_avg:.4f}  EvalLoss={eval_loss:.4f}")
        if f1 > best_f1:
            best_f1 = f1

    print(f"  Best F1: {best_f1:.4f}")
    del model
    torch.cuda.empty_cache()
    return best_f1, acc


def main():
    device_hw = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    raw = load_dataset("nyu-mll/glue", "mrpc")
    def preprocess(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"],
                        padding="max_length", max_length=MAX_SEQ_LENGTH, truncation=True)
    tokenized = raw.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=default_data_collator)
    eval_loader = DataLoader(tokenized["validation"], batch_size=128,
                             shuffle=False, collate_fn=default_data_collator)

    print(f"MRPC: Train={len(tokenized['train'])}, Val={len(tokenized['validation'])}")
    print(f"Train batches={len(train_loader)}, Eval batches={len(eval_loader)}")

    results = {}
    # (alpha, lr) pairs — lr scaled down for larger alpha to avoid divergence
    configs = [
        (0.02, 0.002),
        (0.1,  0.001),
        (0.5,  0.0005),
        (1.0,  0.0003),
        (3.0,  0.0001),
    ]

    for alpha, lr in configs:
        for noise in [0.06, 0.0]:
            key = f"a{alpha}_n{noise}"
            f1, acc = run_single(alpha, lr, noise, device_hw, train_loader, eval_loader)
            results[key] = {'alpha': alpha, 'lr': lr, 'noise': noise, 'f1': f1, 'acc': acc}

    print(f"\n{'='*80}")
    print(f"  MRPC COMPARISON: out_noise=0.06 vs 0.0 across alpha values")
    print(f"{'='*80}")
    print(f"  {'Alpha':>8} {'LR':>10} | {'noise=0.06 F1':>14} {'Acc':>8} | {'noise=0.0 F1':>14} {'Acc':>8} | {'Delta F1':>10}")
    print(f"  {'-'*74}")
    for alpha, lr in configs:
        r_def = results[f"a{alpha}_n0.06"]
        r_fix = results[f"a{alpha}_n0.0"]
        delta = r_fix['f1'] - r_def['f1']
        print(f"  {alpha:>8.2f} {lr:>10.4f} | {r_def['f1']:>14.4f} {r_def['acc']:>8.4f} | {r_fix['f1']:>14.4f} {r_fix['acc']:>8.4f} | {delta:>+10.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
