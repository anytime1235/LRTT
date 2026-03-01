#!/usr/bin/env python3
"""
Single trial: SST-2 AnalogSGD, QKV + classifier trainable
Using best HP from classifier-frozen sweep (Trial 16, 86.58%)
"""
import os, sys, json, torch, numpy as np
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    default_data_collator, get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import (
    UnitCellRPUConfig,
    IOParameters, UpdateParameters,
    NoiseManagementType, BoundManagementType,
)
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsReferenceDevice
from aihwkit.optim import AnalogSGD

# ── Config ──
SEED = 42
MODEL_NAME = "google/mobilebert-uncased"
BATCH_SIZE = 64
EPOCHS = 3
MAX_SEQ_LENGTH = 128
WARMUP_RATIO = 0.1
TARGET_MODULES = ["query", "key", "value", "classifier"]

# Best HP from classifier-frozen SGD sweep Trial 16 (86.58%)
PARAMS = {
    "learning_rate": 0.0022765456563752597,
    "transfer_lr": 0.20709428549474193,
    "fast_lr": 3.0,
    "transfer_every": 475,
    "auto_granularity": 169.0,
    "in_chop_prob": 0.061,
}

torch.manual_seed(SEED)
np.random.seed(SEED)

def list_linear_layers(model):
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.append(name)
    return names

def create_tikitaka_v2_config(transfer_every, transfer_lr, fast_lr,
                               auto_granularity, in_chop_prob):
    a_device = LinearStepDevice(
        w_max=1.0, w_min=-1.0,
        w_max_dtod=0.05, w_min_dtod=0.05,
        dw_min=0.001981, dw_min_std=0.3, dw_min_dtod=0.1,
        write_noise_std=0.0, mult_noise=True,
        gamma_up=-0.1678, gamma_down=0.1410,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        up_down=0.0, up_down_dtod=0.01,
        mean_bound_reference=True,
        lifetime=0, lifetime_dtod=0.0,
        reset=0.0, reset_dtod=0.0,
    )
    c_device = SoftBoundsReferenceDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_std=0.0, dw_min_dtod=0.0,
        write_noise_std=0.0, up_down=0.0, up_down_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0,
        mult_noise=False,
    )
    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[a_device, a_device, c_device],
            transfer_every=transfer_every,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            gamma=0.0,
            auto_scale=True,
            auto_granularity=auto_granularity,
            buffer_granularity=1.0,
            auto_momentum=0.99,
            in_chop_prob=in_chop_prob,
            in_chop_random=True,
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(
                desired_bl=1,
                update_bl_management=False,
                update_management=False,
            ),
        )
    )
    rpu_config.forward = IOParameters(
        is_perfect=False, inp_noise=0.0, out_noise=0.0,
        out_noise_std=0.0, w_noise=0.0,
        noise_management=NoiseManagementType.ABS_MAX,
        bound_management=BoundManagementType.ITERATIVE,
    )
    rpu_config.backward = IOParameters(
        is_perfect=False, inp_noise=0.0, out_noise=0.0,
        out_noise_std=0.0, w_noise=0.0,
        noise_management=NoiseManagementType.ABS_MAX,
        bound_management=BoundManagementType.ITERATIVE,
    )
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target modules: {TARGET_MODULES}")
    print(f"Params: {json.dumps(PARAMS, indent=2)}")

    # Load data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw = load_dataset("nyu-mll/glue", "sst2")

    def tokenize(examples):
        return tokenizer(examples["sentence"], truncation=True,
                         max_length=MAX_SEQ_LENGTH, padding="max_length")

    tokenized = raw.map(tokenize, batched=True,
                        remove_columns=[c for c in raw["train"].column_names if c not in ["label"]])
    tokenized.set_format("torch")

    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=default_data_collator)
    eval_loader = DataLoader(tokenized["validation"], batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=default_data_collator)

    # Create model
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    # Classifier is TRAINABLE — use pretrained init (no reinit)
    print("[INFO] Classifier is TRAINABLE (in TARGET_MODULES) — using pretrained init")

    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("classifier")  # classifier always digital

    rpu_config = create_tikitaka_v2_config(**{k: PARAMS[k] for k in
                    ["transfer_every", "transfer_lr", "fast_lr", "auto_granularity", "in_chop_prob"]})
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    # Set trainability
    trainable_count = 0
    frozen_count = 0
    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        is_head = "classifier" in name
        if is_head:
            param.requires_grad = True  # classifier always trainable in this experiment
            trainable_count += param.numel()
        elif "bias" in name:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            param.requires_grad = is_target
            if is_target:
                trainable_count += param.numel()
            else:
                frozen_count += param.numel()

    print(f"Trainable digital params: {trainable_count:,}")
    model = model.to(device)

    # Optimizer & scheduler
    optimizer = AnalogSGD(model.parameters(), lr=PARAMS["learning_rate"])
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Train
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 100 == 0:
                print(f"  Ep{epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
        acc = correct / total
        print(f"[Epoch {epoch}/{EPOCHS}] Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc

    print(f"\n{'='*60}")
    print(f"RESULT: Best Accuracy = {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Baseline (classifier frozen): 86.58%")
    print(f"Delta: {(best_acc - 0.8658)*100:+.2f}%p")
    print(f"{'='*60}")

    # Save
    results_dir = "/data/results/tikitaka/glue"
    os.makedirs(results_dir, exist_ok=True)
    result = {
        "experiment": "sst2_sgd_classifier_trainable_single_trial",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "sst2",
        "optimizer": "AnalogSGD",
        "classifier_trainable": True,
        "target_modules": TARGET_MODULES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "params": PARAMS,
        "hp_source": "classifier-frozen SGD sweep Trial 16 (86.58%)",
        "best_accuracy": best_acc,
        "baseline_frozen_acc": 0.8658,
        "delta": best_acc - 0.8658,
    }
    outfile = os.path.join(results_dir, "sst2_sgd_classifier_trainable_trial16hp.json")
    with open(outfile, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {outfile}")


if __name__ == "__main__":
    main()
