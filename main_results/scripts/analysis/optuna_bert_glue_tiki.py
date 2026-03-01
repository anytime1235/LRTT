# -*- coding: utf-8 -*-
"""optuna_bert_glue_tiki.py — Optuna sweep for BERT-base + GLUE with TikiTaka / IdealDevice.

Derived from optuna_bert_squad_tiki.py. Replaces SQuAD-specific data/model/eval
with GLUE equivalents. Training loop uses outputs.loss directly (HF handles loss).

Usage:
    python optuna_bert_glue_tiki.py --glue-task sst2 --n-trials 5 --epochs 2 \\
        --lora-target qkv --optimizer AnalogSGD --out-dir ./results/glue_optuna_sst2

All flags:
    --glue-task {cola,sst2,mrpc,qqp,mnli,qnli,rte,wnli,stsb}
    --max-seq-length 128
    --train-subset-size 0    (0 = n_step * batch_size)
    --eval-subset-size 0
    --study-name <str>
    --n-trials <int>
    --epochs <int>
    --batch-size <int>
    --optimizer AnalogSGD|AnalogAdam
    --lora-target none|qonly|konly|vonly|qkv|ffn|all
    --target-ideal           (use IdealDevice instead of TikiTaka)
    --target-analog          (use SingleRPU instead of TikiTaka)
    --lr <float>
    --classifier-lr <float>
    --out-dir <path>
"""

import argparse
import gc
import json
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import optuna
from optuna.trial import TrialState

import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.configs import SingleRPUConfig, UnitCellRPUConfig, IOParameters, UpdateParameters
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice, IdealDevice
from aihwkit.simulator.configs.utils import BoundManagementType, NoiseManagementType

# =============================================================================
# GLUE Task Config
# =============================================================================

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp":  ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte":  ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

TASK_TO_NUM_LABELS = {
    "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
    "mnli": 3, "qnli": 2, "rte": 2, "stsb": 1, "wnli": 2,
}

IS_REGRESSION = {"stsb"}

# Primary metric per task (for Optuna maximize direction)
TASK_TO_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "f1",
    "qqp":  "f1",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte":  "accuracy",
    "stsb": "pearsonr",
    "wnli": "accuracy",
}

# =============================================================================
# Global Constants
# =============================================================================

USE_CUDA = torch.cuda.is_available()
DEVICE   = torch.device("cuda" if USE_CUDA else "cpu")

SEED           = 42
MODEL_NAME     = "bert-base-uncased"
BATCH_SIZE     = 32
EVAL_BATCH_SIZE = 128
MAX_SEQ_LENGTH = 128
N_EPOCHS       = 2
WARMUP_RATIO   = 0.05
EARLY_STOP_PATIENCE = 2

LORA_TARGET  = "qkv"
HEAD_LAYER   = "train"
TARGET_IDEAL  = False
TARGET_ANALOG = False
TARGET_LAYERS = None

TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE  = 0

CLIP_ANALOG_GRAD      = False
ANALOG_TILE_MAX_NORM  = 1.0
ANALOG_TILE_MIN_NORM  = 0.1

GLUE_TASK = "sst2"

OPT_CONFIG = {
    "optimizer":         "AnalogSGD",
    "tune_wd":           False,
    "tune_momentum":     False,
    "tune_nesterov":     False,
    "shared_lr":         True,
    "lr_range":          [1e-4, 1.0],
    "nontarget_digital": True,
    "nontarget_ideal":   False,
    "analog_only_warmup": True,
    "train_layernorm":   True,
    "backward_perfect":  False,
    "forward_perfect":   False,
    "units_in_mbatch":   True,
    "desired_bl":        31,
    "use_v2":            True,
    "scale_transfer_lr": True,
    "auto_scale":        True,
}

RESULTS = "/data/results/glue_tiki"
os.makedirs(RESULTS, exist_ok=True)
os.environ["WANDB_MODE"] = "offline"

# =============================================================================
# TikiTaka Device Functions
# =============================================================================

def _create_a_device():
    return LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=0.3,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=0.0,
        lifetime_dtod=0.0,
        reset=0.0,
        reset_dtod=0.0,
    )


def _create_b_device():
    return SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0,
        up_down=0.0, up_down_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=False,
    )


def create_tikitaka_config(transfer_every, transfer_lr, fast_lr,
                            auto_scale=False, desired_bl=31, use_v2=True):
    a_device = _create_a_device()
    b_device = _create_b_device()
    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[a_device, b_device],
            transfer_every=transfer_every,
            units_in_mbatch=OPT_CONFIG.get("units_in_mbatch", True),
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=OPT_CONFIG.get("scale_transfer_lr", use_v2),
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(
                desired_bl=desired_bl,
                update_bl_management=False if use_v2 else True,
                update_management=False if use_v2 else True,
            ),
            no_buffer=not use_v2,
            in_chop_prob=0.1 if use_v2 else 0.0,
            out_chop_prob=0.0,
            auto_scale=auto_scale,
            auto_momentum=0.99,
        )
    )
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    if OPT_CONFIG.get("backward_perfect", False):
        rpu_config.backward.is_perfect = True
    if OPT_CONFIG.get("forward_perfect", False):
        rpu_config.forward.is_perfect = True

    io_bits = OPT_CONFIG.get("io_bits", None)
    if io_bits is not None:
        io_res = 1.0 / (2 ** io_bits - 2)
        for io in [rpu_config.forward, rpu_config.backward]:
            io.inp_res = io_res
            io.out_res = io_res

    rpu_config.mapping.digital_bias             = True
    rpu_config.mapping.weight_scaling_omega     = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling        = OPT_CONFIG.get("learn_out_scaling", False)
    rpu_config.mapping.out_scaling_columnwise   = OPT_CONFIG.get("learn_out_scaling", False)
    return rpu_config


def create_single_rpu_config():
    b_device = _create_b_device()
    rpu_config = SingleRPUConfig(device=b_device)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    if OPT_CONFIG.get("backward_perfect", False):
        rpu_config.backward.is_perfect = True
    if OPT_CONFIG.get("forward_perfect", False):
        rpu_config.forward.is_perfect = True
    rpu_config.mapping.digital_bias             = True
    rpu_config.mapping.weight_scaling_omega     = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling        = OPT_CONFIG.get("learn_out_scaling", False)
    rpu_config.mapping.out_scaling_columnwise   = OPT_CONFIG.get("learn_out_scaling", False)
    return rpu_config


def create_ideal_config():
    rpu_config = SingleRPUConfig(device=IdealDevice())
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    if OPT_CONFIG.get("backward_perfect", False):
        rpu_config.backward.is_perfect = True
    if OPT_CONFIG.get("forward_perfect", False):
        rpu_config.forward.is_perfect = True
    rpu_config.mapping.digital_bias             = True
    rpu_config.mapping.weight_scaling_omega     = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling        = OPT_CONFIG.get("learn_out_scaling", False)
    rpu_config.mapping.out_scaling_columnwise   = OPT_CONFIG.get("learn_out_scaling", False)
    return rpu_config


# =============================================================================
# GLUE Data Loading (verbatim from paper_figures_glue.py)
# =============================================================================

def load_glue_data(task, tokenizer, batch_size, seed, max_length=128,
                   train_subset_size=0, eval_subset_size=0):
    """Load GLUE task with dynamic padding."""
    assert task in TASK_TO_KEYS, f"Unknown GLUE task: {task}"
    key1, key2 = TASK_TO_KEYS[task]

    raw = load_dataset("nyu-mll/glue", task)

    def preprocess(examples):
        if key2 is None:
            return tokenizer(examples[key1], max_length=max_length, truncation=True)
        return tokenizer(examples[key1], examples[key2],
                         max_length=max_length, truncation=True)

    # Train split
    train_split = raw["train"]
    tok_train = train_split.map(
        preprocess, batched=True,
        remove_columns=[c for c in train_split.column_names if c != "label"]
    )
    tok_train = tok_train.rename_column("label", "labels")

    n_train = (min(train_subset_size, len(tok_train))
               if train_subset_size > 0 else len(tok_train))
    tok_train = tok_train.shuffle(seed=seed).select(range(n_train))

    # Eval split (use validation, or validation_matched for MNLI)
    eval_key = "validation_matched" if task == "mnli" else "validation"
    eval_split = raw[eval_key]
    tok_eval = eval_split.map(
        preprocess, batched=True,
        remove_columns=[c for c in eval_split.column_names if c != "label"]
    )
    tok_eval = tok_eval.rename_column("label", "labels")

    n_eval = (min(eval_subset_size, len(tok_eval))
              if eval_subset_size > 0 else len(tok_eval))
    tok_eval = tok_eval.select(range(n_eval))

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        tok_train, batch_size=batch_size, shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(seed),
    )
    eval_loader = DataLoader(
        tok_eval, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=collator,
    )

    print(f"  GLUE({task}): train={n_train}, eval={n_eval}")
    return train_loader, eval_loader


# =============================================================================
# Model Creation
# =============================================================================

def _list_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def _classify_encoder_layer(layer_name):
    if "attention" in layer_name:
        return "attention"
    return "ffn"


def _is_tikitaka_target(layer_name, lora_target, target_layers, always_digital):
    if any(d in layer_name for d in always_digital):
        return False
    if "encoder" not in layer_name:
        return False
    if target_layers is not None:
        m = re.search(r"layer\.(\d+)", layer_name)
        if m is None or int(m.group(1)) not in target_layers:
            return False
    cat = _classify_encoder_layer(layer_name)
    if lora_target == "none":
        return False
    elif lora_target == "qkv":
        return cat == "attention"
    elif lora_target == "ffn":
        return cat == "ffn"
    elif lora_target == "all":
        return cat in ("attention", "ffn")
    elif lora_target in ("qonly", "konly", "vonly"):
        patterns = {"qonly": ["query"], "konly": ["key"], "vonly": ["value"]}[lora_target]
        return any(p in layer_name for p in patterns)
    return False


def create_model(params, num_labels, task):
    """BERT-base SequenceClassification with selective TikiTaka / Ideal analog layers.

    Excludes ["classifier", "pooler"] from analog (always digital, trainable).
    STS-B: problem_type="regression" set in model config (HF handles MSE loss).
    """
    always_digital = ["classifier", "pooler"]

    model_kwargs = {"num_labels": num_labels}
    if task in IS_REGRESSION:
        model_kwargs["problem_type"] = "regression"

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, **model_kwargs
    )

    # Re-init classifier with fixed seed for reproducibility
    if hasattr(model, "classifier"):
        torch.manual_seed(SEED)
        for mod in model.classifier.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    all_linear = _list_linear_layers(model)

    tikitaka_layers = [
        n for n in all_linear
        if _is_tikitaka_target(n, LORA_TARGET, TARGET_LAYERS, always_digital)
    ]
    non_target_encoder = [
        n for n in all_linear
        if n not in tikitaka_layers and "encoder" in n
        and not any(d in n for d in always_digital)
    ]

    target_layer_names = set(tikitaka_layers)
    tikitaka_count = ideal_count = target_analog_count = 0

    def _frozen_noop(x_in, d_in, *a, **kw):
        return None

    # Pass 1: target layers
    if tikitaka_layers and LORA_TARGET != "none":
        if TARGET_IDEAL:
            ideal_cfg = create_ideal_config()
            model = convert_to_analog(
                model, ideal_cfg,
                exclude_modules=[n for n in all_linear if n not in tikitaka_layers]
            )
            ideal_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
        elif TARGET_ANALOG:
            single_cfg = create_single_rpu_config()
            model = convert_to_analog(
                model, single_cfg,
                exclude_modules=[n for n in all_linear if n not in tikitaka_layers]
            )
            target_analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
        else:
            tiki_cfg = create_tikitaka_config(
                transfer_every=int(params["transfer_every"]),
                transfer_lr=params["transfer_lr"],
                fast_lr=params["fast_lr"],
                auto_scale=OPT_CONFIG.get("auto_scale", True),
                desired_bl=int(params["desired_bl"]),
                use_v2=OPT_CONFIG.get("use_v2", True),
            )
            model = convert_to_analog(
                model, tiki_cfg,
                exclude_modules=[n for n in all_linear if n not in tikitaka_layers]
            )
            tikitaka_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))

    # Pass 2: non-target encoder layers
    if non_target_encoder and not OPT_CONFIG.get("nontarget_digital", True):
        if OPT_CONFIG.get("nontarget_ideal", False):
            nt_cfg = create_ideal_config()
        else:
            nt_cfg = create_single_rpu_config()
        model = convert_to_analog(
            model, nt_cfg,
            exclude_modules=[n for n in all_linear if n not in non_target_encoder],
            inplace=True,
        )
        for name, m in model.named_modules():
            if isinstance(m, AnalogLinear) and name not in target_layer_names:
                for tile in m.analog_tiles():
                    tile.update = _frozen_noop

    # Gradient control
    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext):
            param.requires_grad = True
        elif any(d in name for d in always_digital):
            param.requires_grad = (HEAD_LAYER == "train")
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = OPT_CONFIG.get("train_layernorm", True)
        elif "out_scaling" in name:
            param.requires_grad = OPT_CONFIG.get("learn_out_scaling", False)
        else:
            param.requires_grad = False

    n_analog = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  AnalogLinear: {n_analog}, Trainable params: {n_trainable:,}")
    return model.to(DEVICE)


# =============================================================================
# GLUE Evaluation
# =============================================================================

def _batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def evaluate_model_glue(model, eval_loader, task_name, device):
    """Evaluate GLUE model. Returns metric dict."""
    metric = evaluate.load("glue", task_name)
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            bd = _batch_to_device(batch, device)
            logits = model(**bd).logits
            if task_name in IS_REGRESSION:
                preds = logits.squeeze(-1)
            else:
                preds = logits.argmax(dim=-1)
            metric.add_batch(
                predictions=preds.cpu(),
                references=batch["labels"],
            )
    model.train()
    return metric.compute()


def _primary_metric(results, task_name):
    """Extract primary metric value for Optuna objective."""
    key = TASK_TO_METRIC.get(task_name, "accuracy")
    if key in results:
        return float(results[key])
    # Fallback: return first numeric value
    for v in results.values():
        if isinstance(v, (int, float)):
            return float(v)
    return float("nan")


# =============================================================================
# LR Scheduler
# =============================================================================

def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps,
                                     min_lr_rate=0.0, warmup_analog_only=True):
    def _make_lambda(apply_warmup):
        def lr_lambda(current_step):
            if apply_warmup and current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = max(0.0, float(current_step - num_warmup_steps)) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))
        return lr_lambda

    if warmup_analog_only:
        lambdas = []
        for group in optimizer.param_groups:
            is_analog = any(isinstance(p, AnalogContext) for p in group["params"])
            lambdas.append(_make_lambda(apply_warmup=is_analog))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    else:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, _make_lambda(apply_warmup=True))


# =============================================================================
# Optuna Objective
# =============================================================================

def objective(trial, train_loader, eval_loader, tokenizer):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # LR
    _lr_range = OPT_CONFIG.get("lr_range", None)
    if _lr_range is not None:
        learning_rate = trial.suggest_float("learning_rate", _lr_range[0], _lr_range[1], log=True)
    else:
        learning_rate = OPT_CONFIG.get("lr_override", 2e-3)

    classifier_lr = learning_rate if OPT_CONFIG.get("shared_lr", True) else \
        OPT_CONFIG.get("classifier_lr", learning_rate)

    # TikiTaka params
    _has_tikitaka = LORA_TARGET != "none" and not TARGET_IDEAL and not TARGET_ANALOG
    if not _has_tikitaka:
        fast_lr = transfer_lr = 1.0
    else:
        fast_lr = 1.0
        _tlr_upper = 1000.0
        if OPT_CONFIG.get("scale_transfer_lr", True):
            _tlr_upper = min(1000.0, 1.0 / learning_rate)
        transfer_lr = trial.suggest_float("transfer_lr", 1.0, _tlr_upper, log=True)

    desired_bl     = OPT_CONFIG.get("desired_bl", 31)
    transfer_every = OPT_CONFIG.get("transfer_every_override", 1)

    weight_decay = 0.0
    if OPT_CONFIG["tune_wd"]:
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)

    min_lr_rate = 0.5
    optimizer_name = OPT_CONFIG["optimizer"]

    params = {
        "transfer_every": transfer_every,
        "transfer_lr":    transfer_lr,
        "fast_lr":        fast_lr,
        "desired_bl":     desired_bl,
    }

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting (task={GLUE_TASK}, "
          f"metric={TASK_TO_METRIC[GLUE_TASK]})")
    print(f"  lr={learning_rate:.2e}, classifier_lr={classifier_lr:.2e}, "
          f"wd={weight_decay:.2e}")
    print(f"  transfer_lr={transfer_lr:.4e}, desired_bl={desired_bl}, "
          f"optimizer={optimizer_name}")
    print(f"{'='*70}")

    model = None
    try:
        set_seed(SEED)
        model = create_model(params, num_labels=TASK_TO_NUM_LABELS[GLUE_TASK],
                             task=GLUE_TASK)

        # Optimizer
        analog_params  = [p for p in model.parameters()
                          if isinstance(p, AnalogContext) and p.requires_grad]
        digital_params = [p for p in model.parameters()
                          if not isinstance(p, AnalogContext) and p.requires_grad]

        param_groups = []
        if analog_params:
            param_groups.append({"params": analog_params, "lr": learning_rate})
        if digital_params:
            param_groups.append({"params": digital_params, "lr": classifier_lr})

        _pg = param_groups if param_groups else model.parameters()

        if optimizer_name == "AnalogSGD":
            optimizer = AnalogSGD(_pg, weight_decay=weight_decay)
        else:
            optimizer = AnalogAdam(_pg, weight_decay=weight_decay)
        optimizer.regroup_param_groups()

        num_training_steps = len(train_loader) * N_EPOCHS
        warmup_steps       = int(num_training_steps * WARMUP_RATIO)
        _warmup_analog_only = OPT_CONFIG.get("analog_only_warmup", True)
        scheduler = get_linear_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_rate=min_lr_rate,
            warmup_analog_only=_warmup_analog_only,
        )

        best_metric = float("-inf")
        epochs_without_improvement = 0
        global_step = 0

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
            for batch in pbar:
                global_step += 1
                bd = _batch_to_device(batch, DEVICE)

                optimizer.zero_grad()
                # GLUE batch: input_ids, attention_mask, labels
                # HF SequenceClassification computes loss automatically with labels
                outputs = model(**bd)
                loss = outputs.loss
                loss.backward()

                # Clip digital grads
                _dparams = [p for p in model.parameters()
                            if not isinstance(p, AnalogContext) and p.grad is not None]
                if _dparams:
                    torch.nn.utils.clip_grad_norm_(_dparams, max_norm=1.0)

                scheduler.step()
                # Sync analog tile lr
                for _pg2 in optimizer.param_groups:
                    for _p in _pg2["params"]:
                        if isinstance(_p, AnalogContext):
                            _p.analog_tile.set_learning_rate(_pg2["lr"])
                optimizer.step()

                loss_val = loss.item()
                total_loss += loss_val
                num_batches += 1
                pbar.set_postfix(loss=f"{loss_val:.4f}")

                if not np.isfinite(loss_val) or loss_val > 1e8:
                    tqdm.write(f"[Trial {trial.number}] Loss diverged, stopping early.")
                    trial.set_user_attr("diverged", True)
                    return float("-inf")

            train_loss = total_loss / num_batches if num_batches > 0 else 0.0

            results = evaluate_model_glue(model, eval_loader, GLUE_TASK, DEVICE)
            metric_val = _primary_metric(results, GLUE_TASK)

            improved = ""
            if metric_val > best_metric:
                best_metric = metric_val
                epochs_without_improvement = 0
                improved = " *"
            else:
                epochs_without_improvement += 1

            current_lr = optimizer.param_groups[0]["lr"]
            tqdm.write(
                f"[Trial {trial.number}] Ep{epoch:3d} | "
                f"{TASK_TO_METRIC[GLUE_TASK]}={metric_val:.4f} | "
                f"Best={best_metric:.4f} | Loss={train_loss:.4f} | "
                f"LR={current_lr:.2e} | NoImp={epochs_without_improvement}{improved}"
            )
            tqdm.write(f"  Full results: {results}")

            trial.report(metric_val, epoch)
            trial.set_user_attr(f"train_loss_ep{epoch}", train_loss)

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}")
                break

            if trial.should_prune():
                tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        print(f"\n[Trial {trial.number}] Best {TASK_TO_METRIC[GLUE_TASK]}: {best_metric:.4f}")
        return best_metric

    except Exception as e:
        error_msg = str(e)[:500]
        trial.set_user_attr("error", error_msg)
        print(f"[Trial {trial.number}] Error: {error_msg}")
        raise

    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Visualization
# =============================================================================

def visualize_study(study, save_dir):
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete:
        print("No completed trials to visualize.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    nums   = [t.number for t in complete]
    values = [t.value  for t in complete]
    axes[0].scatter(nums, values, alpha=0.6)
    axes[0].plot(nums, [max(values[:i+1]) for i in range(len(values))],
                 "r-", linewidth=2, label="Best so far")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel(TASK_TO_METRIC[GLUE_TASK])
    axes[0].set_title("Optimization History")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    try:
        importances = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importances.keys())[::-1], list(importances.values())[::-1])
        axes[1].set_xlabel("Importance")
        axes[1].set_title("Parameter Importance")
    except Exception:
        axes[1].text(0.5, 0.5, "Not enough trials", ha="center", va="center",
                     transform=axes[1].transAxes)
    plt.tight_layout()
    path = os.path.join(save_dir, f"visualization_bert_glue_{GLUE_TASK}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved → {path}")


def print_study_summary(study):
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"\n{'='*60}")
    print("STUDY SUMMARY")
    print(f"{'='*60}")
    print(f"Study: {study.study_name}, Trials: {len(study.trials)} ({len(complete)} complete)")
    if complete:
        vals = [t.value for t in complete]
        print(f"Best {TASK_TO_METRIC[GLUE_TASK]}: {max(vals):.4f}, "
              f"Mean: {sum(vals)/len(vals):.4f}")
        print(f"Best params: {study.best_params}")


# =============================================================================
# Main
# =============================================================================

def main():
    global BATCH_SIZE, N_EPOCHS, WARMUP_RATIO, LORA_TARGET, HEAD_LAYER
    global TARGET_IDEAL, TARGET_ANALOG, TARGET_LAYERS, CLIP_ANALOG_GRAD
    global GLUE_TASK, TRAIN_SUBSET_SIZE, EVAL_SUBSET_SIZE, RESULTS

    parser = argparse.ArgumentParser(
        description="Optuna sweep for BERT-base GLUE TikiTaka")
    parser.add_argument("--glue-task", type=str, default="sst2",
                        choices=list(TASK_TO_KEYS.keys()))
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--train-subset-size", type=int, default=0)
    parser.add_argument("--eval-subset-size",  type=int, default=0)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--n-trials",  type=int, default=10)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--optimizer", type=str, default="AnalogSGD",
                        choices=["AnalogSGD", "AnalogAdam"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs",    type=int, default=N_EPOCHS)
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--lora-target", type=str, default=LORA_TARGET,
                        choices=["none", "qonly", "konly", "vonly", "qkv", "ffn", "all"])
    parser.add_argument("--head-layer", type=str, default=HEAD_LAYER,
                        choices=["train", "freeze"])
    parser.add_argument("--target-ideal",  action="store_true", default=False)
    parser.add_argument("--target-analog", action="store_true", default=False)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--lr-range",   type=float, nargs=2, default=None)
    parser.add_argument("--classifier-lr", type=float, default=None)
    parser.add_argument("--shared-lr",  action="store_true", default=True)
    parser.add_argument("--no-wd",      action="store_true", default=True)
    parser.add_argument("--desired-bl", type=int, default=31)
    parser.add_argument("--out-dir",    type=str, default="./results/glue_optuna")
    args = parser.parse_args()

    GLUE_TASK         = args.glue_task
    BATCH_SIZE        = args.batch_size
    N_EPOCHS          = args.epochs
    WARMUP_RATIO      = args.warmup_ratio
    LORA_TARGET       = args.lora_target
    HEAD_LAYER        = args.head_layer
    TARGET_IDEAL      = args.target_ideal
    TARGET_ANALOG     = args.target_analog
    TRAIN_SUBSET_SIZE = args.train_subset_size
    EVAL_SUBSET_SIZE  = args.eval_subset_size
    RESULTS           = args.out_dir
    os.makedirs(RESULTS, exist_ok=True)

    OPT_CONFIG["optimizer"]  = args.optimizer
    OPT_CONFIG["tune_wd"]    = not args.no_wd
    OPT_CONFIG["shared_lr"]  = args.shared_lr
    OPT_CONFIG["desired_bl"] = args.desired_bl
    if args.lr is not None:
        OPT_CONFIG["lr_override"] = args.lr
        OPT_CONFIG["lr_range"]    = None
    if args.lr_range is not None:
        OPT_CONFIG["lr_range"] = args.lr_range
    if args.classifier_lr is not None:
        OPT_CONFIG["classifier_lr"] = args.classifier_lr

    from datetime import datetime
    timestamp  = datetime.now().strftime("%m%d_%H%M")
    study_name = args.study_name or f"bert_glue_{GLUE_TASK}_{timestamp}"
    storage    = f"sqlite:///{RESULTS}/optuna_{study_name}.db"

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    MAX_SEQ_LENGTH = args.max_seq_length

    print(f"[Config] Task={GLUE_TASK}, Metric={TASK_TO_METRIC[GLUE_TASK]}, "
          f"NumLabels={TASK_TO_NUM_LABELS[GLUE_TASK]}")
    print(f"[Config] BSZ={BATCH_SIZE}, Epochs={N_EPOCHS}, "
          f"MaxLen={MAX_SEQ_LENGTH}, Device={DEVICE}")

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print_study_summary(study)
        visualize_study(study, RESULTS)
        return

    train_loader, eval_loader = load_glue_data(
        task=GLUE_TASK,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        seed=SEED,
        max_length=MAX_SEQ_LENGTH,
        train_subset_size=TRAIN_SUBSET_SIZE,
        eval_subset_size=EVAL_SUBSET_SIZE,
    )

    sampler = optuna.samplers.TPESampler(seed=SEED)
    prune_warmup = max(1, N_EPOCHS // 3)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=prune_warmup
        ),
        load_if_exists=True,
    )

    print(f"\nStudy: {study_name}, Device: {DEVICE}, Trials: {args.n_trials}")

    study.optimize(
        lambda trial: objective(trial, train_loader, eval_loader, tokenizer),
        n_trials=args.n_trials,
        catch=(Exception,),
        show_progress_bar=False,
    )

    print_study_summary(study)
    visualize_study(study, RESULTS)

    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if complete:
        best_path = os.path.join(RESULTS, f"best_params_{study_name}.json")
        with open(best_path, "w") as f:
            json.dump({
                "task":        GLUE_TASK,
                "metric":      TASK_TO_METRIC[GLUE_TASK],
                "best_value":  study.best_value,
                "best_params": study.best_params,
            }, f, indent=2)
        print(f"Best params saved → {best_path}")


if __name__ == "__main__":
    main()
