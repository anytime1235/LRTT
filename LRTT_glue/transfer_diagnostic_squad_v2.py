#!/usr/bin/env python3
"""Transfer Diagnostic: Track per-transfer metrics for SQuAD (MobileBERT, rank 8, V-only, hybrid reinit).

Investigates whether LRTT transfers become meaningless in later epochs —
specifically, whether C stops changing despite transfers being triggered.

Adapts the MNIST transfer_diagnostic.py approach to MobileBERT on SQuAD
using the same model/config as sweep_lrtt_squad_rank8_v_only_15ep.py.

Metrics tracked per transfer:
- a_norm: ‖A[:, :rank]‖_F
- b_norm: ‖B[:rank, :]‖_F
- ab_magnitude: ‖tlr × A @ B‖_F  (intended transfer signal)
- c_norm: ‖C‖_F before transfer
- delta_c_norm: ‖C_after - C_before‖_F  (actual change in C)
- delta_ratio: delta_c_norm / ab_magnitude  (transfer efficiency)
- unchanged_elem_ratio: fraction of C elements where |ΔC| < 1e-7
- cosine_sim: cosine(AB_intended, delta_C)
- sign_agree_all: fraction of nonzero elements with matching sign
- sign_agree_changed: fraction of CHANGED elements with matching sign
- signal_ratio_changed: |projection onto AB| / |delta_c| among changed elements
"""

import os

os.environ["LRTT_SILENT"] = "1"

import sys
import csv
import re
import string
import math
import collections
from dataclasses import dataclass, fields
from collections import Counter
from time import time

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from torch.utils.data import DataLoader
import evaluate

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

# LRTT imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LRTT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(LRTT_ROOT, "src"))
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# =============================================================================
# Fixed Parameters (from sweep_lrtt_squad_rank8_v_only_15ep.py DEFAULT)
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "google/mobilebert-uncased"
TARGET_MODULES = ["value"]
RANK = 8
LR = 0.00971
TLR = 2.12e-4
TE = 2300
REINIT_GAIN = 0.1
LORA_ALPHA = 1.0
BATCH_SIZE = 32
NUM_EPOCHS = 15
WARMUP_STEPS = 500
MIN_LR_RATE = 0.1
SEED = 42

OUTPUT_DIR = "/root/results/squad_v2"


# =============================================================================
# SQuAD F1 Evaluation Helpers
# =============================================================================


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# =============================================================================
# Transfer Tracker (from experiments/transfer_diagnostic.py)
# =============================================================================


@dataclass
class TransferRecord:
    """Single transfer event record."""

    epoch: int
    batch_idx: int
    layer_name: str
    a_norm: float
    b_norm: float
    ab_magnitude: float
    c_norm: float
    delta_c_norm: float
    delta_ratio: float
    unchanged_elem_ratio: float
    # Signal vs noise metrics
    cosine_sim: float  # cosine(AB_intended, delta_C): 1=perfect signal, 0=pure noise
    sign_agree_all: float  # fraction of all elements with matching sign
    sign_agree_changed: float  # fraction of CHANGED elements with matching sign
    signal_ratio_changed: float  # among changed elements: |projection onto AB| / |delta_c|

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class TransferTracker:
    """Collects per-transfer diagnostic metrics."""

    def __init__(self) -> None:
        self.records: list[TransferRecord] = []
        self.current_epoch: int = 0
        self.current_batch: int = 0

    def add(self, record: TransferRecord) -> None:
        self.records.append(record)

    def epoch_summary(self, epoch: int) -> dict:
        """Compute summary statistics for a given epoch."""
        epoch_records = [r for r in self.records if r.epoch == epoch]
        if not epoch_records:
            return {"epoch": epoch, "n_transfers": 0}

        def _stats(values: list[float]) -> dict:
            return {
                "mean": sum(values) / len(values),
                "max": max(values),
                "min": min(values),
            }

        return {
            "epoch": epoch,
            "n_transfers": len(epoch_records),
            "a_norm": _stats([r.a_norm for r in epoch_records]),
            "b_norm": _stats([r.b_norm for r in epoch_records]),
            "ab_magnitude": _stats([r.ab_magnitude for r in epoch_records]),
            "c_norm": _stats([r.c_norm for r in epoch_records]),
            "delta_c_norm": _stats([r.delta_c_norm for r in epoch_records]),
            "delta_ratio": _stats([r.delta_ratio for r in epoch_records]),
            "unchanged_elem_ratio": _stats([r.unchanged_elem_ratio for r in epoch_records]),
            "cosine_sim": _stats([r.cosine_sim for r in epoch_records]),
            "sign_agree_all": _stats([r.sign_agree_all for r in epoch_records]),
            "sign_agree_changed": _stats([r.sign_agree_changed for r in epoch_records]),
            "signal_ratio_changed": _stats([r.signal_ratio_changed for r in epoch_records]),
        }

    def all_records_as_dicts(self) -> list[dict]:
        return [r.to_dict() for r in self.records]


# =============================================================================
# Monkey-patching (from experiments/transfer_diagnostic.py)
# =============================================================================


def patch_controller(controller, tracker: TransferTracker, layer_name: str) -> None:
    """Monkey-patch controller.ab_weight_transfer to record metrics."""
    original_transfer = controller.ab_weight_transfer
    rank = controller.rank
    tlr = controller.transfer_lr

    def tracked_transfer(method=None):
        # --- Before transfer: read A, B, C ---
        A = controller.tile_a.get_weights()[0]  # [d_size, rank_padded]
        B = controller.tile_b.get_weights()[0]  # [rank_padded, x_size]
        C_before = controller.tile_c.get_weights()[0].clone()  # [d_size, x_size]

        # Only use active rank columns/rows
        A_lr = A[:, :rank]  # [d_size, rank]
        B_lr = B[:rank, :]  # [rank, x_size]

        a_norm = torch.linalg.norm(A_lr).item()
        b_norm = torch.linalg.norm(B_lr).item()
        ab_product = tlr * (A_lr @ B_lr)
        ab_magnitude = torch.linalg.norm(ab_product).item()
        c_norm = torch.linalg.norm(C_before).item()

        # --- Perform actual transfer ---
        original_transfer(method=method)

        # --- After transfer: measure C change ---
        C_after = controller.tile_c.get_weights()[0]
        delta_c = C_after - C_before
        delta_c_norm = torch.linalg.norm(delta_c).item()

        # Transfer efficiency
        delta_ratio = delta_c_norm / ab_magnitude if ab_magnitude > 1e-12 else 0.0

        # Fraction of unchanged elements
        changed_mask = delta_c.abs() >= 1e-7
        unchanged = 1.0 - changed_mask.float().mean().item()

        # --- Signal vs noise analysis ---
        ab_flat = ab_product.flatten()
        dc_flat = delta_c.flatten()

        # Cosine similarity: how aligned is actual change with intended signal
        cos_denom = torch.linalg.norm(ab_flat) * torch.linalg.norm(dc_flat)
        cosine_sim = (torch.dot(ab_flat, dc_flat) / cos_denom).item() if cos_denom > 1e-12 else 0.0

        # Sign agreement (all elements where both are nonzero)
        both_nonzero = (ab_flat.abs() > 1e-12) & (dc_flat.abs() > 1e-12)
        if both_nonzero.sum() > 0:
            sign_agree_all = (
                (ab_flat[both_nonzero].sign() == dc_flat[both_nonzero].sign()).float().mean().item()
            )
        else:
            sign_agree_all = 0.0

        # Sign agreement among CHANGED elements only
        changed_flat = changed_mask.flatten()
        changed_and_signal = changed_flat & (ab_flat.abs() > 1e-12)
        if changed_and_signal.sum() > 0:
            sign_agree_changed = (
                (ab_flat[changed_and_signal].sign() == dc_flat[changed_and_signal].sign())
                .float()
                .mean()
                .item()
            )
        else:
            sign_agree_changed = 0.0

        # Signal ratio among changed elements
        if changed_flat.sum() > 0 and ab_magnitude > 1e-12:
            dc_changed = dc_flat[changed_flat]
            ab_changed = ab_flat[changed_flat]
            proj = torch.dot(dc_changed, ab_changed) / torch.linalg.norm(ab_changed)
            signal_ratio_changed = abs(proj.item()) / torch.linalg.norm(dc_changed).item()
        else:
            signal_ratio_changed = 0.0

        record = TransferRecord(
            epoch=tracker.current_epoch,
            batch_idx=tracker.current_batch,
            layer_name=layer_name,
            a_norm=a_norm,
            b_norm=b_norm,
            ab_magnitude=ab_magnitude,
            c_norm=c_norm,
            delta_c_norm=delta_c_norm,
            delta_ratio=delta_ratio,
            unchanged_elem_ratio=unchanged,
            cosine_sim=cosine_sim,
            sign_agree_all=sign_agree_all,
            sign_agree_changed=sign_agree_changed,
            signal_ratio_changed=signal_ratio_changed,
        )
        tracker.add(record)

    controller.ab_weight_transfer = tracked_transfer


# =============================================================================
# Cosine Schedule with Min LR (from sweep script)
# =============================================================================


def get_cosine_with_min_lr_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_rate: float = 0.1,
):
    """Cosine schedule with warmup and a minimum LR floor."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_rate + (1.0 - min_lr_rate) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# LRTT Config (from sweep script)
# =============================================================================


def create_lrtt_config(
    rank: int,
    transfer_every: int,
    transfer_lr: float,
    lora_alpha: float = 1.0,
    reinit_gain: float = 0.1,
    reinit_mode: str = "hybrid",
) -> PythonLRTTRPUConfig:
    """Create LRTT config with 6T1C analog device simulation."""
    # Calculate lifetime for 6T1C
    TAU_SEC = 46505.0
    dt_batch_sec = 1.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

    # A/B tiles: LinearStepDevice (6T1C)
    ab_device = LinearStepDevice(
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
        lifetime=lifetime,
        lifetime_dtod=0.0,
        reset=0.0,
        reset_dtod=0.0,
    )

    # C tile: SoftBoundsDevice (noise-free), w_max=3.0 for MobileBERT pretrained weights
    c_device = SoftBoundsDevice(
        dw_min=0.001,
        w_max=3.0,
        w_min=-3.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=True,
    )

    # LRTT Device config
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        reinit_gain=reinit_gain,
        reinit_mode=reinit_mode,
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = transfer_lr
    device_config.units_in_mbatch = True
    device_config.forward_inject = False
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


def list_linear_layers(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


# =============================================================================
# SQuAD Model & Data (from sweep script)
# =============================================================================


def create_squad_model(device: torch.device, reinit_mode: str = "hybrid") -> nn.Module:
    """Create SQuAD model with LRTT (V-only)."""
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("qa_outputs")

    rpu_config = create_lrtt_config(
        rank=RANK,
        transfer_every=TE,
        transfer_lr=TLR,
        lora_alpha=LORA_ALPHA,
        reinit_gain=REINIT_GAIN,
        reinit_mode=reinit_mode,
    )

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        param.requires_grad = is_target or "qa_outputs" in name

    return model.to(device)


def load_squad_data(tokenizer):
    """Load and tokenize SQuAD dataset (10K train, 2K eval subsets)."""
    raw_datasets = load_dataset("squad")

    eval_examples = raw_datasets["validation"].select(
        range(min(2000, len(raw_datasets["validation"])))
    )

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]

        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]

            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_eval(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs["offset_mapping"]

        for i in range(len(inputs["input_ids"])):
            sequence_ids = inputs.sequence_ids(i)
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset_mapping[i])
            ]

        inputs["example_id"] = [
            examples["id"][sample_map[i]] for i in range(len(inputs["input_ids"]))
        ]

        return inputs

    tokenized_train = raw_datasets["train"].map(
        preprocess_train, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    train_subset = tokenized_train.shuffle(seed=SEED).select(
        range(min(10000, len(tokenized_train)))
    )

    tokenized_eval = eval_examples.map(
        preprocess_eval, batched=True, remove_columns=raw_datasets["validation"].column_names
    )

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator
    )

    return train_loader, tokenized_eval, eval_examples


# =============================================================================
# SQuAD Evaluation (from sweep script)
# =============================================================================


def postprocess_squad_predictions(
    examples,
    features,
    all_start_logits,
    all_end_logits,
    n_best_size: int = 20,
    max_answer_length: int = 30,
):
    """Post-process SQuAD predictions with n-best answer selection."""
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]

        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[
            :n_best_size
        ]

        if len(predictions) == 0:
            all_predictions[example["id"]] = ""
        else:
            best_pred = predictions[0]
            start_char, end_char = best_pred["offsets"]
            all_predictions[example["id"]] = context[start_char:end_char]

    return all_predictions


def evaluate_squad(model, eval_features, eval_examples, tokenizer, device) -> tuple[float, float]:
    """Evaluate SQuAD model. Returns (f1, exact_match)."""
    model.eval()

    all_start_logits = []
    all_end_logits = []

    def squad_eval_collate_fn(features):
        offset_mappings = [f.pop("offset_mapping") for f in features]
        example_ids = [f.pop("example_id") for f in features]
        batch = default_data_collator(features)
        batch["offset_mapping"] = offset_mappings
        batch["example_id"] = example_ids
        for i, f in enumerate(features):
            f["offset_mapping"] = offset_mappings[i]
            f["example_id"] = example_ids[i]
        return batch

    eval_loader = DataLoader(
        eval_features, batch_size=BATCH_SIZE, shuffle=False, collate_fn=squad_eval_collate_fn
    )

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    model.train()

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    predictions = postprocess_squad_predictions(
        eval_examples,
        eval_features,
        all_start_logits,
        all_end_logits,
        n_best_size=20,
        max_answer_length=30,
    )

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted_predictions, references=references)

    return results["f1"], results["exact_match"]


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_steps", type=int, default=0, help="Stop after N steps (0 = full run)"
    )
    parser.add_argument(
        "--reinit_mode", type=str, default="hybrid", choices=["hybrid", "decay"],
        help="Reinit mode for LRTT (default: hybrid)"
    )
    args = parser.parse_args()
    max_steps = args.max_steps
    reinit_mode = args.reinit_mode

    print("=" * 70)
    print(f"LRTT Transfer Diagnostic — SQuAD (MobileBERT, rank 8, V-only, {reinit_mode})")
    print("=" * 70)
    print(f"Config: rank={RANK}, te={TE}, lr={LR}, tlr={TLR}")
    print(f"  reinit_mode={reinit_mode}, decay_factor=1.0, units_in_mbatch=True")
    print(f"  target_modules={TARGET_MODULES}, lora_alpha={LORA_ALPHA}")
    print(f"Epochs: {NUM_EPOCHS}, batch_size={BATCH_SIZE}, seed={SEED}")
    if max_steps > 0:
        print(f"  ** QUICK TEST: stopping after {max_steps} steps **")
    print(f"Scheduler: cosine (warmup={WARMUP_STEPS}, min_lr_rate={MIN_LR_RATE})")
    print(f"Device: {DEVICE}")
    print()

    set_seed(SEED)

    # Load tokenizer and data
    print("Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_squad_data(tokenizer)
    n_batches = len(train_loader)
    # units_in_mbatch=True: te=193 means ~193 samples = ~6 batches per transfer
    expected_transfers_per_batch = 1.0 / (TE / BATCH_SIZE) if TE >= BATCH_SIZE else BATCH_SIZE / TE
    print(f"Train batches/epoch: {n_batches}")
    print(f"Eval features: {len(eval_features)}, Eval examples: {len(eval_examples)}")
    print(f"Expected ~{n_batches * BATCH_SIZE / TE:.0f} transfers per tile per epoch")
    print()

    # Create model
    print("Creating model...")
    model = create_squad_model(DEVICE, reinit_mode=reinit_mode)

    # Set up tracker and patch controllers
    tracker = TransferTracker()
    patched_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LRTTSimulatorTile):
            patch_controller(module.controller, tracker, layer_name=name)
            patched_count += 1
            print(f"  Patched: {name}")

    print(f"Total patched tiles: {patched_count}")
    print()

    # Optimizer & scheduler
    optimizer = AnalogAdam(model.parameters(), lr=LR)
    optimizer.regroup_param_groups()

    num_training_steps = n_batches * NUM_EPOCHS
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps,
        min_lr_rate=MIN_LR_RATE,
    )

    # Initial evaluation
    print("Initial evaluation (epoch 0)...")
    init_f1, init_em = evaluate_squad(model, eval_features, eval_examples, tokenizer, DEVICE)
    print(f"  F1={init_f1:.2f}, EM={init_em:.2f}")
    print()

    # Training loop
    epoch_metrics: list[dict] = []
    global_step = 0
    stop_early = False

    for epoch in range(1, NUM_EPOCHS + 1):
        if stop_early:
            break
        t0 = time()
        model.train()
        tracker.current_epoch = epoch
        total_loss = 0.0
        num_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            tracker.current_batch = batch_idx

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            start_positions = batch["start_positions"].to(DEVICE)
            end_positions = batch["end_positions"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_steps += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

            if max_steps > 0 and global_step >= max_steps:
                print(f"\n  Reached max_steps={max_steps}, stopping training.")
                stop_early = True
                break

        avg_loss = total_loss / num_steps if num_steps > 0 else 0.0

        # Evaluate
        f1, em = evaluate_squad(model, eval_features, eval_examples, tokenizer, DEVICE)
        elapsed = time() - t0

        # Transfer summary for this epoch
        n_transfers = sum(1 for r in tracker.records if r.epoch == epoch)
        summary = tracker.epoch_summary(epoch)
        mean_cosine = summary.get("cosine_sim", {}).get("mean", 0.0) if n_transfers > 0 else 0.0
        mean_signal = (
            summary.get("signal_ratio_changed", {}).get("mean", 0.0) if n_transfers > 0 else 0.0
        )
        mean_delta_ratio = (
            summary.get("delta_ratio", {}).get("mean", 0.0) if n_transfers > 0 else 0.0
        )
        mean_unchanged = (
            summary.get("unchanged_elem_ratio", {}).get("mean", 0.0) if n_transfers > 0 else 0.0
        )

        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS} | loss={avg_loss:.4f} F1={f1:.2f} EM={em:.2f} "
            f"| transfers={n_transfers} cos={mean_cosine:.3f} sig={mean_signal:.3f} "
            f"Δratio={mean_delta_ratio:.3f} unchanged={mean_unchanged:.3f} "
            f"| lr={current_lr:.6f} | {elapsed:.1f}s"
        )

        epoch_metrics.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "f1": f1,
                "em": em,
                "lr": current_lr,
                "n_transfers": n_transfers,
                "mean_cosine_sim": mean_cosine,
                "mean_signal_ratio": mean_signal,
                "mean_delta_ratio": mean_delta_ratio,
                "mean_unchanged_ratio": mean_unchanged,
            }
        )

    # --- Save outputs ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Per-transfer CSV
    transfer_csv = os.path.join(OUTPUT_DIR, f"transfer_diagnostic_{reinit_mode}_transfers.csv")
    fieldnames = [f.name for f in fields(TransferRecord)]
    with open(transfer_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in tracker.records:
            writer.writerow(rec.to_dict())

    # Per-epoch summary CSV
    epoch_csv = os.path.join(OUTPUT_DIR, f"transfer_diagnostic_{reinit_mode}_epochs.csv")
    metric_keys = [
        "a_norm",
        "b_norm",
        "ab_magnitude",
        "c_norm",
        "delta_c_norm",
        "delta_ratio",
        "unchanged_elem_ratio",
        "cosine_sim",
        "sign_agree_all",
        "sign_agree_changed",
        "signal_ratio_changed",
    ]
    epoch_fieldnames = ["epoch", "train_loss", "f1", "em", "lr", "n_transfers"]
    for key in metric_keys:
        for stat in ["mean", "min", "max"]:
            epoch_fieldnames.append(f"{key}_{stat}")

    with open(epoch_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_fieldnames)
        writer.writeheader()
        for idx, em_data in enumerate(epoch_metrics):
            epoch = em_data["epoch"]
            summary = tracker.epoch_summary(epoch)
            row: dict = {
                "epoch": epoch,
                "train_loss": em_data["train_loss"],
                "f1": em_data["f1"],
                "em": em_data["em"],
                "lr": em_data["lr"],
                "n_transfers": summary["n_transfers"],
            }
            for key in metric_keys:
                stats = summary.get(key, {})
                for stat in ["mean", "min", "max"]:
                    row[f"{key}_{stat}"] = stats.get(stat, "")
            writer.writerow(row)

    print()
    actual_epochs = len(epoch_metrics)
    print(
        f"Done. {len(tracker.records)} total transfers recorded across {actual_epochs} epochs ({global_step} steps)."
    )
    print(f"  Per-transfer CSV: {transfer_csv}")
    print(f"  Per-epoch CSV:    {epoch_csv}")
    print(f"  Final F1: {epoch_metrics[-1]['f1']:.2f}, EM: {epoch_metrics[-1]['em']:.2f}")


if __name__ == "__main__":
    main()
