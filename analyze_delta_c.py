#!/home/jovyan/work/ml/.venv310/bin/python
# coding=utf-8
"""TikiTaka vs Digital Merge Delta C Analysis.

This script measures delta C (C tile / Slow tile change) during transfer
for both Digital Merge and TikiTaka v2 to find equivalent conditions.

Key Hypothesis:
- TikiTaka (transfer_lr=1.362): accuracy 70.5%
- Digital Merge (transfer_lr=0.00531): accuracy 51.6%
- transfer_lr 차이로 인해 delta C 크기가 다름

Analysis Method:
1. Digital Merge: delta_C = transfer_lr * (A @ B)
2. TikiTaka: delta_Slow = transfer_lr * read_from_Fast (+ auto_scale, chopper effects)

Expected Result:
- Digital Merge의 ||A @ B||가 작아서 transfer_lr을 크게 해야 TikiTaka와 동등
- 예상 동등 조건: transfer_lr ≈ 100~1000 (기존 대비 100~1000배)
"""

import os
import sys
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
# AdamW doesn't trigger tile updates, must use AnalogAdam for LRTT tiles

# aihwkit imports (from system first, then override with LRTT src)
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs import (
    UnitCellRPUConfig,
    IOParameters,
    UpdateParameters,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import (
    FloatingPointDevice,
    SoftBoundsDevice,
    LinearStepDevice,
    SoftBoundsReferenceDevice,
)

# LRTT-specific imports (override with local src)
sys.path.insert(0, '/data/LRTT_transformer/src')
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DeltaCConfig:
    """Configuration for delta C analysis."""
    # Model settings
    model_name: str = "google/mobilebert-uncased"
    task_name: str = "sst2"
    max_seq_length: int = 128
    batch_size: int = 16  # Reduced for memory
    seed: int = 42

    # Training settings
    num_steps: int = 250  # Steps to run for analysis
    eval_every: int = 50  # Evaluate accuracy every N steps

    # Digital Merge optimal settings (Trial #42)
    dm_lr: float = 0.000659
    dm_transfer_lr: float = 0.00531
    dm_transfer_every: int = 63
    dm_rank: int = 8
    dm_lora_alpha: float = 1.0

    # TikiTaka v2 optimal settings (Trial #48)
    tt_lr: float = 3.710e-04
    tt_transfer_lr: float = 1.362
    tt_transfer_every: int = 100
    tt_fast_lr: float = 0.926
    tt_auto_granularity: float = 107.09
    tt_in_chop_prob: float = 0.071

    # Target modules
    target_modules: List[str] = field(default_factory=lambda: ["query"])

    # Output
    output_dir: str = "/data"


# =============================================================================
# Utility Functions
# =============================================================================

def list_linear_layers(model: nn.Module) -> List[str]:
    """List all linear layer names in the model."""
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(name)
    return linear_layers


def compute_frobenius_norm(tensor: torch.Tensor) -> float:
    """Compute Frobenius norm."""
    return torch.norm(tensor, p='fro').item()


# =============================================================================
# Digital Merge Delta C Measurement
# =============================================================================

class DigitalMergeDeltaMeasurer:
    """Measure delta C for Digital Merge (LoRA-style A/B/C tiles)."""

    def __init__(self, config: DeltaCConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Storage for measurements
        self.measurements: List[Dict] = []
        self.ab_norms: List[Dict] = []

    def create_config(
        self,
        transfer_lr: float,
    ) -> PythonLRTTRPUConfig:
        """Create LRTT config with Digital A,B + Analog C."""
        ab_device = FloatingPointDevice()
        c_device = SoftBoundsDevice(
            dw_min=0.001, w_max=1.0, w_min=-1.0,
            dw_min_dtod=0.0, dw_min_std=0.0, write_noise_std=0.0, mult_noise=True,
        )

        device_config = PythonLRTTDevice(
            rank=self.config.dm_rank,
            transfer_every=self.config.dm_transfer_every,
            lora_alpha=self.config.dm_lora_alpha,
            reinit_gain=0.1,
            reinit_mode="decay",
            decay_factor=1.0,
            unit_cell_devices=[ab_device, ab_device, c_device],
        )
        device_config.transfer_lr = transfer_lr
        device_config.forward_inject = True
        device_config.transfer_method = "onehot"
        device_config.update_mode = "lora"
        device_config.a_init_mode = "zero"

        return PythonLRTTRPUConfig(device=device_config)

    def create_model(self, transfer_lr: float) -> nn.Module:
        """Create model with Digital Merge configuration."""
        model_config = AutoConfig.from_pretrained(self.config.model_name, num_labels=2)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, config=model_config
        )

        all_linear = list_linear_layers(model)
        exclude = [name for name in all_linear
                   if not any(t in name for t in self.config.target_modules)]
        exclude.append("classifier")

        rpu_config = self.create_config(transfer_lr)
        model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

        for name, param in model.named_parameters():
            is_target = any(t in name for t in self.config.target_modules)
            if is_target or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.to(self.device)
        return model

    def get_tile_weights(self, model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract A, B, C weights from all LRTT layers."""
        weights = {}
        for name, module in model.named_modules():
            # AnalogLinear stores tile in 'analog_module' (not 'analog_tile')
            if hasattr(module, 'analog_module'):
                tile = module.analog_module
                if hasattr(tile, 'tile_a') and hasattr(tile, 'tile_b') and hasattr(tile, 'tile_c'):
                    A = tile.tile_a.get_weights()[0].clone()
                    B = tile.tile_b.get_weights()[0].clone()
                    C = tile.tile_c.get_weights()[0].clone()
                    weights[name] = {'A': A, 'B': B, 'C': C}
        return weights

    def measure_ab_norms(self, model: nn.Module, step: int, transfer_lr: float) -> Dict:
        """Measure ||A||, ||B||, ||A @ B|| for all layers."""
        weights = self.get_tile_weights(model)

        layer_norms = {}
        for layer_name, w in weights.items():
            A = w['A']
            B = w['B']
            AB = A @ B

            layer_norms[layer_name] = {
                'A_norm': compute_frobenius_norm(A),
                'B_norm': compute_frobenius_norm(B),
                'AB_norm': compute_frobenius_norm(AB),
                'delta_C_norm': transfer_lr * compute_frobenius_norm(AB),
            }

        # Aggregate across layers
        total_AB_norm = sum(ln['AB_norm'] for ln in layer_norms.values())
        total_delta_C_norm = sum(ln['delta_C_norm'] for ln in layer_norms.values())

        return {
            'step': step,
            'transfer_lr': transfer_lr,
            'total_AB_norm': total_AB_norm,
            'total_delta_C_norm': total_delta_C_norm,
            'layer_norms': layer_norms,
        }

    def run_training_with_measurement(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        transfer_lr: float,
    ) -> Dict:
        """Run training and measure delta C at each transfer."""
        print(f"\n{'='*60}")
        print(f"Digital Merge Delta C Measurement")
        print(f"transfer_lr = {transfer_lr}")
        print(f"{'='*60}")

        set_seed(self.config.seed)
        model = self.create_model(transfer_lr)
        optimizer = AnalogAdam(model.parameters(), lr=self.config.dm_lr)
        criterion = nn.CrossEntropyLoss()

        # Debug: verify layer detection
        test_weights = self.get_tile_weights(model)
        print(f"Found {len(test_weights)} LRTT layers: {list(test_weights.keys())}")
        if test_weights:
            sample_layer = list(test_weights.keys())[0]
            print(f"Sample layer '{sample_layer}' shapes: A={test_weights[sample_layer]['A'].shape}, "
                  f"B={test_weights[sample_layer]['B'].shape}, C={test_weights[sample_layer]['C'].shape}")

        measurements = []
        accuracies = []

        step = 0
        model.train()

        pbar = tqdm(total=self.config.num_steps, desc=f"Training (transfer_lr={transfer_lr})")

        for batch in train_loader:
            if step >= self.config.num_steps:
                break

            # Get weights before step
            weights_before = self.get_tile_weights(model)

            # Training step
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            # Reset update flags for LRTT tiles (important for next step)
            for _, module in model.named_modules():
                if hasattr(module, 'analog_module') and hasattr(module.analog_module, '_reset_update_flag'):
                    module.analog_module._reset_update_flag()

            step += 1
            pbar.update(1)

            # Check if transfer occurred (every transfer_every steps)
            if step % self.config.dm_transfer_every == 0:
                weights_after = self.get_tile_weights(model)

                # Measure delta C
                for layer_name in weights_before:
                    C_before = weights_before[layer_name]['C']
                    C_after = weights_after[layer_name]['C']
                    A = weights_before[layer_name]['A']
                    B = weights_before[layer_name]['B']

                    delta_C = C_after - C_before
                    AB = A @ B

                    measurement = {
                        'step': step,
                        'layer': layer_name,
                        'transfer_lr': transfer_lr,
                        'A_norm': compute_frobenius_norm(A),
                        'B_norm': compute_frobenius_norm(B),
                        'AB_norm': compute_frobenius_norm(AB),
                        'C_before_norm': compute_frobenius_norm(C_before),
                        'C_after_norm': compute_frobenius_norm(C_after),
                        'delta_C_norm': compute_frobenius_norm(delta_C),
                        'expected_delta_C_norm': transfer_lr * compute_frobenius_norm(AB),
                    }
                    measurements.append(measurement)

            # Evaluate periodically
            if step % self.config.eval_every == 0:
                acc = self.evaluate(model, eval_loader)
                accuracies.append({'step': step, 'accuracy': acc, 'transfer_lr': transfer_lr})
                pbar.set_postfix(acc=f"{acc:.4f}")

        pbar.close()

        # Final evaluation
        final_acc = self.evaluate(model, eval_loader)
        accuracies.append({'step': step, 'accuracy': final_acc, 'transfer_lr': transfer_lr})

        # Cleanup model to free GPU memory
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'measurements': measurements,
            'accuracies': accuracies,
            'final_accuracy': final_acc,
            'transfer_lr': transfer_lr,
        }

    def evaluate(self, model: nn.Module, eval_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        model.train()
        return correct / total if total > 0 else 0.0


# =============================================================================
# TikiTaka v2 Delta C Measurement
# =============================================================================

class TikiTakaDeltaMeasurer:
    """Measure delta C (Slow tile change) for TikiTaka v2."""

    def __init__(self, config: DeltaCConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_config(self) -> UnitCellRPUConfig:
        """Create TikiTaka v2 config with 6T1C Fast + SoftBoundsReference Slow."""
        sixt1c_device = LinearStepDevice(
            dw_min=0.001981,
            gamma_up=-0.1678,
            gamma_down=0.1410,
            dw_min_dtod=0.1,
            up_down_dtod=0.01,
            w_max_dtod=0.05,
            w_min_dtod=0.05,
            gamma_up_dtod=0.05,
            gamma_down_dtod=0.05,
            dw_min_std=0.3,
            write_noise_std=0.0,
            mult_noise=True,
            mean_bound_reference=True,
            lifetime=0.0,
        )

        softbounds_device = SoftBoundsReferenceDevice(
            dw_min=0.001,
            w_max=1.0,
            w_min=-1.0,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            write_noise_std=0.0,
            mult_noise=True,
        )

        rpu_config = UnitCellRPUConfig(
            device=ChoppedTransferCompound(
                unit_cell_devices=[sixt1c_device, softbounds_device],
                transfer_every=self.config.tt_transfer_every,
                units_in_mbatch=False,
                n_reads_per_transfer=1,
                transfer_columns=True,
                gamma=0.0,
                transfer_lr=self.config.tt_transfer_lr,
                fast_lr=self.config.tt_fast_lr,
                scale_transfer_lr=True,
                auto_scale=True,
                auto_granularity=self.config.tt_auto_granularity,
                buffer_granularity=1.0,
                auto_momentum=0.99,
                in_chop_prob=self.config.tt_in_chop_prob,
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

        return rpu_config

    def create_model(self) -> nn.Module:
        """Create model with TikiTaka v2 configuration."""
        model_config = AutoConfig.from_pretrained(self.config.model_name, num_labels=2)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, config=model_config
        )

        all_linear = list_linear_layers(model)
        exclude = [name for name in all_linear
                   if not any(t in name for t in self.config.target_modules)]
        exclude.append("classifier")

        rpu_config = self.create_config()
        model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

        for name, param in model.named_parameters():
            is_target = any(t in name for t in self.config.target_modules)
            if is_target or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.to(self.device)
        return model

    def get_tile_weights(self, model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract Fast and Slow tile weights from all TikiTaka layers."""
        weights = {}
        for name, module in model.named_modules():
            # AnalogLinear stores tile in 'analog_module' (not 'analog_tile')
            if hasattr(module, 'analog_module'):
                tile = module.analog_module
                # ChoppedTransferCompound has two device states: [0] = Fast, [1] = Slow
                if hasattr(tile, 'get_hidden_parameters'):
                    try:
                        # Get weights for each device in the compound
                        hidden = tile.get_hidden_parameters()
                        combined_weight = tile.get_weights()[0].clone()  # Combined weight

                        # For 2-device compound: device 0 = Fast, device 1 = Slow
                        # The visible weight is the Slow tile when gamma=0
                        weights[name] = {
                            'combined': combined_weight,
                            'transfer_lr': self.config.tt_transfer_lr,
                        }
                    except Exception as e:
                        print(f"Warning: Could not get hidden params for {name}: {e}")
                elif hasattr(tile, 'get_weights'):
                    # Fallback: just get visible weights
                    weights[name] = {
                        'combined': tile.get_weights()[0].clone(),
                        'transfer_lr': self.config.tt_transfer_lr,
                    }
        return weights

    def run_training_with_measurement(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
    ) -> Dict:
        """Run training and measure delta Slow at each transfer."""
        print(f"\n{'='*60}")
        print(f"TikiTaka v2 Delta C Measurement")
        print(f"transfer_lr = {self.config.tt_transfer_lr}")
        print(f"transfer_every = {self.config.tt_transfer_every}")
        print(f"{'='*60}")

        set_seed(self.config.seed)
        model = self.create_model()
        optimizer = AnalogAdam(model.parameters(), lr=self.config.tt_lr)
        criterion = nn.CrossEntropyLoss()

        # Debug: verify layer detection
        test_weights = self.get_tile_weights(model)
        print(f"Found {len(test_weights)} TikiTaka layers: {list(test_weights.keys())}")
        if test_weights:
            sample_layer = list(test_weights.keys())[0]
            print(f"Sample layer '{sample_layer}' combined shape: {test_weights[sample_layer]['combined'].shape}")

        measurements = []
        accuracies = []

        step = 0
        model.train()

        pbar = tqdm(total=self.config.num_steps, desc="TikiTaka v2 Training")

        for batch in train_loader:
            if step >= self.config.num_steps:
                break

            # Get weights before step
            weights_before = self.get_tile_weights(model)

            # Training step
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            step += 1
            pbar.update(1)

            # Check if transfer occurred (every transfer_every steps)
            if step % self.config.tt_transfer_every == 0:
                weights_after = self.get_tile_weights(model)

                # Measure delta for Slow tile
                for layer_name in weights_before:
                    W_before = weights_before[layer_name]['combined']
                    W_after = weights_after[layer_name]['combined']

                    delta_W = W_after - W_before

                    measurement = {
                        'step': step,
                        'layer': layer_name,
                        'transfer_lr': self.config.tt_transfer_lr,
                        'W_before_norm': compute_frobenius_norm(W_before),
                        'W_after_norm': compute_frobenius_norm(W_after),
                        'delta_W_norm': compute_frobenius_norm(delta_W),
                    }
                    measurements.append(measurement)

            # Evaluate periodically
            if step % self.config.eval_every == 0:
                acc = self.evaluate(model, eval_loader)
                accuracies.append({'step': step, 'accuracy': acc})
                pbar.set_postfix(acc=f"{acc:.4f}")

        pbar.close()

        # Final evaluation
        final_acc = self.evaluate(model, eval_loader)
        accuracies.append({'step': step, 'accuracy': final_acc})

        return {
            'measurements': measurements,
            'accuracies': accuracies,
            'final_accuracy': final_acc,
            'transfer_lr': self.config.tt_transfer_lr,
        }

    def evaluate(self, model: nn.Module, eval_loader: DataLoader) -> float:
        """Evaluate model accuracy."""
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        model.train()
        return correct / total if total > 0 else 0.0


# =============================================================================
# Equivalent Condition Finder
# =============================================================================

def find_equivalent_transfer_lr(
    dm_measurements: Dict,
    tt_measurements: Dict,
) -> Dict:
    """Find transfer_lr for Digital Merge that produces equivalent delta C to TikiTaka."""

    # Average delta C norms
    dm_delta_c_norms = [m['delta_C_norm'] for m in dm_measurements['measurements']]
    tt_delta_w_norms = [m['delta_W_norm'] for m in tt_measurements['measurements']]

    dm_avg_delta_c = np.mean(dm_delta_c_norms) if dm_delta_c_norms else 0
    tt_avg_delta_w = np.mean(tt_delta_w_norms) if tt_delta_w_norms else 0

    # Calculate AB norm (from dm measurements)
    dm_ab_norms = [m['AB_norm'] for m in dm_measurements['measurements']]
    dm_avg_ab = np.mean(dm_ab_norms) if dm_ab_norms else 0

    # Find equivalent transfer_lr
    # ||delta_C_dm|| = transfer_lr_dm * ||A @ B||
    # ||delta_W_tt|| = tt_transfer_lr * (effective contribution from Fast tile)
    # We want: transfer_lr_dm * ||A @ B|| = ||delta_W_tt||
    # So: transfer_lr_dm = ||delta_W_tt|| / ||A @ B||

    dm_current_transfer_lr = dm_measurements['transfer_lr']

    if dm_avg_ab > 1e-10:
        equivalent_transfer_lr = tt_avg_delta_w / dm_avg_ab
        scale_factor = equivalent_transfer_lr / dm_current_transfer_lr
    else:
        equivalent_transfer_lr = float('nan')
        scale_factor = float('nan')

    return {
        'dm_avg_delta_c': dm_avg_delta_c,
        'dm_avg_ab_norm': dm_avg_ab,
        'dm_current_transfer_lr': dm_current_transfer_lr,
        'tt_avg_delta_w': tt_avg_delta_w,
        'tt_transfer_lr': tt_measurements['transfer_lr'],
        'equivalent_transfer_lr': equivalent_transfer_lr,
        'scale_factor': scale_factor,
        'ratio_tt_to_dm': tt_avg_delta_w / dm_avg_delta_c if dm_avg_delta_c > 1e-10 else float('nan'),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("TikiTaka vs Digital Merge Delta C Analysis")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)

    config = DeltaCConfig()

    # Load data
    print("\nLoading data...")
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    raw_datasets = load_dataset("nyu-mll/glue", config.task_name)

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=config.max_seq_length,
            truncation=True
        )

    tokenized = raw_datasets.map(preprocess, batched=True)
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=default_data_collator
    )
    eval_loader = DataLoader(
        tokenized["validation"],
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=default_data_collator
    )

    print(f"Train: {len(tokenized['train'])}, Eval: {len(tokenized['validation'])}")

    # ==========================================================================
    # Step 1: Digital Merge Measurement with multiple transfer_lr values
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Digital Merge ||A @ B|| Measurement")
    print("=" * 80)

    dm_measurer = DigitalMergeDeltaMeasurer(config)

    # Test with multiple transfer_lr values
    # Note: transfer_lr > 1.0 causes CUDA/OOM issues
    transfer_lr_values = [
        config.dm_transfer_lr,  # Original: 0.00531
        0.1,
        0.5,
    ]

    dm_results = {}
    for transfer_lr in transfer_lr_values:
        print(f"\n--- Digital Merge with transfer_lr = {transfer_lr} ---")

        # Clear GPU memory before each run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        try:
            result = dm_measurer.run_training_with_measurement(
                train_loader, eval_loader, transfer_lr
            )
            dm_results[transfer_lr] = result
            print(f"Final accuracy: {result['final_accuracy']:.4f}")
        except (RuntimeError, Exception) as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg or "cublas" in error_msg:
                print(f"CUDA/OOM error at transfer_lr={transfer_lr}, skipping...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                continue
            else:
                raise

    # Save Digital Merge results
    dm_output = {
        'timestamp': timestamp,
        'config': asdict(config),
        'results': {}
    }
    for tlr, result in dm_results.items():
        # Convert numpy types for JSON serialization
        serializable_result = {
            'transfer_lr': tlr,
            'final_accuracy': result['final_accuracy'],
            'measurements': result['measurements'],
            'accuracies': result['accuracies'],
        }
        dm_output['results'][str(tlr)] = serializable_result

    dm_output_path = os.path.join(config.output_dir, "delta_c_analysis_digital_merge.json")
    with open(dm_output_path, 'w') as f:
        json.dump(dm_output, f, indent=2, default=str)
    print(f"\nDigital Merge results saved to: {dm_output_path}")

    # Clear GPU memory before TikiTaka
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==========================================================================
    # Step 2: TikiTaka v2 Measurement
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 2: TikiTaka v2 delta_Slow Measurement")
    print("=" * 80)

    tt_measurer = TikiTakaDeltaMeasurer(config)
    tt_result = tt_measurer.run_training_with_measurement(train_loader, eval_loader)
    print(f"\nTikiTaka v2 final accuracy: {tt_result['final_accuracy']:.4f}")

    # Save TikiTaka results
    tt_output = {
        'timestamp': timestamp,
        'config': asdict(config),
        'result': {
            'transfer_lr': tt_result['transfer_lr'],
            'final_accuracy': tt_result['final_accuracy'],
            'measurements': tt_result['measurements'],
            'accuracies': tt_result['accuracies'],
        }
    }

    tt_output_path = os.path.join(config.output_dir, "delta_c_analysis_tikitaka.json")
    with open(tt_output_path, 'w') as f:
        json.dump(tt_output, f, indent=2, default=str)
    print(f"TikiTaka v2 results saved to: {tt_output_path}")

    # ==========================================================================
    # Step 3: Find Equivalent Conditions
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Equivalent Condition Analysis")
    print("=" * 80)

    # Compare with original transfer_lr
    original_dm = dm_results[config.dm_transfer_lr]
    equiv_analysis = find_equivalent_transfer_lr(original_dm, tt_result)

    print("\n--- Analysis Results ---")
    print(f"Digital Merge (transfer_lr={config.dm_transfer_lr}):")
    print(f"  Average ||A @ B||: {equiv_analysis['dm_avg_ab_norm']:.6f}")
    print(f"  Average ||delta_C||: {equiv_analysis['dm_avg_delta_c']:.6f}")
    print(f"  Final accuracy: {original_dm['final_accuracy']:.4f}")

    print(f"\nTikiTaka v2 (transfer_lr={config.tt_transfer_lr}):")
    print(f"  Average ||delta_Slow||: {equiv_analysis['tt_avg_delta_w']:.6f}")
    print(f"  Final accuracy: {tt_result['final_accuracy']:.4f}")

    print(f"\n--- Equivalent Condition ---")
    print(f"  Ratio (TikiTaka/DigitalMerge): {equiv_analysis['ratio_tt_to_dm']:.2f}x")
    print(f"  Equivalent transfer_lr for DM: {equiv_analysis['equivalent_transfer_lr']:.4f}")
    print(f"  Scale factor needed: {equiv_analysis['scale_factor']:.2f}x")

    # Check if any of the tested transfer_lr values matched
    print("\n--- Accuracy Comparison at Different transfer_lr ---")
    print(f"{'transfer_lr':>12} | {'Final Accuracy':>14} | {'Avg ||delta_C||':>15}")
    print("-" * 50)
    for tlr in transfer_lr_values:
        result = dm_results[tlr]
        dm_delta_norms = [m['delta_C_norm'] for m in result['measurements']]
        avg_delta = np.mean(dm_delta_norms) if dm_delta_norms else 0
        print(f"{tlr:>12.5f} | {result['final_accuracy']:>14.4f} | {avg_delta:>15.6f}")

    print(f"\nTikiTaka v2 comparison:")
    print(f"{config.tt_transfer_lr:>12.5f} | {tt_result['final_accuracy']:>14.4f} | {equiv_analysis['tt_avg_delta_w']:>15.6f}")

    # Save equivalent conditions
    equiv_output = {
        'timestamp': timestamp,
        'digital_merge': {
            'transfer_lr': config.dm_transfer_lr,
            'avg_ab_norm': equiv_analysis['dm_avg_ab_norm'],
            'avg_delta_c': equiv_analysis['dm_avg_delta_c'],
            'final_accuracy': original_dm['final_accuracy'],
        },
        'tikitaka_v2': {
            'transfer_lr': config.tt_transfer_lr,
            'avg_delta_w': equiv_analysis['tt_avg_delta_w'],
            'final_accuracy': tt_result['final_accuracy'],
        },
        'equivalent_condition': {
            'equivalent_transfer_lr': equiv_analysis['equivalent_transfer_lr'],
            'scale_factor': equiv_analysis['scale_factor'],
            'ratio_tt_to_dm': equiv_analysis['ratio_tt_to_dm'],
        },
        'all_dm_results': {
            str(tlr): {
                'final_accuracy': dm_results[tlr]['final_accuracy'],
                'avg_delta_c': np.mean([m['delta_C_norm'] for m in dm_results[tlr]['measurements']])
                               if dm_results[tlr]['measurements'] else 0,
            }
            for tlr in transfer_lr_values
        }
    }

    equiv_output_path = os.path.join(config.output_dir, "delta_c_equivalent_conditions.json")
    with open(equiv_output_path, 'w') as f:
        json.dump(equiv_output, f, indent=2, default=str)
    print(f"\nEquivalent conditions saved to: {equiv_output_path}")

    print("\n" + "=" * 80)
    print("Delta C Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
