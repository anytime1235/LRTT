#!/usr/bin/env python
# coding=utf-8
"""Phase 5: 해결책 검증 실험 - MobileBERT LRTT 문제 해결 방안 테스트"""

import torch
import numpy as np
import math
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from copy import deepcopy

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice, ConstantStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LRTT_SILENT"] = "1"


def create_lrtt_config_custom_range(w_max: float = 1.0):
    """Create LRTT config with custom w_max/w_min range."""
    SOFTBOUNDS_CONFIG = {
        'dw_min': 0.001,
        'w_max': w_max,
        'w_min': -w_max,
        'dw_min_dtod': 0.0,
        'dw_min_std': 0.0,
        'up_down': 0.0,
    }

    ab_device = LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=w_max,
        w_min=-w_max,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
    )
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    device_config = PythonLRTTDevice(
        rank=4,
        transfer_every=1000,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="decay",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.001
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    return PythonLRTTRPUConfig(device=device_config)


def normalize_model_weights(model, target_max: float = 1.0):
    """Normalize all model weights to fit within target range.

    Args:
        model: PyTorch model to normalize
        target_max: Target maximum absolute weight value

    Returns:
        Normalized model and scaling factors per layer
    """
    scaling_factors = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name or 'bias' in name:
                max_abs = param.data.abs().max().item()
                if max_abs > target_max:
                    scale = target_max / max_abs
                    param.data.mul_(scale)
                    scaling_factors[name] = scale
                else:
                    scaling_factors[name] = 1.0

    return model, scaling_factors


def test_forward_pass(model, tokenizer, device, test_name: str):
    """Test forward pass and return logits."""
    text = "The movie was really great and I enjoyed it very much."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        try:
            outputs = model(**inputs)
            logits = outputs.logits

            is_ok = logits.abs().max().item() < 100
            status = "✓ OK" if is_ok else "✗ FAILED"

            print(f"{test_name}")
            print(f"    Logits: {logits.cpu().numpy()}")
            print(f"    Range:  [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            print(f"    Status: {status}")

            return logits, is_ok
        except Exception as e:
            print(f"{test_name}")
            print(f"    Error: {e}")
            return None, False


def experiment_1_normalize_weights():
    """실험 1: MobileBERT 가중치를 ±1로 정규화 후 LRTT 적용"""
    print("\n" + "="*80)
    print("실험 1: 가중치 정규화 (±1 범위로 스케일링)")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "google/mobilebert-uncased"

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Check original weights
    print("\n[1] 원본 가중치 범위:")
    for name, param in model.named_parameters():
        if 'attention.self.key.bias' in name and 'layer.2.' in name:
            print(f"    {name}: [{param.min():.4f}, {param.max():.4f}]")
            break

    # Normalize weights
    print("\n[2] 가중치 정규화 적용...")
    model, scaling_factors = normalize_model_weights(model, target_max=1.0)

    # Check normalized weights
    print("\n[3] 정규화 후 가중치 범위:")
    for name, param in model.named_parameters():
        if 'attention.self.key.bias' in name and 'layer.2.' in name:
            print(f"    {name}: [{param.min():.4f}, {param.max():.4f}]")
            print(f"    Scaling factor: {scaling_factors.get(name, 1.0):.4f}")
            break

    # Test before LRTT
    print("\n[4] LRTT 변환 전 테스트:")
    model.to(device)
    test_forward_pass(model, tokenizer, device, "    Normalized model (no LRTT)")

    # Apply LRTT
    print("\n[5] LRTT 변환 후 테스트:")
    rpu_config = create_lrtt_config_custom_range(w_max=1.0)
    model_lrtt = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])
    model_lrtt.to(device)

    logits, is_ok = test_forward_pass(model_lrtt, tokenizer, device, "    Normalized + LRTT")

    return is_ok


def experiment_2_expand_wmax():
    """실험 2: LRTT의 w_max/w_min을 ±25로 확장"""
    print("\n" + "="*80)
    print("실험 2: w_max/w_min 확장 (±25)")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "google/mobilebert-uncased"

    # Load model (without normalization)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Check original weights
    print("\n[1] 원본 가중치 범위 (수정 없음):")
    for name, param in model.named_parameters():
        if 'attention.self.key.bias' in name and 'layer.2.' in name:
            print(f"    {name}: [{param.min():.4f}, {param.max():.4f}]")
            break

    # Apply LRTT with expanded range
    print("\n[2] LRTT 변환 (w_max=25):")
    rpu_config = create_lrtt_config_custom_range(w_max=25.0)
    model_lrtt = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])
    model_lrtt.to(device)

    logits, is_ok = test_forward_pass(model_lrtt, tokenizer, device, "    MobileBERT + LRTT (w_max=25)")

    return is_ok


def experiment_3_exclude_problem_layers():
    """실험 3: 문제 레이어만 LRTT에서 제외"""
    print("\n" + "="*80)
    print("실험 3: 문제 레이어 제외")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "google/mobilebert-uncased"

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Problem layers to exclude (attention.self.key with large biases)
    exclude_modules = ["classifier"]

    # Find all attention.self.key layers and layers with large weights
    for name, param in model.named_parameters():
        if 'attention.self.key' in name:
            layer_name = name.rsplit('.', 1)[0]  # Remove 'weight' or 'bias'
            if layer_name not in exclude_modules:
                exclude_modules.append(layer_name)

    print(f"\n[1] 제외할 레이어 ({len(exclude_modules)}개):")
    for name in exclude_modules[:10]:
        print(f"    - {name}")
    if len(exclude_modules) > 10:
        print(f"    ... and {len(exclude_modules) - 10} more")

    # Apply LRTT with exclusions
    print("\n[2] LRTT 변환 (문제 레이어 제외):")
    rpu_config = create_lrtt_config_custom_range(w_max=1.0)
    model_lrtt = convert_to_analog(model, rpu_config, exclude_modules=exclude_modules)
    model_lrtt.to(device)

    logits, is_ok = test_forward_pass(model_lrtt, tokenizer, device, "    MobileBERT + LRTT (key layers excluded)")

    return is_ok


def experiment_4_bert_with_inflated_weights():
    """실험 4: BERT-base 가중치를 ±20으로 인위적 확대 후 LRTT"""
    print("\n" + "="*80)
    print("실험 4: BERT-base 가중치 인위적 확대 (MobileBERT 문제 재현)")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-uncased"

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Check original weights
    print("\n[1] 원본 BERT-base 가중치 범위:")
    for name, param in model.named_parameters():
        if 'attention.self.key.bias' in name and 'layer.2.' in name:
            print(f"    {name}: [{param.min():.4f}, {param.max():.4f}]")
            break

    # Inflate weights like MobileBERT
    print("\n[2] 가중치 인위적 확대 (x10):")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'attention.self.key.bias' in name:
                param.data.mul_(10.0)  # Multiply by 10 to simulate MobileBERT range

    for name, param in model.named_parameters():
        if 'attention.self.key.bias' in name and 'layer.2.' in name:
            print(f"    {name}: [{param.min():.4f}, {param.max():.4f}]")
            break

    # Apply LRTT
    print("\n[3] LRTT 변환:")
    rpu_config = create_lrtt_config_custom_range(w_max=1.0)
    model_lrtt = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])
    model_lrtt.to(device)

    logits, is_ok = test_forward_pass(model_lrtt, tokenizer, device, "    BERT-base (inflated) + LRTT")

    return is_ok


def experiment_5_layer_specific_scaling():
    """실험 5: 레이어별 개별 스케일링 (attention bias만 정규화)"""
    print("\n" + "="*80)
    print("실험 5: 레이어별 개별 스케일링")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "google/mobilebert-uncased"

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Only normalize attention.self.key biases (the main problem)
    print("\n[1] attention.self.key.bias만 정규화:")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'attention.self.key.bias' in name:
                max_abs = param.data.abs().max().item()
                if max_abs > 1.0:
                    scale = 1.0 / max_abs
                    param.data.mul_(scale)
                    print(f"    {name}: scaled by {scale:.4f}")

    # Apply LRTT
    print("\n[2] LRTT 변환:")
    rpu_config = create_lrtt_config_custom_range(w_max=1.0)
    model_lrtt = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])
    model_lrtt.to(device)

    logits, is_ok = test_forward_pass(model_lrtt, tokenizer, device, "    MobileBERT (key bias normalized) + LRTT")

    return is_ok


def main():
    print("="*80)
    print("PHASE 5: MobileBERT LRTT 해결책 검증 실험")
    print("="*80)

    results = {}

    # Experiment 1: Weight normalization
    try:
        results['exp1_normalize'] = experiment_1_normalize_weights()
    except Exception as e:
        print(f"Experiment 1 failed: {e}")
        results['exp1_normalize'] = False

    # Experiment 2: Expand w_max
    try:
        results['exp2_expand_wmax'] = experiment_2_expand_wmax()
    except Exception as e:
        print(f"Experiment 2 failed: {e}")
        results['exp2_expand_wmax'] = False

    # Experiment 3: Exclude problem layers
    try:
        results['exp3_exclude_layers'] = experiment_3_exclude_problem_layers()
    except Exception as e:
        print(f"Experiment 3 failed: {e}")
        results['exp3_exclude_layers'] = False

    # Experiment 4: Inflate BERT-base weights
    try:
        results['exp4_bert_inflated'] = experiment_4_bert_with_inflated_weights()
    except Exception as e:
        print(f"Experiment 4 failed: {e}")
        results['exp4_bert_inflated'] = False

    # Experiment 5: Layer-specific scaling
    try:
        results['exp5_layer_scaling'] = experiment_5_layer_specific_scaling()
    except Exception as e:
        print(f"Experiment 5 failed: {e}")
        results['exp5_layer_scaling'] = False

    # Summary
    print("\n" + "="*80)
    print("실험 결과 요약")
    print("="*80)
    print()

    exp_names = {
        'exp1_normalize': '가중치 전체 정규화 (±1)',
        'exp2_expand_wmax': 'w_max/w_min 확장 (±25)',
        'exp3_exclude_layers': '문제 레이어 제외',
        'exp4_bert_inflated': 'BERT 가중치 확대 (문제 재현)',
        'exp5_layer_scaling': 'key bias만 정규화',
    }

    for key, name in exp_names.items():
        status = "✓ 성공" if results.get(key, False) else "✗ 실패"
        print(f"{name:<40} {status}")

    print("\n" + "="*80)
    print("결론")
    print("="*80)

    successful = [k for k, v in results.items() if v]
    if successful:
        print("\n효과적인 해결책:")
        for key in successful:
            print(f"  - {exp_names[key]}")
    else:
        print("\n모든 실험이 실패했습니다.")

    print("\n권장 사항:")
    if results.get('exp2_expand_wmax'):
        print("  1. LRTT 설정에서 w_max/w_min을 ±25로 확장 (가장 간단)")
    if results.get('exp1_normalize'):
        print("  2. 모델 가중치를 ±1 범위로 정규화 후 LRTT 적용")
    if results.get('exp3_exclude_layers'):
        print("  3. 문제가 되는 attention.self.key 레이어를 LRTT에서 제외")
    if results.get('exp5_layer_scaling'):
        print("  4. attention.self.key.bias만 선택적으로 정규화")

    print()


if __name__ == "__main__":
    main()
