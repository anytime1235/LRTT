#!/usr/bin/env python
# coding=utf-8
"""Phase 3: Forward Pass 추적 - logits 폭발의 정확한 원인 파악"""

import torch
import numpy as np
import math
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LRTT_SILENT"] = "1"


def create_lrtt_config(rank: int = 4, te: int = 1000, tlr: float = 0.001):
    """Create LRTT config for testing."""
    SOFTBOUNDS_CONFIG = {
        'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
        'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    }

    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
    )
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="decay",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = tlr
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    return PythonLRTTRPUConfig(device=device_config)


class ForwardHook:
    """레이어별 출력 캡처를 위한 hook"""
    def __init__(self, name):
        self.name = name
        self.output = None
        self.input = None

    def __call__(self, module, input, output):
        self.input = input
        if isinstance(output, tuple):
            self.output = output[0]
        else:
            self.output = output


def trace_forward_pass(model_name: str, use_lrtt: bool = True):
    """Forward pass를 추적하여 어디서 폭발하는지 확인"""
    print(f"\n{'='*80}")
    print(f"PHASE 3: Forward Pass 추적 - {model_name}")
    print(f"LRTT: {'ON' if use_lrtt else 'OFF'}")
    print(f"{'='*80}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model_config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config, use_safetensors=True
    )

    # Convert to LRTT if needed
    if use_lrtt:
        rpu_config = create_lrtt_config(rank=4, te=1000, tlr=0.001)
        model = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])

    model.to(device)
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 테스트 입력 생성
    text = "The movie was really great and I enjoyed it very much."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Hook 등록
    hooks = {}
    hook_handles = []

    # 추적할 레이어 패턴
    layer_patterns = [
        'embeddings',
        'encoder.layer.0.attention',
        'encoder.layer.0.output',
        'encoder.layer.2.attention',
        'encoder.layer.4.attention',
        'encoder.layer.6.attention',
        'encoder.layer.11.attention',
        'encoder.layer.23.attention' if 'mobilebert' in model_name else 'encoder.layer.11.output',
        'classifier',
    ]

    for name, module in model.named_modules():
        matched = False
        for pattern in layer_patterns:
            if pattern in name and (
                isinstance(module, torch.nn.Linear) or
                hasattr(module, 'analog_module') or
                'attention' in name and 'self' in name
            ):
                matched = True
                break

        if matched or 'classifier' in name:
            hook = ForwardHook(name)
            hooks[name] = hook
            handle = module.register_forward_hook(hook)
            hook_handles.append(handle)

    print(f"\n[1] 등록된 hooks: {len(hooks)}")

    # Forward pass
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
    except Exception as e:
        print(f"Forward pass failed: {e}")
        # 정리
        for handle in hook_handles:
            handle.remove()
        return

    print(f"\n[2] 최종 출력")
    print(f"    Logits: {logits.cpu().numpy()}")
    print(f"    Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

    # 폭발 여부 확인
    is_exploded = abs(logits).max().item() > 100

    print(f"\n[3] 레이어별 출력 분석")
    print("-"*80)
    print(f"{'Layer Name':<50} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-"*80)

    explosion_layer = None

    for name, hook in sorted(hooks.items(), key=lambda x: x[0]):
        if hook.output is not None:
            out = hook.output
            if isinstance(out, torch.Tensor):
                out_min = out.min().item()
                out_max = out.max().item()
                out_mean = out.mean().item()
                out_std = out.std().item()

                # 폭발 표시
                marker = ""
                if abs(out_max) > 1000 or abs(out_min) > 1000:
                    marker = " ⚠️ EXPLOSION!"
                    if explosion_layer is None:
                        explosion_layer = name

                print(f"{name[:50]:<50} {out_min:>12.4f} {out_max:>12.4f} {out_mean:>12.4f} {out_std:>12.4f}{marker}")

    # 정리
    for handle in hook_handles:
        handle.remove()

    if explosion_layer:
        print(f"\n[4] 폭발 시작 레이어: {explosion_layer}")
    elif is_exploded:
        print(f"\n[4] 최종 출력에서 폭발 발생 (중간 레이어에서는 감지 안됨)")
    else:
        print(f"\n[4] 정상 (폭발 없음)")

    return logits, explosion_layer


def compare_layer_outputs(model_name: str):
    """LRTT 적용 전후 같은 레이어의 출력 비교"""
    print(f"\n{'='*80}")
    print(f"PHASE 3: LRTT 전후 레이어 출력 비교 - {model_name}")
    print(f"{'='*80}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Original model
    print("\n[1] Original Model (No LRTT)")
    model_orig = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, use_safetensors=True
    )
    model_orig.to(device)
    model_orig.eval()

    # 2. LRTT model
    print("\n[2] LRTT Model")
    model_lrtt = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, use_safetensors=True
    )
    rpu_config = create_lrtt_config(rank=4, te=1000, tlr=0.001)
    model_lrtt = convert_to_analog(model_lrtt, rpu_config, exclude_modules=["classifier"])
    model_lrtt.to(device)
    model_lrtt.eval()

    # Tokenizer & inputs
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "The movie was really great."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        out_orig = model_orig(**inputs)
        out_lrtt = model_lrtt(**inputs)

    print("\n[3] 결과 비교")
    print("-"*60)
    print(f"Original logits: {out_orig.logits.cpu().numpy()}")
    print(f"LRTT logits:     {out_lrtt.logits.cpu().numpy()}")
    print(f"\nOriginal range:  [{out_orig.logits.min():.4f}, {out_orig.logits.max():.4f}]")
    print(f"LRTT range:      [{out_lrtt.logits.min():.4f}, {out_lrtt.logits.max():.4f}]")

    diff = (out_lrtt.logits - out_orig.logits).abs().max().item()
    print(f"\n최대 차이: {diff:.4f}")

    return out_orig.logits, out_lrtt.logits


def analyze_attention_scores(model_name: str, use_lrtt: bool = True):
    """Attention score 분석 - softmax 전후 값 확인"""
    print(f"\n{'='*80}")
    print(f"PHASE 3: Attention Score 분석 - {model_name}")
    print(f"{'='*80}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_config = AutoConfig.from_pretrained(model_name, num_labels=2, output_attentions=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config, use_safetensors=True
    )

    if use_lrtt:
        rpu_config = create_lrtt_config(rank=4, te=1000, tlr=0.001)
        model = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "Hello world"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward with attention outputs
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    print(f"\n[1] Attention 정보")
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        print(f"    Attention layers: {len(outputs.attentions)}")
        for i, attn in enumerate(outputs.attentions[:3]):  # First 3 layers
            print(f"    Layer {i}: shape={attn.shape}, range=[{attn.min():.4f}, {attn.max():.4f}]")
    else:
        print("    Attention outputs not available")

    print(f"\n[2] Logits")
    print(f"    {outputs.logits.cpu().numpy()}")


def main():
    print("="*80)
    print("PHASE 3: Forward Pass 심층 분석")
    print("="*80)

    # 1. MobileBERT without LRTT (baseline)
    print("\n" + "="*80)
    print("Test 1: MobileBERT WITHOUT LRTT (Baseline)")
    print("="*80)
    trace_forward_pass("google/mobilebert-uncased", use_lrtt=False)

    # 2. MobileBERT with LRTT (problem case)
    print("\n" + "="*80)
    print("Test 2: MobileBERT WITH LRTT (Problem Case)")
    print("="*80)
    trace_forward_pass("google/mobilebert-uncased", use_lrtt=True)

    # 3. BERT-base with LRTT (should work)
    print("\n" + "="*80)
    print("Test 3: BERT-base WITH LRTT")
    print("="*80)
    trace_forward_pass("bert-base-uncased", use_lrtt=True)

    # 4. 비교 분석
    print("\n" + "="*80)
    print("Test 4: MobileBERT 전후 비교")
    print("="*80)
    compare_layer_outputs("google/mobilebert-uncased")

    print("\n" + "="*80)
    print("Phase 3 분석 완료!")
    print("="*80)


if __name__ == "__main__":
    main()
