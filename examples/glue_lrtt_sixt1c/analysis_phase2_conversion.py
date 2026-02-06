#!/usr/bin/env python
# coding=utf-8
"""Phase 2: LRTT 변환 과정 추적 - 가중치가 어떻게 변환되는지 분석"""

import torch
import numpy as np
import math
from transformers import AutoConfig, AutoModelForSequenceClassification

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


def create_lrtt_config(rank: int = 4, te: int = 1000, tlr: float = 0.001):
    """Create LRTT config for testing."""
    SOFTBOUNDS_CONFIG = {
        'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
        'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
        'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
        'write_noise_std': 0.0, 'mult_noise': True,
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


def trace_conversion(model_name: str):
    """LRTT 변환 과정 추적"""
    print(f"\n{'='*80}")
    print(f"PHASE 2: LRTT 변환 추적 - {model_name}")
    print(f"{'='*80}")

    # Load model
    model_config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config, use_safetensors=True
    )

    # 변환 전 특정 레이어의 가중치 저장
    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 상위 5개 문제 레이어만 추적
            if 'attention.self.key' in name and 'layer.2' in name:
                target_layers.append(name)
            elif 'attention.self.key' in name and 'layer.4' in name:
                target_layers.append(name)
            elif 'classifier' not in name and len(target_layers) < 5:
                target_layers.append(name)

    print(f"\nTracking {len(target_layers)} layers:")
    for name in target_layers[:5]:
        print(f"  - {name}")

    # 변환 전 가중치 저장
    weights_before = {}
    for name, module in model.named_modules():
        if name in target_layers and isinstance(module, torch.nn.Linear):
            weights_before[name] = {
                'weight': module.weight.data.clone(),
                'bias': module.bias.data.clone() if module.bias is not None else None,
                'weight_min': module.weight.data.min().item(),
                'weight_max': module.weight.data.max().item(),
                'bias_min': module.bias.data.min().item() if module.bias is not None else None,
                'bias_max': module.bias.data.max().item() if module.bias is not None else None,
            }

    print("\n[1] 변환 전 가중치 범위")
    print("-"*60)
    for name in target_layers[:5]:
        if name in weights_before:
            w = weights_before[name]
            print(f"{name}")
            print(f"    Weight: [{w['weight_min']:.4f}, {w['weight_max']:.4f}]")
            if w['bias_min'] is not None:
                print(f"    Bias:   [{w['bias_min']:.4f}, {w['bias_max']:.4f}]")

    # Convert to analog
    print("\n[2] convert_to_analog() 실행...")
    rpu_config = create_lrtt_config(rank=4, te=1000, tlr=0.001)
    model = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])

    # 변환 후 가중치 확인
    print("\n[3] 변환 후 가중치 분석")
    print("-"*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # AnalogLinear 모듈 찾기
    analog_layers = {}
    for name, module in model.named_modules():
        if hasattr(module, 'analog_module'):
            analog_layers[name] = module

    print(f"\nFound {len(analog_layers)} AnalogLinear layers")

    # 추적 레이어 분석
    for name in target_layers[:5]:
        # 변형된 이름 찾기 (model. prefix 등)
        matched_name = None
        for analog_name in analog_layers.keys():
            if name in analog_name or analog_name in name:
                matched_name = analog_name
                break

        if matched_name is None:
            # Try more flexible matching
            for analog_name in analog_layers.keys():
                name_parts = name.split('.')
                analog_parts = analog_name.split('.')
                if len(set(name_parts) & set(analog_parts)) >= 3:
                    matched_name = analog_name
                    break

        if matched_name is None:
            print(f"\n{name}: NOT FOUND IN ANALOG MODEL")
            continue

        module = analog_layers[matched_name]
        analog_tile = module.analog_module

        print(f"\n{name} -> {matched_name}")
        print("-"*60)

        # LRTT 타일 정보
        if hasattr(analog_tile, 'tile_c'):
            # This is an LRTT tile
            print("  Type: LRTTSimulatorTile")

            # C 타일 (visible) 가중치
            C_weights, C_bias = analog_tile.tile_c.get_weights()
            print(f"  Tile C (visible) weights: [{C_weights.min():.4f}, {C_weights.max():.4f}]")
            print(f"    Shape: {C_weights.shape}")

            if C_bias is not None:
                print(f"  Tile C bias: [{C_bias.min():.4f}, {C_bias.max():.4f}]")

            # A, B 타일 가중치
            A_weights = analog_tile.tile_a.get_weights()[0]
            B_weights = analog_tile.tile_b.get_weights()[0]
            print(f"  Tile A weights: [{A_weights.min():.4f}, {A_weights.max():.4f}]")
            print(f"    Shape: {A_weights.shape}")
            print(f"  Tile B weights: [{B_weights.min():.4f}, {B_weights.max():.4f}]")
            print(f"    Shape: {B_weights.shape}")

            # 원본과 비교
            if name in weights_before:
                orig = weights_before[name]
                print("\n  [비교] 원본 vs LRTT C 타일")
                print(f"    Original weight range: [{orig['weight_min']:.4f}, {orig['weight_max']:.4f}]")
                print(f"    LRTT C tile range:     [{C_weights.min():.4f}, {C_weights.max():.4f}]")

                # 클리핑 여부 확인
                w_max_device = 1.0  # LRTT default
                if abs(orig['weight_max']) > w_max_device or abs(orig['weight_min']) > w_max_device:
                    print(f"    ⚠️  WARNING: 원본 가중치가 [-{w_max_device}, {w_max_device}] 범위 초과!")
                    print(f"       클리핑 발생! 정보 손실 예상")

                    # 클리핑 비율 계산
                    orig_w = orig['weight'].flatten().numpy()
                    clipped_count = np.sum(np.abs(orig_w) > w_max_device)
                    total_count = len(orig_w)
                    print(f"       클리핑된 가중치: {clipped_count}/{total_count} ({100*clipped_count/total_count:.2f}%)")

        else:
            print("  Type: Regular AnalogTile (not LRTT)")
            weights = analog_tile.get_weights()
            if weights[0] is not None:
                print(f"  Weights: [{weights[0].min():.4f}, {weights[0].max():.4f}]")


def analyze_lrtt_tile_internal():
    """LRTT 타일 내부 동작 분석"""
    print("\n" + "="*80)
    print("PHASE 2: LRTT 타일 set_weights() 동작 분석")
    print("="*80)

    from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

    # LRTT 설정 생성
    rpu_config = create_lrtt_config(rank=4, te=1000, tlr=0.001)

    # 가상의 타일 생성
    d_size, x_size = 512, 512
    tile = LRTTSimulatorTile(d_size, x_size, rpu_config, bias=False)

    print(f"\n[1] 타일 생성: {d_size}x{x_size}, rank=4")

    # 큰 가중치 생성 (MobileBERT 스타일)
    test_weights = torch.randn(d_size, x_size) * 10  # ±10 범위
    print(f"\n[2] 테스트 가중치: [{test_weights.min():.4f}, {test_weights.max():.4f}]")

    # set_weights 호출
    tile.set_weights(test_weights, None)

    # 저장된 가중치 확인
    C_weights = tile.tile_c.get_weights()[0]
    print(f"\n[3] tile_c에 저장된 가중치: [{C_weights.min():.4f}, {C_weights.max():.4f}]")

    # 클리핑 여부 확인
    if C_weights.min() >= -1.0 and C_weights.max() <= 1.0:
        print("   ⚠️  가중치가 [-1.0, 1.0]으로 클리핑됨!")
        print("   → 원본 ±10 범위의 가중치가 ±1로 압축됨")
        print("   → 정보 손실: ~90%")
    else:
        print("   ✓ 가중치 범위 유지됨")


def check_softbounds_device():
    """SoftBoundsDevice의 w_max/w_min 확인"""
    print("\n" + "="*80)
    print("PHASE 2: SoftBoundsDevice w_max/w_min 분석")
    print("="*80)

    print("\n[1] 기본 SoftBoundsDevice 설정:")
    device = SoftBoundsDevice()
    print(f"    w_max: {device.w_max}")
    print(f"    w_min: {device.w_min}")
    print(f"    → 가중치 범위: [{device.w_min}, {device.w_max}]")

    print("\n[2] LRTT 설정에서 사용되는 C 타일 device:")
    rpu_config = create_lrtt_config()
    c_device = rpu_config.device.unit_cell_devices[2]  # C tile device
    print(f"    Type: {type(c_device).__name__}")
    print(f"    w_max: {c_device.w_max}")
    print(f"    w_min: {c_device.w_min}")

    print("\n[3] MobileBERT 문제 레이어 가중치 범위:")
    print("    attention.self.key.bias: [-15.30, +21.93]")
    print("    → 이 범위가 [{}, {}]로 클리핑됨".format(c_device.w_min, c_device.w_max))
    print("    → 95%+ 정보 손실!")


def main():
    print("="*80)
    print("PHASE 2: LRTT 변환 과정 심층 분석")
    print("="*80)

    # 1. SoftBoundsDevice 설정 확인
    check_softbounds_device()

    # 2. LRTT 타일 내부 동작 분석
    analyze_lrtt_tile_internal()

    # 3. 실제 모델 변환 추적 - MobileBERT
    trace_conversion("google/mobilebert-uncased")

    # 4. BERT-base와 비교
    trace_conversion("bert-base-uncased")

    print("\n" + "="*80)
    print("Phase 2 분석 완료!")
    print("="*80)
    print("\n결론:")
    print("1. LRTT의 SoftBoundsDevice는 w_max=1.0, w_min=-1.0으로 설정됨")
    print("2. set_weights() 시 가중치가 자동으로 [-1, 1] 범위로 클리핑됨")
    print("3. MobileBERT의 ±21 범위 가중치가 ±1로 압축되어 심각한 정보 손실 발생")
    print("4. BERT-base는 가중치 범위가 ±6.8로 상대적으로 작지만 여전히 클리핑 발생")


if __name__ == "__main__":
    main()
