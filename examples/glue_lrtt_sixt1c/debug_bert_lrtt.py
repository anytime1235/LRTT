#!/usr/bin/env python
# coding=utf-8
"""BERT-base LRTT 학습 문제 심층 분석 스크립트.

분석 항목:
1. 모델 변환 검증: convert_to_analog가 올바르게 적용되는지
2. Forward pass 검증: 출력이 의미있는지
3. Gradient flow 검증: gradient가 제대로 전파되는지
4. Weight 변화 검증: 학습 중 weight가 업데이트되는지
5. Loss 변화 검증: loss가 감소하는지
"""

import os
import sys
import torch
import numpy as np

# Use installed aihwkit from venv
# LRTT src path is handled by the environment

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.nn.modules.linear import AnalogLinear

import warnings
warnings.filterwarnings("ignore")

MODEL_NAME = "bert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def create_simple_lrtt_config(rank=4, transfer_every=100):
    """간단한 LRTT config 생성 (idealized)."""
    from aihwkit.simulator.configs import IdealDevice

    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="standard",
        unit_cell_devices=[IdealDevice(), IdealDevice(), IdealDevice()],
    )
    device_config.transfer_lr = 1.0
    device_config.forward_inject = False
    device_config.update_mode = "lora"

    return PythonLRTTRPUConfig(device=device_config)


def analyze_model_conversion():
    """1. 모델 변환 분석."""
    print_section("1. 모델 변환 분석")

    # 원본 모델 로드
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    print(f"원본 모델 레이어 수: {sum(1 for _ in model.modules())}")

    # Linear 레이어 카운트
    linear_layers = [(name, module) for name, module in model.named_modules()
                     if isinstance(module, torch.nn.Linear)]
    print(f"Linear 레이어 수: {len(linear_layers)}")

    for name, layer in linear_layers[:5]:
        print(f"  - {name}: {layer.in_features} -> {layer.out_features}")
    print("  ...")

    # LRTT 변환
    rpu_config = create_simple_lrtt_config(rank=4, transfer_every=100)
    model_analog = convert_to_analog(model, rpu_config, exclude_modules=["classifier"], verbose=True)

    # AnalogLinear 레이어 카운트
    analog_layers = [(name, module) for name, module in model_analog.named_modules()
                     if isinstance(module, AnalogLinear)]
    print(f"\nAnalogLinear 레이어 수: {len(analog_layers)}")

    # LRTT 타일 확인
    from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
    lrtt_tiles = []
    for name, module in model_analog.named_modules():
        if hasattr(module, 'analog_tile') and isinstance(module.analog_tile, LRTTSimulatorTile):
            lrtt_tiles.append((name, module.analog_tile))

    print(f"LRTT 타일 수: {len(lrtt_tiles)}")

    if lrtt_tiles:
        name, tile = lrtt_tiles[0]
        print(f"\n첫 번째 LRTT 타일 ({name}):")
        print(f"  - d_size (out): {tile.d_size}")
        print(f"  - x_size (in): {tile.x_size}")
        print(f"  - rank: {tile.rank}")
        print(f"  - transfer_every: {tile.transfer_every}")
        print(f"  - lora_alpha: {tile.lora_alpha}")
        print(f"  - forward_inject: {tile.controller.forward_inject_enabled}")

        # A, B, C 타일 weight 확인
        A_w = tile.tile_a.get_weights()[0]
        B_w = tile.tile_b.get_weights()[0]
        C_w = tile.tile_c.get_weights()[0]
        print(f"\n  A 타일: shape={A_w.shape}, mean={A_w.mean():.6f}, std={A_w.std():.6f}")
        print(f"  B 타일: shape={B_w.shape}, mean={B_w.mean():.6f}, std={B_w.std():.6f}")
        print(f"  C 타일: shape={C_w.shape}, mean={C_w.mean():.6f}, std={C_w.std():.6f}")

    return model_analog, lrtt_tiles


def analyze_forward_pass(model_analog):
    """2. Forward pass 분석."""
    print_section("2. Forward Pass 분석")

    model_analog.to(DEVICE)
    model_analog.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 테스트 입력
    texts = [
        "This movie is great!",
        "This movie is terrible!",
        "I love this product.",
        "I hate this product.",
    ]

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_analog(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    print("입력 텍스트와 예측 확률:")
    for i, text in enumerate(texts):
        print(f"  '{text[:30]}...' -> P(pos)={probs[i, 1]:.4f}, P(neg)={probs[i, 0]:.4f}")

    print(f"\nLogits 통계:")
    print(f"  mean: {logits.mean():.6f}")
    print(f"  std: {logits.std():.6f}")
    print(f"  min: {logits.min():.6f}")
    print(f"  max: {logits.max():.6f}")

    # 예측이 의미있는지 확인
    predictions = logits.argmax(dim=-1)
    print(f"\n예측: {predictions.tolist()}")
    print(f"모든 예측이 동일한가? {len(set(predictions.tolist())) == 1}")

    return logits


def analyze_gradient_flow(model_analog):
    """3. Gradient flow 분석."""
    print_section("3. Gradient Flow 분석")

    model_analog.to(DEVICE)
    model_analog.train()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 테스트 입력
    texts = ["This is a test sentence.", "Another test sentence."]
    labels = torch.tensor([1, 0]).to(DEVICE)

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Forward pass
    outputs = model_analog(**inputs, labels=labels)
    loss = outputs.loss

    print(f"Loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Gradient 확인
    print("\nGradient 분석:")

    grad_stats = []
    for name, param in model_analog.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_stats.append((name, grad_norm, param.grad.mean().item(), param.grad.std().item()))

    # 상위 10개 gradient 출력
    grad_stats.sort(key=lambda x: x[1], reverse=True)
    print("\n가장 큰 gradient를 가진 파라미터 (상위 10개):")
    for name, norm, mean, std in grad_stats[:10]:
        print(f"  {name[:50]}: norm={norm:.6f}, mean={mean:.8f}, std={std:.8f}")

    # Gradient가 0인 파라미터 확인
    zero_grads = [name for name, param in model_analog.named_parameters()
                  if param.grad is not None and param.grad.abs().max() < 1e-10]
    print(f"\nGradient가 거의 0인 파라미터 수: {len(zero_grads)}")

    # LRTT 관련 파라미터 gradient 확인
    print("\nLRTT 관련 gradient 확인:")
    for name, param in model_analog.named_parameters():
        if 'tile' in name.lower() or 'analog' in name.lower():
            if param.grad is not None:
                print(f"  {name}: grad_norm={param.grad.norm().item():.8f}")

    return loss


def analyze_weight_updates(model_analog, lrtt_tiles):
    """4. Weight 업데이트 분석."""
    print_section("4. Weight 업데이트 분석")

    model_analog.to(DEVICE)
    model_analog.train()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Optimizer 설정
    optimizer = AnalogSGD(model_analog.parameters(), lr=0.01)

    # 첫 번째 LRTT 타일의 초기 weight 저장
    if lrtt_tiles:
        name, tile = lrtt_tiles[0]
        A_before = tile.tile_a.get_weights()[0].clone()
        B_before = tile.tile_b.get_weights()[0].clone()
        C_before = tile.tile_c.get_weights()[0].clone()
        print(f"초기 A weight norm: {A_before.norm():.6f}")
        print(f"초기 B weight norm: {B_before.norm():.6f}")
        print(f"초기 C weight norm: {C_before.norm():.6f}")

    # 몇 스텝 학습
    print("\n10 스텝 학습 진행...")
    losses = []
    for step in range(10):
        texts = [f"Test sentence {step} positive", f"Test sentence {step} negative"]
        labels = torch.tensor([1, 0]).to(DEVICE)

        inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        optimizer.zero_grad()
        outputs = model_analog(**inputs, labels=labels)
        loss = outputs.loss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if step % 3 == 0:
            print(f"  Step {step}: loss = {loss.item():.6f}")

    print(f"\nLoss 변화: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print(f"Loss 감소율: {(losses[0] - losses[-1]) / losses[0] * 100:.2f}%")

    # Weight 변화 확인
    if lrtt_tiles:
        name, tile = lrtt_tiles[0]
        A_after = tile.tile_a.get_weights()[0]
        B_after = tile.tile_b.get_weights()[0]
        C_after = tile.tile_c.get_weights()[0]

        A_diff = (A_after - A_before).norm()
        B_diff = (B_after - B_before).norm()
        C_diff = (C_after - C_before).norm()

        print(f"\nWeight 변화량:")
        print(f"  A 타일: {A_diff:.8f} (상대: {A_diff / A_before.norm() * 100:.4f}%)")
        print(f"  B 타일: {B_diff:.8f} (상대: {B_diff / B_before.norm() * 100:.4f}%)")
        print(f"  C 타일: {C_diff:.8f} (상대: {C_diff / C_before.norm() * 100:.4f}%)")

        # Transfer 상태 확인
        print(f"\n컨트롤러 상태:")
        print(f"  step_count: {tile.controller.step_count}")
        print(f"  transfer_count: {tile.controller.transfer_count}")

    return losses


def analyze_lrtt_mechanics(lrtt_tiles):
    """5. LRTT 메커니즘 분석."""
    print_section("5. LRTT 메커니즘 상세 분석")

    if not lrtt_tiles:
        print("LRTT 타일이 없습니다!")
        return

    name, tile = lrtt_tiles[0]
    controller = tile.controller

    print(f"LRTT 컨트롤러 설정:")
    print(f"  forward_inject_enabled: {controller.forward_inject_enabled}")
    print(f"  update_mode: {controller.update_mode}")
    print(f"  transfer_method: {controller.transfer_method}")
    print(f"  transfer_mode: {controller.transfer_mode}")
    print(f"  transfer_lr: {controller.transfer_lr}")
    print(f"  lora_alpha: {controller.lora_alpha}")
    print(f"  reinit_mode: {controller.reinit_mode}")
    print(f"  reinit_gain: {controller.reinit_gain}")

    # Forward inject = False일 때의 문제점 분석
    if not controller.forward_inject_enabled:
        print("\n⚠️  forward_inject = False 설정됨!")
        print("   이 경우 forward pass에서 y = C @ x 만 계산됩니다.")
        print("   A, B 타일은 gradient 축적용으로만 사용됩니다.")
        print("   → C 타일이 제대로 업데이트되지 않으면 학습이 안 됩니다!")

    # C 타일 업데이트 방식 확인
    print(f"\nC 타일 업데이트 방식:")
    print(f"  transfer_every: {tile.transfer_every} steps")
    print(f"  현재 step: {controller.step_count}")
    print(f"  다음 transfer까지: {tile.transfer_every - (controller.step_count % tile.transfer_every)} steps")


def compare_with_baseline():
    """6. Digital baseline과 비교."""
    print_section("6. Digital Baseline 비교")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Digital 모델
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model_digital = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    model_digital.to(DEVICE)
    model_digital.train()

    # LRTT 모델
    model_analog = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    rpu_config = create_simple_lrtt_config(rank=4, transfer_every=10)  # 빠른 transfer
    model_analog = convert_to_analog(model_analog, rpu_config, exclude_modules=["classifier"])
    model_analog.to(DEVICE)
    model_analog.train()

    # Optimizer
    optimizer_digital = torch.optim.AdamW(model_digital.parameters(), lr=2e-5)
    optimizer_analog = AnalogSGD(model_analog.parameters(), lr=0.01)

    # 동일한 데이터로 학습
    texts = ["This is great!", "This is bad!", "Amazing product", "Terrible experience"]
    labels = torch.tensor([1, 0, 1, 0]).to(DEVICE)

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    print("20 스텝 학습 비교...")
    digital_losses = []
    analog_losses = []

    for step in range(20):
        # Digital
        optimizer_digital.zero_grad()
        outputs_d = model_digital(**inputs, labels=labels)
        loss_d = outputs_d.loss
        digital_losses.append(loss_d.item())
        loss_d.backward()
        optimizer_digital.step()

        # Analog
        optimizer_analog.zero_grad()
        outputs_a = model_analog(**inputs, labels=labels)
        loss_a = outputs_a.loss
        analog_losses.append(loss_a.item())
        loss_a.backward()
        optimizer_analog.step()

        if step % 5 == 0:
            print(f"  Step {step}: Digital={loss_d.item():.4f}, Analog={loss_a.item():.4f}")

    print(f"\n최종 비교:")
    print(f"  Digital: {digital_losses[0]:.4f} -> {digital_losses[-1]:.4f} (감소: {(digital_losses[0]-digital_losses[-1])/digital_losses[0]*100:.1f}%)")
    print(f"  Analog:  {analog_losses[0]:.4f} -> {analog_losses[-1]:.4f} (감소: {(analog_losses[0]-analog_losses[-1])/analog_losses[0]*100:.1f}%)")


def analyze_sweep_config():
    """7. Sweep에서 사용된 config 분석."""
    print_section("7. Sweep Config 분석")

    # sweep_bert_base_optuna.py에서 사용된 설정
    print("sweep_bert_base_optuna.py 설정:")
    print("  RANKS = [1, 4, 8, 16, 32, 64]")
    print("  TRANSFER_EVERYS = [1, 10, 50, 100, 500, 1000, 2000, 5000]")
    print("  LIFETIMES = [100, 1000, 10000, 46505, 100000]")
    print("  LR range: [1e-4, 1.0]")
    print("  TLR range: [1e-4, 10.0]")

    print("\n완료된 Trial 결과:")
    print("  Trial 0: rank=4, te=1000, lr=5e-5, tlr=8.26e-4 -> 49.66%")
    print("  Trial 1: rank=32, te=5000, lr=5e-5, tlr=3.33e-3 -> 50.57%")
    print("  Trial 2: rank=4, te=50, lr=5e-5, tlr=9.55e-4 -> 48.97%")

    print("\n⚠️  문제점 분석:")
    print("  1. forward_inject=False: A,B가 forward에 기여 안 함")
    print("  2. transfer_every가 너무 큼: 1000~5000 steps")
    print("     - SST-2는 67349 samples / 32 batch = 2105 steps/epoch")
    print("     - 3 epochs = 6315 steps 동안 transfer가 1~6회만 발생")
    print("  3. Learning rate가 매우 작음: 5e-5 (AnalogSGD용으로는 작을 수 있음)")


def main():
    print("=" * 70)
    print(" BERT-base LRTT Fine-tuning 문제 심층 분석")
    print("=" * 70)

    # 1. 모델 변환 분석
    model_analog, lrtt_tiles = analyze_model_conversion()

    # 2. Forward pass 분석
    analyze_forward_pass(model_analog)

    # 3. Gradient flow 분석
    analyze_gradient_flow(model_analog)

    # 4. Weight 업데이트 분석
    analyze_weight_updates(model_analog, lrtt_tiles)

    # 5. LRTT 메커니즘 분석
    analyze_lrtt_mechanics(lrtt_tiles)

    # 6. Baseline 비교
    compare_with_baseline()

    # 7. Sweep config 분석
    analyze_sweep_config()

    print_section("결론 및 권장사항")
    print("""
주요 문제점:
1. forward_inject=False로 인해 A,B가 forward에 기여하지 않음
2. transfer_every가 너무 커서 C 타일 업데이트가 거의 없음
3. Learning rate 설정이 LRTT에 최적화되지 않음

권장 수정사항:
1. forward_inject=True 또는 transfer_every를 매우 작게 (1~10)
2. transfer_lr을 높게 설정 (0.1~1.0)
3. 또는 idealized config 사용하여 먼저 검증
4. BERT의 경우 Layer-wise LR decay 적용 고려
""")


if __name__ == "__main__":
    main()
