#!/usr/bin/env python
# coding=utf-8
"""
SQuAD NaN 문제 원인 추적

3가지 방식을 10 step씩 테스트하여 비교:
1. Original lora_on_analog_hardware (TorchInferenceRPU, QKV only)
2. LRTT-LoRA fp_lora mode (FloatingPoint device)
3. LRTT-LoRA sixt1c_lora mode (6T1C device) - 현재 실패

목표: 어느 시점에서 loss=0, grad_norm=nan이 발생하는지 정확히 파악
"""

import sys
import os
import torch
import subprocess
from datetime import datetime

# Results storage
results = {
    "original_lora_on_analog": {},
    "lrtt_fp_lora": {},
    "lrtt_sixt1c_lora": {}
}

TEST_STEPS = 10  # 10 steps만 테스트
PYTHON = "/data/venvs/aihwkit_gpu/bin/python"

print("=" * 80)
print("SQuAD NaN 문제 원인 추적 진단")
print("=" * 80)
print(f"각 방식별로 {TEST_STEPS} steps 테스트")
print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

def parse_training_log(log_file):
    """로그 파일에서 loss와 grad_norm 추출"""
    losses = []
    grad_norms = []

    if not os.path.exists(log_file):
        return losses, grad_norms

    with open(log_file, 'r') as f:
        for line in f:
            if "'loss':" in line and "'grad_norm':" in line:
                try:
                    # Extract loss
                    loss_start = line.find("'loss':") + 8
                    loss_end = line.find(",", loss_start)
                    loss_str = line[loss_start:loss_end].strip()
                    loss = float(loss_str)

                    # Extract grad_norm
                    grad_start = line.find("'grad_norm':") + 13
                    grad_end = line.find(",", grad_start)
                    grad_str = line[grad_start:grad_end].strip()
                    grad_norm = float(grad_str) if grad_str.lower() != 'nan' else float('nan')

                    losses.append(loss)
                    grad_norms.append(grad_norm)
                except:
                    pass

    return losses, grad_norms

def print_results(name, losses, grad_norms):
    """결과 출력"""
    print(f"\n{name}:")
    print("-" * 60)

    if not losses:
        print("  ❌ No training logs found")
        return

    for i, (loss, grad) in enumerate(zip(losses[:TEST_STEPS], grad_norms[:TEST_STEPS]), 1):
        status = "✅" if loss > 0 and not (loss != loss or grad != grad) else "❌"
        print(f"  Step {i:2d}: loss={loss:12.4e}, grad_norm={grad:12.4f} {status}")

    # 분석
    has_nan = any(l != l or g != g for l, g in zip(losses[:TEST_STEPS], grad_norms[:TEST_STEPS]))
    has_zero_loss = any(l == 0.0 for l in losses[:TEST_STEPS])

    if has_nan:
        first_nan = next(i for i, (l, g) in enumerate(zip(losses, grad_norms), 1) if l != l or g != g)
        print(f"\n  ⚠️  NaN 발생: Step {first_nan}")
    if has_zero_loss:
        first_zero = next(i for i, l in enumerate(losses, 1) if l == 0.0)
        print(f"  ⚠️  Zero loss: Step {first_zero}")

    if not has_nan and not has_zero_loss:
        print(f"\n  ✅ 정상: 모든 step에서 학습 진행")

# =============================================================================
# Test 1: Original lora_on_analog_hardware
# =============================================================================
print("\n" + "=" * 80)
print("Test 1: Original lora_on_analog_hardware (QKV only)")
print("=" * 80)
print("TorchInferenceRPU + Weight Clipping")
print("Target modules: query, key, value (no dense)")
print()

# Create modified run_glue.py that:
# 1. Uses QKV only (no dense)
# 2. Runs for 10 steps only
# 3. Uses SQuAD-like settings

test1_script = """
import sys
sys.path.insert(0, '/data/lora_on_analog_hardware/lora_training_glue')

# Import and modify original code
# This is complex, so let's use a simple wrapper approach
print("Test 1 would run original lora_on_analog_hardware here")
print("Skipping for now - implementation needed")
"""

# For now, skip Test 1 as it requires significant modification
print("⏭️  Test 1 skipped (requires modification of original code)")
print("   Focus on LRTT-LoRA tests first")

# =============================================================================
# Test 2: LRTT-LoRA FP mode (FloatingPoint device)
# =============================================================================
print("\n" + "=" * 80)
print("Test 2: LRTT-LoRA FP mode")
print("=" * 80)
print("FloatingPoint device (exact arithmetic)")
print("Target modules: query, key, value")
print()

log_file_fp = "/tmp/test_fp_lora_10steps.log"

# Modify sweep script to run 10 steps only
cmd_fp = [
    PYTHON,
    "/data/LRTT_transformer/experiments/sweep_lrtt_lora_optuna.py",
    "--task", "squad",
    "--mode", "fp_lora",
    "--rank", "8",
    "--target_modules", "query", "key", "value",
    "--n_trials", "1",
    "--study_name", "test_fp_10steps"
]

print(f"Running: {' '.join(cmd_fp)}")
print(f"Log: {log_file_fp}")
print("Progress: ", end="", flush=True)

# Run with environment variable to limit steps
env = os.environ.copy()
env["WANDB_MODE"] = "offline"
# Note: We need to modify the script to support max_steps, or let it run and kill after 10 steps

with open(log_file_fp, 'w') as log_f:
    process = subprocess.Popen(
        cmd_fp,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env
    )

    # Monitor for 10 steps (approximately 5-10 minutes for SQuAD)
    import time
    start_time = time.time()
    timeout = 600  # 10 minutes max

    while process.poll() is None:
        elapsed = time.time() - start_time

        # Check if we have 10 steps
        losses, grads = parse_training_log(log_file_fp)
        if len(losses) >= TEST_STEPS:
            print(f"\n✓ Got {len(losses)} steps, stopping...")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
            break

        if elapsed > timeout:
            print(f"\n⏱️  Timeout after {timeout}s, stopping...")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
            break

        time.sleep(10)
        print(".", end="", flush=True)

losses_fp, grads_fp = parse_training_log(log_file_fp)
print_results("FP-LoRA", losses_fp, grads_fp)

# =============================================================================
# Test 3: LRTT-LoRA 6T1C mode (현재 실패하는 모드)
# =============================================================================
print("\n" + "=" * 80)
print("Test 3: LRTT-LoRA 6T1C mode (현재 실패)")
print("=" * 80)
print("6T1C LinearStepDevice")
print("Target modules: query, key, value")
print("learn_out_scaling: False (방금 변경)")
print()

log_file_6t1c = "/tmp/test_6t1c_lora_10steps.log"

cmd_6t1c = [
    PYTHON,
    "/data/LRTT_transformer/experiments/sweep_lrtt_lora_optuna.py",
    "--task", "squad",
    "--mode", "sixt1c_lora",
    "--rank", "8",
    "--target_modules", "query", "key", "value",
    "--n_trials", "1",
    "--study_name", "test_6t1c_10steps"
]

print(f"Running: {' '.join(cmd_6t1c)}")
print(f"Log: {log_file_6t1c}")
print("Progress: ", end="", flush=True)

with open(log_file_6t1c, 'w') as log_f:
    process = subprocess.Popen(
        cmd_6t1c,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env
    )

    start_time = time.time()

    while process.poll() is None:
        elapsed = time.time() - start_time

        losses, grads = parse_training_log(log_file_6t1c)
        if len(losses) >= TEST_STEPS:
            print(f"\n✓ Got {len(losses)} steps, stopping...")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
            break

        if elapsed > timeout:
            print(f"\n⏱️  Timeout after {timeout}s, stopping...")
            process.terminate()
            time.sleep(2)
            if process.poll() is None:
                process.kill()
            break

        time.sleep(10)
        print(".", end="", flush=True)

losses_6t1c, grads_6t1c = parse_training_log(log_file_6t1c)
print_results("6T1C-LoRA", losses_6t1c, grads_6t1c)

# =============================================================================
# 비교 분석
# =============================================================================
print("\n" + "=" * 80)
print("비교 분석")
print("=" * 80)

print("\n결과 요약:")
print(f"  FP-LoRA:   {len(losses_fp)} steps logged")
print(f"  6T1C-LoRA: {len(losses_6t1c)} steps logged")

if len(losses_fp) >= 3 and len(losses_6t1c) >= 3:
    print("\n처음 3 steps 비교:")
    print(f"{'Mode':<15} {'Step 1':>15} {'Step 2':>15} {'Step 3':>15}")
    print("-" * 60)
    print(f"{'FP-LoRA':<15} {losses_fp[0]:>15.2e} {losses_fp[1]:>15.2e} {losses_fp[2]:>15.2e}")
    print(f"{'6T1C-LoRA':<15} {losses_6t1c[0]:>15.2e} {losses_6t1c[1]:>15.2e} {losses_6t1c[2]:>15.2e}")

print("\n결론:")
if len(losses_fp) >= TEST_STEPS and all(l > 0 for l in losses_fp[:TEST_STEPS]):
    if len(losses_6t1c) < TEST_STEPS or any(l == 0 or l != l for l in losses_6t1c[:TEST_STEPS]):
        print("  ✅ FP-LoRA는 정상 작동")
        print("  ❌ 6T1C-LoRA만 실패")
        print("  → 문제는 6T1C LinearStepDevice 또는 관련 설정에 있음")
    else:
        print("  ✅ 두 모드 모두 정상 작동")
        print("  → 문제가 해결되었거나 다른 조건에서 발생")
else:
    if len(losses_6t1c) >= TEST_STEPS and all(l > 0 for l in losses_6t1c[:TEST_STEPS]):
        print("  ❌ FP-LoRA 실패")
        print("  ✅ 6T1C-LoRA 정상")
        print("  → 매우 이상한 상황 (재테스트 필요)")
    else:
        print("  ❌ 두 모드 모두 실패")
        print("  → 문제는 LRTT 아키텍처 자체 또는 다른 공통 요인")

print("\n" + "=" * 80)
print(f"진단 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print(f"\n상세 로그:")
print(f"  FP-LoRA:   {log_file_fp}")
print(f"  6T1C-LoRA: {log_file_6t1c}")
