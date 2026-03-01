"""
실험 시작 전 최종 설정 검증
"""

import sys
import torch

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import importlib.util
spec = importlib.util.spec_from_file_location(
    "sweep_frozen",
    "/data/LRTT_transformer/LRTT_glue/sweep_sixt1c_lora_glue_frozen_classifier.py"
)
sweep_frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep_frozen)


def verify_all_settings():
    """모든 설정 최종 확인"""

    print("="*80)
    print("  실험 시작 전 최종 설정 검증")
    print("="*80)

    issues = []

    # 1. LR 범위 확인
    print("\n[1] Learning Rate 범위")
    print("-"*80)
    lr_min = sweep_frozen.LR_MIN
    lr_max = sweep_frozen.LR_MAX
    print(f"  LR_MIN: {lr_min}")
    print(f"  LR_MAX: {lr_max}")

    if lr_min == 2e-4 and lr_max == 2e-2:
        print(f"  ✓ 올바름: [2e-4, 2e-2]")
    else:
        print(f"  ✗ 잘못됨! 예상: [2e-4, 2e-2], 실제: [{lr_min}, {lr_max}]")
        issues.append("LR 범위가 잘못됨")

    # 2. SEED 확인
    print("\n[2] Seed 설정")
    print("-"*80)
    seed = sweep_frozen.SEED
    print(f"  SEED: {seed}")

    if seed == 42:
        print(f"  ✓ 올바름: 42")
    else:
        print(f"  ✗ 잘못됨! 예상: 42, 실제: {seed}")
        issues.append("SEED가 42가 아님")

    # 3. 모델 생성하여 설정 확인
    print("\n[3] 모델 설정 확인")
    print("-"*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 첫 번째 모델
    print("\n  첫 번째 모델 생성...")
    model1 = sweep_frozen.create_glue_model(
        "sst2", device, ["query", "key", "value"],
        fp_lora=False, lora_alpha=1.0
    )

    # Classifier 확인
    classifier_trainable = False
    classifier_weight1 = None
    for name, param in model1.named_parameters():
        if 'classifier' in name:
            if param.requires_grad:
                classifier_trainable = True
            if name == 'base_model.model.classifier.weight':
                classifier_weight1 = param.clone().detach().cpu()

    print(f"\n  Classifier trainable: {classifier_trainable}")
    if not classifier_trainable:
        print(f"  ✓ Classifier FROZEN")
    else:
        print(f"  ✗ Classifier TRAINABLE (잘못됨!)")
        issues.append("Classifier가 trainable임")

    # LoRA 확인
    lora_count = 0
    for name, param in model1.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_count += 1

    print(f"\n  Trainable LoRA params: {lora_count}")
    if lora_count > 0:
        print(f"  ✓ LoRA trainable")
    else:
        print(f"  ✗ LoRA가 trainable이 아님!")
        issues.append("LoRA가 trainable이 아님")

    # Embeddings 확인
    embedding_trainable = False
    for name, param in model1.named_parameters():
        if 'embedding' in name.lower() and param.requires_grad:
            embedding_trainable = True

    print(f"\n  Embeddings trainable: {embedding_trainable}")
    if not embedding_trainable:
        print(f"  ✓ Embeddings FROZEN")
    else:
        print(f"  ✗ Embeddings TRAINABLE (잘못됨!)")
        issues.append("Embeddings가 trainable임")

    # 4. Classifier seed 고정 확인
    print("\n[4] Classifier Seed 고정 확인")
    print("-"*80)

    print("\n  두 번째 모델 생성...")
    model2 = sweep_frozen.create_glue_model(
        "sst2", device, ["query", "key", "value"],
        fp_lora=False, lora_alpha=1.0
    )

    classifier_weight2 = None
    for name, param in model2.named_parameters():
        if name == 'base_model.model.classifier.weight':
            classifier_weight2 = param.clone().detach().cpu()

    if classifier_weight1 is not None and classifier_weight2 is not None:
        diff = (classifier_weight1 - classifier_weight2).abs().max().item()
        print(f"\n  Classifier difference: {diff:.10f}")

        if diff < 1e-6:
            print(f"  ✓ Classifier seed 고정됨 (동일한 초기화)")
        else:
            print(f"  ✗ Classifier seed 고정 안됨 (다른 초기화)!")
            issues.append("Classifier seed가 고정되지 않음")

    # 5. Target modules 확인
    print("\n[5] Target Modules")
    print("-"*80)

    qkv_count = {'query': 0, 'key': 0, 'value': 0}
    for name, module in model1.named_modules():
        if 'lora' in name.lower():
            if 'query' in name.lower():
                qkv_count['query'] += 1
            elif 'key' in name.lower():
                qkv_count['key'] += 1
            elif 'value' in name.lower():
                qkv_count['value'] += 1

    print(f"  Query LoRA layers: {qkv_count['query']}")
    print(f"  Key LoRA layers: {qkv_count['key']}")
    print(f"  Value LoRA layers: {qkv_count['value']}")

    if qkv_count['query'] > 0 and qkv_count['key'] > 0 and qkv_count['value'] > 0:
        print(f"  ✓ QKV 모두 LoRA 적용됨")
    else:
        print(f"  ✗ QKV 중 일부만 LoRA 적용됨!")
        issues.append("QKV 중 일부만 LoRA 적용")

    # 6. Shell script 확인
    print("\n[6] Shell Script 설정")
    print("-"*80)

    with open("/data/LRTT_transformer/experiments/sweep_lora_alpha_frozen_classifier.sh", "r") as f:
        script_content = f.read()

    alphas = ["0.01", "0.1", "1.0", "10.0", "100.0"]
    missing_alphas = []
    for alpha in alphas:
        if f"--lora_alpha {alpha}" not in script_content:
            missing_alphas.append(alpha)

    if len(missing_alphas) == 0:
        print(f"  ✓ 모든 alpha 값 포함: {', '.join(alphas)}")
    else:
        print(f"  ✗ 누락된 alpha 값: {', '.join(missing_alphas)}")
        issues.append(f"Alpha 값 누락: {missing_alphas}")

    if "--n_trials 10" in script_content:
        print(f"  ✓ Trials: 10 per alpha")
    else:
        print(f"  ✗ Trials 설정이 잘못됨")
        issues.append("Trials 설정 오류")

    # 최종 요약
    print("\n" + "="*80)
    print("  최종 검증 결과")
    print("="*80)

    print("\n  [요청 설정]")
    print("    Task: SST-2")
    print("    Mode: Sixt1c (analog 6T1C)")
    print("    Target: QKV")
    print("    LoRA alpha: 0.01, 0.1, 1.0, 10.0, 100.0")
    print("    LR range: [2e-4, 2e-2]")
    print("    Trials: 10 per alpha (총 50)")
    print("    Classifier: FROZEN (seed=42)")
    print("    Embeddings: FROZEN")
    print("    Optimizer: AnalogSGD")

    print("\n  [검증 결과]")
    if len(issues) == 0:
        print("    ✓✓✓ 모든 설정이 올바릅니다!")
        print("\n    실험을 시작할 준비가 되었습니다.")
    else:
        print("    ✗ 다음 문제들이 발견되었습니다:")
        for i, issue in enumerate(issues, 1):
            print(f"      {i}. {issue}")
        print("\n    문제를 수정한 후 다시 확인하세요.")

    print("\n" + "="*80 + "\n")

    return len(issues) == 0


if __name__ == "__main__":
    success = verify_all_settings()
    exit(0 if success else 1)
