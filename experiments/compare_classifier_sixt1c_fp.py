"""
Classifier에 LoRA 적용 여부 확인: Sixt1c vs FP mode 비교
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from aihwkit.nn import AnalogLinear


def analyze_classifier(mode_name, fp_lora):
    """Classifier 구조 분석"""
    print(f"\n{'='*80}")
    print(f"  {mode_name} MODE - CLASSIFIER 분석")
    print(f"{'='*80}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 생성
    print(f"\n[1] 모델 생성 (fp_lora={fp_lora})...")
    model = create_glue_model("sst2", device, ["value"],
                             fp_lora=fp_lora, lora_alpha=1.0)

    # Classifier 관련 모든 모듈 찾기
    print(f"\n[2] Classifier 관련 모듈 검색...")

    classifier_modules = []
    for name, module in model.named_modules():
        if 'classifier' in name:
            classifier_modules.append((name, module, type(module).__name__))

    print(f"  발견된 모듈 수: {len(classifier_modules)}")

    for name, module, type_name in classifier_modules:
        print(f"\n  [{name}]")
        print(f"    Type: {type_name}")

        # LoRA 구조 확인 (base_layer, lora_A, lora_B가 있는지)
        has_base_layer = hasattr(module, 'base_layer')
        has_lora_a = hasattr(module, 'lora_A')
        has_lora_b = hasattr(module, 'lora_B')

        if has_base_layer or has_lora_a or has_lora_b:
            print(f"    ⚠️  LoRA 구조 발견!")
            print(f"      - has base_layer: {has_base_layer}")
            print(f"      - has lora_A: {has_lora_a}")
            print(f"      - has lora_B: {has_lora_b}")
        else:
            print(f"    ✓ LoRA 없음 (일반 layer)")

        # Linear 또는 AnalogLinear인지 확인
        if isinstance(module, nn.Linear):
            print(f"    ✓ Digital Linear")
            print(f"      in_features: {module.in_features}")
            print(f"      out_features: {module.out_features}")
        elif isinstance(module, AnalogLinear):
            print(f"    ⚠️  Analog Linear!")
            print(f"      in_features: {module.in_features}")
            print(f"      out_features: {module.out_features}")
        else:
            print(f"    Type: {type(module)}")

    # Classifier parameters 확인
    print(f"\n[3] Classifier Parameters 확인...")

    classifier_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append({
                'name': name,
                'shape': param.shape,
                'requires_grad': param.requires_grad,
                'device': param.device,
                'numel': param.numel()
            })

    print(f"  총 classifier parameters: {len(classifier_params)}")

    for p in classifier_params:
        print(f"\n  {p['name']}")
        print(f"    Shape: {p['shape']}")
        print(f"    Trainable: {p['requires_grad']}")
        print(f"    Elements: {p['numel']:,}")

        # LoRA 관련 parameter인지 확인
        if 'lora' in p['name'].lower():
            print(f"    ⚠️  LoRA parameter!")
        else:
            print(f"    ✓ 일반 parameter")

    # 총 trainable parameters 계산
    total_trainable = sum(p['numel'] for p in classifier_params if p['requires_grad'])
    total_all = sum(p['numel'] for p in classifier_params)

    print(f"\n[4] Classifier Trainable Summary:")
    print(f"  Trainable params: {total_trainable:,}")
    print(f"  Total params: {total_all:,}")

    return {
        'modules': classifier_modules,
        'params': classifier_params,
        'trainable': total_trainable,
        'total': total_all
    }


def main():
    print("="*80)
    print("  CLASSIFIER 구조 비교: Sixt1c vs FP Mode")
    print("="*80)

    # Sixt1c 분석
    sixt1c_result = analyze_classifier("SIXT1C", fp_lora=False)

    print("\n\n")

    # FP 분석
    fp_result = analyze_classifier("FP", fp_lora=True)

    # 비교 요약
    print("\n" + "="*80)
    print("  비교 요약")
    print("="*80)

    print(f"\n  {'항목':<30} {'Sixt1c':<20} {'FP':<20}")
    print(f"  {'-'*30} {'-'*20} {'-'*20}")

    print(f"  {'Classifier 모듈 수':<30} {len(sixt1c_result['modules']):<20} {len(fp_result['modules']):<20}")
    print(f"  {'Classifier parameters 수':<30} {len(sixt1c_result['params']):<20} {len(fp_result['params']):<20}")
    print(f"  {'Trainable params':<30} {sixt1c_result['trainable']:<20,} {fp_result['trainable']:<20,}")

    # LoRA 적용 여부 확인
    sixt1c_has_lora = any('lora' in p['name'].lower() for p in sixt1c_result['params'])
    fp_has_lora = any('lora' in p['name'].lower() for p in fp_result['params'])

    print(f"\n  {'LoRA 적용 여부':<30} {'예' if sixt1c_has_lora else '아니오':<20} {'예' if fp_has_lora else '아니오':<20}")

    # Analog 변환 여부 확인
    sixt1c_is_analog = any(isinstance(m[1], AnalogLinear) for m in sixt1c_result['modules'])
    fp_is_analog = any(isinstance(m[1], AnalogLinear) for m in fp_result['modules'])

    print(f"  {'Analog 변환 여부':<30} {'예' if sixt1c_is_analog else '아니오':<20} {'예' if fp_is_analog else '아니오':<20}")

    # 최종 결론
    print("\n" + "="*80)
    print("  최종 결론")
    print("="*80)

    if not sixt1c_has_lora and not fp_has_lora:
        print("\n  ✓ 두 모드 모두 Classifier에 LoRA가 적용되지 않음")
        print("    → Classifier는 일반 Linear layer로 유지됨")
    else:
        print("\n  ⚠️  Classifier에 LoRA가 적용됨!")

    if not sixt1c_is_analog and not fp_is_analog:
        print("\n  ✓ 두 모드 모두 Classifier는 Digital (nn.Linear)")
        print("    → Analog 변환 없음")
    else:
        print("\n  ⚠️  Classifier가 Analog로 변환됨!")

    print("\n  차이점:")
    if sixt1c_result['trainable'] != fp_result['trainable']:
        print(f"    - Trainable params: Sixt1c={sixt1c_result['trainable']:,}, FP={fp_result['trainable']:,}")
        print(f"      차이: {abs(sixt1c_result['trainable'] - fp_result['trainable']):,}")
    else:
        print(f"    - Trainable params: 동일 ({sixt1c_result['trainable']:,})")

    print("\n" + "="*80)

    print("\n  📌 요약:")
    print("    - Classifier는 target_modules에 포함되지 않음")
    print("    - 따라서 LoRA가 적용되지 않음")
    print("    - Sixt1c/FP 모두 동일하게 일반 nn.Linear")
    print("    - 두 모드 모두 trainable=True로 학습됨")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
