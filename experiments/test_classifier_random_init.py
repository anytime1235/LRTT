"""
매 trial마다 classifier가 다르게 초기화되는지 확인
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


def test_classifier_initialization():
    """여러 번 모델을 생성하여 classifier 초기화가 매번 다른지 확인"""

    print("="*80)
    print("  CLASSIFIER RANDOM INITIALIZATION 테스트")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifiers = []

    print("\n[1] 5번 모델 생성하여 classifier 가중치 수집...")

    for i in range(5):
        print(f"\n  Trial {i+1}:")
        model = sweep_frozen.create_glue_model(
            "sst2", device, ["query", "key", "value"],
            fp_lora=False, lora_alpha=1.0
        )

        # Classifier weights 추출
        for name, param in model.named_parameters():
            if name == 'base_model.model.classifier.weight':
                weight = param.clone().detach().cpu()
                classifiers.append({
                    'trial': i+1,
                    'weight': weight,
                    'mean': weight.mean().item(),
                    'std': weight.std().item(),
                    'max': weight.max().item(),
                })
                print(f"    Mean: {weight.mean().item():.6f}")
                print(f"    Std: {weight.std().item():.6f}")
                break

        # 메모리 정리
        del model
        torch.cuda.empty_cache()

    # 비교
    print("\n[2] Classifier 가중치 비교")
    print("-"*80)

    all_same = True
    for i in range(1, len(classifiers)):
        diff = (classifiers[i]['weight'] - classifiers[0]['weight']).abs().max().item()
        print(f"\n  Trial 1 vs Trial {i+1}:")
        print(f"    Max difference: {diff:.8f}")

        if diff > 1e-6:
            print(f"    ✗ DIFFERENT - 매번 다르게 초기화됨!")
            all_same = False
        else:
            print(f"    ✓ SAME - 동일하게 초기화됨")

    # 결과
    print("\n" + "="*80)
    print("  결과")
    print("="*80)

    if all_same:
        print("\n  ✓ Classifier가 매번 동일하게 초기화됨")
        print("    → Seed가 고정되어 있음")
        print("    → 공정한 비교 가능")
    else:
        print("\n  ✗ Classifier가 매번 다르게 초기화됨")
        print("    → Seed가 고정되지 않음")
        print("    → Frozen classifier의 경우 문제!")
        print("\n  문제점:")
        print("    - 각 trial이 다른 frozen classifier로 시작")
        print("    - LoRA만 학습되지만 초기 classifier가 다름")
        print("    - 공정한 비교 불가능")
        print("\n  해결책:")
        print("    - Classifier 초기화 전에 seed 고정")
        print("    - 모든 trial이 동일한 classifier로 시작")

    print("\n" + "="*80 + "\n")

    return all_same


if __name__ == "__main__":
    same = test_classifier_initialization()
    exit(0 if same else 1)
