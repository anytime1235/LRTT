"""
Frozen Classifier 설정 검증
"""

import sys
import torch

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

# Import from the MODIFIED script with frozen classifier
import importlib.util
spec = importlib.util.spec_from_file_location(
    "sweep_frozen",
    "/data/LRTT_transformer/LRTT_glue/sweep_sixt1c_lora_glue_frozen_classifier.py"
)
sweep_frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep_frozen)


def verify_frozen_classifier():
    """Frozen classifier 설정 검증"""

    print("="*80)
    print("  FROZEN CLASSIFIER 실험 설정 검증")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[1] 모델 생성 (frozen classifier script)...")
    model = sweep_frozen.create_glue_model(
        "sst2", device, ["query", "key", "value"],
        fp_lora=False, lora_alpha=1.0
    )

    print("\n[2] Trainable Parameters 확인")
    print("-"*80)

    categories = {
        'embedding': {'trainable': 0, 'frozen': 0},
        'encoder': {'trainable': 0, 'frozen': 0},
        'lora': {'trainable': 0, 'frozen': 0},
        'classifier': {'trainable': 0, 'frozen': 0},
    }

    classifier_params = []
    lora_params = []

    for name, param in model.named_parameters():
        if 'embedding' in name.lower():
            if param.requires_grad:
                categories['embedding']['trainable'] += param.numel()
            else:
                categories['embedding']['frozen'] += param.numel()
        elif 'lora' in name.lower():
            if param.requires_grad:
                categories['lora']['trainable'] += param.numel()
                lora_params.append(name)
            else:
                categories['lora']['frozen'] += param.numel()
        elif 'classifier' in name.lower():
            if param.requires_grad:
                categories['classifier']['trainable'] += param.numel()
                classifier_params.append((name, 'TRAINABLE'))
            else:
                categories['classifier']['frozen'] += param.numel()
                classifier_params.append((name, 'FROZEN'))
        elif 'encoder' in name.lower():
            if param.requires_grad:
                categories['encoder']['trainable'] += param.numel()
            else:
                categories['encoder']['frozen'] += param.numel()

    # Print summary
    print("\n  Category Summary:")
    for cat, counts in categories.items():
        total = counts['trainable'] + counts['frozen']
        trainable_pct = (counts['trainable'] / total * 100) if total > 0 else 0

        print(f"\n  {cat.upper()}:")
        print(f"    Trainable: {counts['trainable']:,} ({trainable_pct:.1f}%)")
        print(f"    Frozen: {counts['frozen']:,} ({100-trainable_pct:.1f}%)")

    # Classifier details
    print("\n[3] Classifier 상세 확인")
    print("-"*80)

    if len(classifier_params) == 0:
        print("  ✗ Classifier parameters not found!")
    else:
        for name, status in classifier_params:
            print(f"  {name}: {status}")

    # LoRA details (first 5)
    print("\n[4] LoRA Parameters (처음 5개)")
    print("-"*80)

    for name in lora_params[:5]:
        print(f"  ✓ {name}: TRAINABLE")
    if len(lora_params) > 5:
        print(f"  ... and {len(lora_params)-5} more")

    # Final verification
    print("\n" + "="*80)
    print("  검증 결과")
    print("="*80)

    all_classifier_frozen = categories['classifier']['trainable'] == 0
    lora_trainable = categories['lora']['trainable'] > 0

    if all_classifier_frozen:
        print("\n  ✓✓✓ CLASSIFIER가 완전히 FROZEN입니다!")
    else:
        print("\n  ✗✗✗ CLASSIFIER가 TRAINABLE입니다! (수정 필요)")

    if lora_trainable:
        print("  ✓ LoRA adapters는 TRAINABLE입니다!")
    else:
        print("  ✗ LoRA adapters가 FROZEN입니다! (문제)")

    print("\n  실험 설정:")
    print(f"    Task: SST-2")
    print(f"    Target: QKV")
    print(f"    LR range: [2e-4, 2e-2]")
    print(f"    LoRA alpha: 0.01, 0.1, 1.0, 10.0, 100.0")
    print(f"    Trials: 10 per alpha (50 total)")
    print(f"    Embeddings: FROZEN")
    print(f"    Classifier: {'FROZEN ✓' if all_classifier_frozen else 'TRAINABLE ✗'}")
    print(f"    LoRA: {'TRAINABLE ✓' if lora_trainable else 'FROZEN ✗'}")

    print("\n" + "="*80 + "\n")

    return all_classifier_frozen and lora_trainable


if __name__ == "__main__":
    success = verify_frozen_classifier()
    if success:
        print("✓ 설정 검증 완료! 실험을 시작할 수 있습니다.\n")
    else:
        print("✗ 설정에 문제가 있습니다. 수정이 필요합니다.\n")
    exit(0 if success else 1)
