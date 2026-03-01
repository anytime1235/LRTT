"""
현재 실행 중인 SST-2 sweep에서 embedding이 trainable인지 frozen인지 확인
"""

import sys
import torch

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model


def check_sst2_embedding_status():
    """SST-2 sweep 모델의 embedding trainable 상태 확인"""

    print("="*80)
    print("  SST-2 실험 - EMBEDDING TRAINABLE 상태 확인")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 현재 실행 중인 실험과 동일한 설정으로 모델 생성
    print("\n[1] 모델 생성 (SST-2, target=QKV, sixt1c mode, lora_alpha=1.0)...")
    model = create_glue_model("sst2", device, ["query", "key", "value"],
                             fp_lora=False, lora_alpha=1.0)

    print("\n[2] Embedding 관련 Parameters 확인")
    print("-"*80)

    embedding_params = []
    for name, param in model.named_parameters():
        if 'embedding' in name.lower():
            embedding_params.append({
                'name': name,
                'shape': param.shape,
                'requires_grad': param.requires_grad,
                'numel': param.numel()
            })

    print(f"\n  총 embedding parameters: {len(embedding_params)}")

    trainable_count = 0
    frozen_count = 0
    trainable_params = 0
    frozen_params = 0

    for p in embedding_params:
        status = "✓ TRAINABLE" if p['requires_grad'] else "✗ FROZEN"
        print(f"\n  {p['name']}")
        print(f"    Shape: {p['shape']}")
        print(f"    Elements: {p['numel']:,}")
        print(f"    Status: {status}")

        if p['requires_grad']:
            trainable_count += 1
            trainable_params += p['numel']
        else:
            frozen_count += 1
            frozen_params += p['numel']

    # 전체 trainable parameters 요약
    print("\n[3] 전체 Trainable Parameters 분류")
    print("-"*80)

    categories = {
        'embedding': {'count': 0, 'params': 0},
        'encoder': {'count': 0, 'params': 0},
        'pooler': {'count': 0, 'params': 0},
        'lora': {'count': 0, 'params': 0},
        'classifier': {'count': 0, 'params': 0},
        'other': {'count': 0, 'params': 0}
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'embedding' in name.lower():
            categories['embedding']['count'] += 1
            categories['embedding']['params'] += param.numel()
        elif 'lora' in name.lower():
            categories['lora']['count'] += 1
            categories['lora']['params'] += param.numel()
        elif 'classifier' in name.lower():
            categories['classifier']['count'] += 1
            categories['classifier']['params'] += param.numel()
        elif 'encoder' in name.lower():
            categories['encoder']['count'] += 1
            categories['encoder']['params'] += param.numel()
        elif 'pooler' in name.lower():
            categories['pooler']['count'] += 1
            categories['pooler']['params'] += param.numel()
        else:
            categories['other']['count'] += 1
            categories['other']['params'] += param.numel()

    for cat_name, info in categories.items():
        if info['count'] > 0:
            print(f"\n  {cat_name.upper()}:")
            print(f"    Count: {info['count']}")
            print(f"    Total params: {info['params']:,}")

    # 4. 요약
    print("\n" + "="*80)
    print("  최종 결론")
    print("="*80)

    print(f"\n  Embedding parameters:")
    print(f"    Total: {len(embedding_params)} layers")
    print(f"    Trainable: {trainable_count} layers ({trainable_params:,} params)")
    print(f"    Frozen: {frozen_count} layers ({frozen_params:,} params)")

    print("\n" + "="*80)
    if trainable_count > 0:
        print("  ⚠️  SST-2 실험에서 EMBEDDINGS는 TRAINABLE입니다!")
        print("="*80)
        print("\n  의미:")
        print("    → Embedding도 함께 학습됨")
        print("    → Pretrained embedding이 fine-tuning 됨")
        print("    → 더 많은 parameters가 학습됨")
        print("    → Task-specific하게 embedding 조정 가능")
    else:
        print("  ✓ SST-2 실험에서 EMBEDDINGS는 FROZEN입니다!")
        print("="*80)
        print("\n  의미:")
        print("    → Pretrained embedding 그대로 유지")
        print("    → LoRA adapters + classifier만 학습")
        print("    → Parameter-efficient fine-tuning")
        print("    → PEFT 기본 동작")

    print("\n  [현재 실험 설정 요약]")
    print(f"    Task: SST-2")
    print(f"    Target modules: QKV (query, key, value)")
    print(f"    Mode: Sixt1c (analog)")
    print(f"    LoRA alpha: 0.1, 1.0, 10.0 (3가지 테스트)")
    print(f"    Learning rate: [1e-4, 1e-2] (log scale)")
    print(f"    Trials per alpha: 10")
    print(f"    Optimizer: AnalogSGD")
    print(f"    Embeddings: {'TRAINABLE' if trainable_count > 0 else 'FROZEN'}")

    print("\n" + "="*80 + "\n")

    return trainable_count > 0


def main():
    is_trainable = check_sst2_embedding_status()

    print("\n📌 현재 실행 중인 SST-2 실험:")
    if is_trainable:
        print("   → Embeddings는 TRAINABLE (학습됨)")
    else:
        print("   → Embeddings는 FROZEN (고정됨)")


if __name__ == "__main__":
    main()
