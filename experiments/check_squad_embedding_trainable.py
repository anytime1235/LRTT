"""
SQuAD sweep에서 embedding이 trainable인지 frozen인지 확인
"""

import sys
import torch

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training')

from sweep_sixt1c_lora_squad_adam import create_squad_model
from transformers import AutoTokenizer


def check_squad_embedding_status():
    """SQuAD 모델의 embedding trainable 상태 확인"""

    print("="*80)
    print("  SQUAD 모델 - EMBEDDING TRAINABLE 상태 확인")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # Sixt1c 모드로 모델 생성
    print("\n[1] Sixt1c 모드 모델 생성 (fp_lora=False)...")
    model = create_squad_model(device, ["value"], fp_lora=False)

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

    for p in embedding_params:
        status = "✓ TRAINABLE" if p['requires_grad'] else "✗ FROZEN"
        print(f"\n  {p['name']}")
        print(f"    Shape: {p['shape']}")
        print(f"    Elements: {p['numel']:,}")
        print(f"    Status: {status}")

        if p['requires_grad']:
            trainable_count += 1
        else:
            frozen_count += 1

    # 전체 trainable parameters 요약
    print("\n[3] 전체 Trainable Parameters 분류")
    print("-"*80)

    categories = {
        'embedding': [],
        'encoder': [],
        'pooler': [],
        'lora': [],
        'qa_outputs': [],
        'other': []
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'embedding' in name.lower():
            categories['embedding'].append((name, param.numel()))
        elif 'lora' in name.lower():
            categories['lora'].append((name, param.numel()))
        elif 'qa_outputs' in name.lower():
            categories['qa_outputs'].append((name, param.numel()))
        elif 'encoder' in name.lower():
            categories['encoder'].append((name, param.numel()))
        elif 'pooler' in name.lower():
            categories['pooler'].append((name, param.numel()))
        else:
            categories['other'].append((name, param.numel()))

    for cat_name, params in categories.items():
        if len(params) > 0:
            total = sum(p[1] for p in params)
            print(f"\n  {cat_name.upper()}:")
            print(f"    Count: {len(params)}")
            print(f"    Total params: {total:,}")
            if len(params) <= 10:  # Show details if not too many
                for pname, pnum in params[:5]:
                    print(f"      - {pname}: {pnum:,}")
                if len(params) > 5:
                    print(f"      ... and {len(params)-5} more")

    # 4. 요약
    print("\n" + "="*80)
    print("  최종 결론")
    print("="*80)

    print(f"\n  Embedding parameters:")
    print(f"    Trainable: {trainable_count}")
    print(f"    Frozen: {frozen_count}")

    if trainable_count > 0:
        print(f"\n  ⚠️  EMBEDDINGS는 TRAINABLE입니다!")
        print(f"     → PEFT 기본 설정에서는 embedding이 trainable")
        print(f"     → Fine-tuning 시 embedding도 함께 학습됨")
    else:
        print(f"\n  ✓ EMBEDDINGS는 FROZEN입니다!")
        print(f"     → Pretrained embedding 유지")
        print(f"     → LoRA와 task head만 학습")

    print("\n  [PEFT LoRA 기본 동작]")
    print("    - PEFT 적용 후 대부분의 parameters는 frozen")
    print("    - 기본적으로 trainable:")
    print("      1. LoRA adapters (lora_A, lora_B)")
    print("      2. Task-specific head (qa_outputs)")
    print("    - Embeddings는 기본적으로 FROZEN")
    print("      (modules_to_save에 명시하지 않는 한)")

    print("\n" + "="*80 + "\n")

    return trainable_count > 0


def main():
    is_trainable = check_squad_embedding_status()

    print("\n최종 답변:")
    if is_trainable:
        print("  → Embeddings는 TRAINABLE (학습 가능)")
    else:
        print("  → Embeddings는 FROZEN (고정)")


if __name__ == "__main__":
    main()
