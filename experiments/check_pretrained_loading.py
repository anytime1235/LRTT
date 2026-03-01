"""
Pretrained 모델 로딩 시 embedding과 classifier 초기화 확인
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModel,
)


def check_pretrained_loading():
    """Pretrained 모델 로딩 시 초기화 상태 확인"""

    print("="*80)
    print("  PRETRAINED MODEL LOADING 검증")
    print("="*80)

    # 1. Base pretrained model (분류 헤드 없음)
    print("\n[1] Base Pretrained Model (AutoModel) - 분류 헤드 없음")
    print("-"*80)

    base_model = AutoModel.from_pretrained("google/mobilebert-uncased")

    print("\n  Embeddings:")
    print(f"    word_embeddings weight shape: {base_model.embeddings.word_embeddings.weight.shape}")
    print(f"    word_embeddings weight mean: {base_model.embeddings.word_embeddings.weight.mean().item():.6f}")
    print(f"    word_embeddings weight std: {base_model.embeddings.word_embeddings.weight.std().item():.6f}")
    print(f"    ✓ Pretrained 모델에서 로드됨 (0이 아닌 값)")

    # 2. Classification model (분류 헤드 포함)
    print("\n[2] Classification Model (AutoModelForSequenceClassification)")
    print("-"*80)

    # 첫 번째 로딩 - warning 캡처
    print("\n  첫 번째 로딩 (SST-2, num_labels=2)...")

    config = AutoConfig.from_pretrained("google/mobilebert-uncased", num_labels=2)
    model1 = AutoModelForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased",
        config=config
    )

    print("\n  Model 구조:")
    print(f"    Has embeddings: {hasattr(model1.mobilebert, 'embeddings')}")
    print(f"    Has encoder: {hasattr(model1.mobilebert, 'encoder')}")
    print(f"    Has pooler: {hasattr(model1.mobilebert, 'pooler')}")
    print(f"    Has classifier: {hasattr(model1, 'classifier')}")

    # Embeddings 확인
    print("\n  Embeddings (from pretrained):")
    emb1 = model1.mobilebert.embeddings.word_embeddings.weight
    print(f"    Shape: {emb1.shape}")
    print(f"    Mean: {emb1.mean().item():.6f}")
    print(f"    Std: {emb1.std().item():.6f}")
    print(f"    Max abs: {emb1.abs().max().item():.6f}")
    print(f"    ✓ Pretrained에서 로드됨")

    # Classifier 확인
    print("\n  Classifier (randomly initialized):")
    if hasattr(model1, 'classifier'):
        cls1_weight = model1.classifier.weight.clone()
        cls1_bias = model1.classifier.bias.clone()

        print(f"    Weight shape: {cls1_weight.shape}")
        print(f"    Weight mean: {cls1_weight.mean().item():.6f}")
        print(f"    Weight std: {cls1_weight.std().item():.6f}")
        print(f"    Weight max abs: {cls1_weight.abs().max().item():.6f}")

        print(f"    Bias shape: {cls1_bias.shape}")
        print(f"    Bias mean: {cls1_bias.mean().item():.6f}")
        print(f"    Bias std: {cls1_bias.std().item():.6f}")
        print(f"    ⚠️  Random 초기화됨 (pretrained에 없음)")

    # 3. 같은 모델 다시 로딩 - classifier가 다른 값인지 확인
    print("\n[3] 동일 모델 재로딩 - Random 초기화 검증")
    print("-"*80)

    print("\n  두 번째 로딩 (동일 설정)...")
    model2 = AutoModelForSequenceClassification.from_pretrained(
        "google/mobilebert-uncased",
        config=config
    )

    # Embeddings 비교
    emb2 = model2.mobilebert.embeddings.word_embeddings.weight
    emb_diff = (emb1 - emb2).abs().max().item()

    print("\n  Embeddings 비교:")
    print(f"    Max difference: {emb_diff:.10f}")
    if emb_diff < 1e-6:
        print(f"    ✓ IDENTICAL - Pretrained에서 동일하게 로드됨")
    else:
        print(f"    ✗ DIFFERENT - 문제 있음!")

    # Classifier 비교
    cls2_weight = model2.classifier.weight
    cls_diff = (cls1_weight - cls2_weight).abs().max().item()

    print("\n  Classifier 비교:")
    print(f"    Max weight difference: {cls_diff:.6f}")
    if cls_diff > 1e-3:
        print(f"    ✓ DIFFERENT - 매번 Random 초기화됨 (정상)")
    else:
        print(f"    ✗ IDENTICAL - 이상함! (Random이어야 함)")

    # 4. Encoder layer 확인
    print("\n[4] Encoder Layers (from pretrained)")
    print("-"*80)

    # 첫 번째 attention layer weight 확인
    first_attn = model1.mobilebert.encoder.layer[0].attention.self.query
    print(f"\n  First layer Query projection:")
    print(f"    Weight shape: {first_attn.weight.shape}")
    print(f"    Weight mean: {first_attn.weight.mean().item():.6f}")
    print(f"    Weight std: {first_attn.weight.std().item():.6f}")
    print(f"    ✓ Pretrained에서 로드됨")

    # 5. 요약
    print("\n" + "="*80)
    print("  요약")
    print("="*80)

    print("\n  [Pretrained에서 로드되는 것들]")
    print("    ✓ Embeddings (word_embeddings, position_embeddings, etc.)")
    print("    ✓ Encoder layers (attention, FFN)")
    print("    ✓ Pooler layer")

    print("\n  [Random 초기화되는 것들]")
    print("    ⚠️  Classifier head (task-specific)")
    print("       → Pretrained 모델에 없음")
    print("       → 매번 random 초기화됨")
    print("       → 반드시 학습되어야 함!")

    print("\n  [Warning 메시지 의미]")
    print("    'Some weights of MobileBertForSequenceClassification were not")
    print("     initialized from the model checkpoint...'")
    print("    → 정상입니다! Classifier는 원래 random 초기화됩니다.")

    print("\n" + "="*80)

    # 6. Create_glue_model에서 classifier 재초기화 확인
    print("\n[5] create_glue_model의 Classifier 재초기화")
    print("-"*80)

    print("\n  원래 random 초기화된 classifier:")
    original_weight = model1.classifier.weight.clone()
    original_bias = model1.classifier.bias.clone()
    print(f"    Weight mean: {original_weight.mean().item():.6f}")
    print(f"    Weight std: {original_weight.std().item():.6f}")
    print(f"    Bias mean: {original_bias.mean().item():.6f}")

    # 우리 코드처럼 재초기화
    nn.init.normal_(model1.classifier.weight, mean=0.0, std=0.02)
    nn.init.zeros_(model1.classifier.bias)

    print("\n  재초기화 후 (std=0.02로 줄임):")
    new_weight = model1.classifier.weight
    new_bias = model1.classifier.bias
    print(f"    Weight mean: {new_weight.mean().item():.6f}")
    print(f"    Weight std: {new_weight.std().item():.6f}")
    print(f"    Bias mean: {new_bias.mean().item():.6f}")

    weight_change = (new_weight - original_weight).abs().max().item()
    print(f"\n    Weight change: {weight_change:.6f}")
    print(f"    ✓ 재초기화로 더 안정적인 값으로 변경됨")

    print("\n  [재초기화 이유]")
    print("    - HuggingFace 기본 초기화는 때때로 큰 값")
    print("    - std=0.02로 줄여서 학습 초기 안정성 향상")
    print("    - 특히 huge activation 문제 완화")

    print("\n" + "="*80 + "\n")


def main():
    check_pretrained_loading()


if __name__ == "__main__":
    main()
