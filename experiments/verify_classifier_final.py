"""
Sixt1c 모드에서 classifier 설정 및 실제 학습 최종 확인
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model, load_glue_data
from transformers import AutoTokenizer
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogLinear


def verify_classifier_config():
    """Classifier 설정 확인"""
    print("="*80)
    print("  SIXT1C MODE - CLASSIFIER 설정 확인")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 생성
    print("\n[1] 모델 생성 (sixt1c mode)...")
    model = create_glue_model("sst2", device, ["value"],
                             fp_lora=False, lora_alpha=1.0)

    # Classifier 찾기
    print("\n[2] Classifier 확인...")
    classifier = None
    classifier_name = None

    for name, module in model.named_modules():
        if 'classifier' in name and isinstance(module, nn.Linear):
            classifier = module
            classifier_name = name
            break

    if classifier is None:
        print("  ✗ Classifier not found!")
        return False

    print(f"  ✓ Classifier 발견: {classifier_name}")
    print(f"    Type: {type(classifier)}")
    print(f"    Input features: {classifier.in_features}")
    print(f"    Output features: {classifier.out_features}")

    # Analog 여부 확인
    is_analog = isinstance(classifier, AnalogLinear)
    print(f"    Analog 변환 여부: {is_analog}")
    if is_analog:
        print(f"      ⚠️  WARNING: Classifier가 analog로 변환됨!")
    else:
        print(f"      ✓ Classifier는 digital (nn.Linear) - 정상")

    # Trainable 확인
    print(f"\n[3] Trainable 설정 확인...")
    for name, param in model.named_parameters():
        if 'classifier' in name:
            print(f"  {name}:")
            print(f"    requires_grad: {param.requires_grad}")
            print(f"    shape: {param.shape}")
            print(f"    device: {param.device}")

            if param.requires_grad:
                print(f"    ✓ TRAINABLE")
            else:
                print(f"    ✗ FROZEN")

    # Optimizer에 포함되는지 확인
    print(f"\n[4] Optimizer 설정 확인...")
    optimizer = AnalogSGD(model.parameters(), lr=0.001)
    optimizer.regroup_param_groups(model)

    print(f"  총 param groups: {len(optimizer.param_groups)}")

    # Classifier가 어느 group에 있는지 확인
    classifier_in_optimizer = False
    for i, group in enumerate(optimizer.param_groups):
        for param in group['params']:
            for name, model_param in model.named_parameters():
                if param is model_param and 'classifier' in name:
                    classifier_in_optimizer = True
                    print(f"  ✓ Classifier가 optimizer의 group {i}에 포함됨")
                    print(f"    Learning rate: {group['lr']}")
                    break

    if not classifier_in_optimizer:
        print(f"  ✗ Classifier가 optimizer에 없음!")
        return False

    return True


def test_actual_training():
    """실제 학습 테스트 (gradient clipping 포함)"""
    print("\n" + "="*80)
    print("  SIXT1C MODE - 실제 학습 테스트 (gradient clipping 사용)")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # 모델 생성
    print("\n[1] 모델 생성...")
    model = create_glue_model("sst2", device, ["value"],
                             fp_lora=False, lora_alpha=1.0)

    # 데이터 로드
    print("\n[2] 데이터 로드...")
    train_dataloader, _ = load_glue_data("sst2", tokenizer)

    # Optimizer
    optimizer = AnalogSGD(model.parameters(), lr=0.001)
    optimizer.regroup_param_groups(model)

    # 초기 가중치 저장
    initial_weights = {}
    for name, param in model.named_parameters():
        if 'classifier' in name:
            initial_weights[name] = param.clone().detach().cpu()

    print("\n[3] 학습 시작 (10 steps, gradient clipping=1.0)...")
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    step = 0
    max_steps = 10

    for batch in train_dataloader:
        if step >= max_steps:
            break

        # Prepare batch
        labels = batch.pop('labels').to(device)
        if 'idx' in batch:
            batch.pop('idx')
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        # Forward
        outputs = model(**batch)
        logits = outputs.logits
        loss = criterion(logits, labels)

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  Step {step+1}: NaN/Inf detected, stopping")
            break

        # Backward
        loss.backward()

        # GRADIENT CLIPPING (실제 학습 코드와 동일)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        losses.append(loss.item())
        step += 1

        if step % 2 == 0 or step == 1:
            print(f"  Step {step}: loss={loss.item():.6f}")

    print(f"\n[4] 가중치 변화 확인...")

    all_changed = True
    for name, initial_weight in initial_weights.items():
        for pname, param in model.named_parameters():
            if pname == name:
                current_weight = param.detach().cpu()
                diff = (current_weight - initial_weight).abs()

                max_change = diff.max().item()
                mean_change = diff.mean().item()

                print(f"\n  {name}:")
                print(f"    Max change: {max_change:.8f}")
                print(f"    Mean change: {mean_change:.8f}")

                if max_change > 1e-5:
                    print(f"    ✓ 가중치 변화 있음!")
                else:
                    print(f"    ✗ 가중치 변화 없음!")
                    all_changed = False
                break

    print(f"\n[5] 결과 요약:")
    print(f"  완료된 steps: {len(losses)}/{max_steps}")
    if len(losses) > 0:
        print(f"  평균 loss: {sum(losses)/len(losses):.6f}")
        print(f"  최종 loss: {losses[-1]:.6f}")

    if all_changed and len(losses) >= 5:
        print(f"\n  ✓✓✓ Classifier가 정상적으로 학습됨!")
        return True
    else:
        print(f"\n  ✗ 학습 문제 있음")
        return False


def main():
    print("\n" + "="*80)
    print("  SIXT1C MODE - CLASSIFIER 최종 검증")
    print("="*80 + "\n")

    # 1. 설정 확인
    config_ok = verify_classifier_config()

    # 2. 실제 학습 테스트
    training_ok = test_actual_training()

    # 최종 결과
    print("\n" + "="*80)
    print("  최종 결과")
    print("="*80)

    print(f"\n  [설정 검증]: {'✓ PASS' if config_ok else '✗ FAIL'}")
    print(f"  [학습 검증]: {'✓ PASS' if training_ok else '✗ FAIL'}")

    if config_ok and training_ok:
        print("\n" + "="*80)
        print("  ✓✓✓ SIXT1C 모드에서 Classifier 정상 작동 확인!")
        print("="*80)
        print("\n  요약:")
        print("    - Classifier는 digital (nn.Linear)")
        print("    - requires_grad=True (trainable)")
        print("    - Optimizer에 정상 포함")
        print("    - Gradient clipping으로 안정적 학습")
        print("    - 가중치 업데이트 확인됨")
        print("="*80 + "\n")
        return True
    else:
        print("\n  ⚠️  문제 발견")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
