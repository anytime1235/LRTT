"""
Verify that classifier is actually being trained in both sixt1c and FP modes.
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model, load_glue_data
from transformers import AutoTokenizer
from aihwkit.optim import AnalogSGD


def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def check_classifier_trainability(model, mode_name):
    """Check if classifier parameters are set as trainable."""
    print(f"\n[{mode_name}] Checking classifier trainability...")

    classifier_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append({
                'name': name,
                'requires_grad': param.requires_grad,
                'shape': param.shape,
                'device': param.device,
            })

    print(f"\n  Found {len(classifier_params)} classifier parameters:")
    for p in classifier_params:
        status = "✓ Trainable" if p['requires_grad'] else "✗ Frozen"
        print(f"    {status}: {p['name']}")
        print(f"      Shape: {p['shape']}, Device: {p['device']}")

    all_trainable = all(p['requires_grad'] for p in classifier_params)

    if all_trainable:
        print(f"\n  ✓ All classifier parameters are trainable")
    else:
        print(f"\n  ✗ Some classifier parameters are frozen!")

    return classifier_params, all_trainable


def test_classifier_learning(model, tokenizer, device, mode_name):
    """Test if classifier actually learns during training."""
    print(f"\n[{mode_name}] Testing classifier learning...")

    # Get initial classifier weights
    initial_weights = {}
    for name, param in model.named_parameters():
        if 'classifier' in name:
            initial_weights[name] = param.clone().detach().cpu()

    print(f"\n  Initial classifier weights:")
    for name, weights in initial_weights.items():
        print(f"    {name}:")
        print(f"      Mean: {weights.mean().item():.6f}")
        print(f"      Std: {weights.std().item():.6f}")
        print(f"      Max abs: {weights.abs().max().item():.6f}")

    # Prepare training data
    texts = [
        "This movie is great!",
        "This movie is terrible!",
        "I love this film!",
        "Worst movie ever!",
    ] * 5  # 20 samples
    labels_list = [1, 0, 1, 0] * 5

    # Setup optimizer
    optimizer = AnalogSGD(model.parameters(), lr=0.01)
    optimizer.regroup_param_groups(model)

    # Training
    model.train()
    criterion = nn.CrossEntropyLoss()

    print(f"\n  Training on {len(texts)} samples...")

    losses = []
    for i, (text, label) in enumerate(zip(texts, labels_list)):
        inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                          max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.tensor([label]).to(device)

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % 5 == 0:
            avg_loss = sum(losses[-5:]) / 5
            print(f"    Step {i+1}/{len(texts)}: loss = {avg_loss:.4f}")

    avg_final_loss = sum(losses[-5:]) / 5

    # Check weight changes
    print(f"\n  Checking weight changes...")

    weight_changes = {}
    for name, param in model.named_parameters():
        if name in initial_weights:
            current_weights = param.clone().detach().cpu()
            diff = (current_weights - initial_weights[name]).abs()

            weight_changes[name] = {
                'max_diff': diff.max().item(),
                'mean_diff': diff.mean().item(),
                'current_mean': current_weights.mean().item(),
                'current_std': current_weights.std().item(),
            }

    print(f"\n  Final classifier weights:")
    for name, changes in weight_changes.items():
        print(f"    {name}:")
        print(f"      Current - Mean: {changes['current_mean']:.6f}, Std: {changes['current_std']:.6f}")
        print(f"      Change  - Max: {changes['max_diff']:.6f}, Mean: {changes['mean_diff']:.6f}")

    # Determine if learning occurred
    max_change = max(c['max_diff'] for c in weight_changes.values())
    mean_change = sum(c['mean_diff'] for c in weight_changes.values()) / len(weight_changes)

    learning_threshold = 1e-4
    is_learning = max_change > learning_threshold

    print(f"\n  Summary:")
    print(f"    Final loss: {avg_final_loss:.4f}")
    print(f"    Max weight change: {max_change:.6f}")
    print(f"    Mean weight change: {mean_change:.6f}")

    if is_learning:
        print(f"\n  ✓ Classifier IS learning (weights changed significantly)")
    else:
        print(f"\n  ✗ Classifier NOT learning (weights didn't change enough)")

    return is_learning, max_change, avg_final_loss


def main():
    print_section("CLASSIFIER TRAINING VERIFICATION")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_name = "sst2"
    target_modules = ["value"]
    lora_alpha = 1.0

    print(f"\nConfiguration:")
    print(f"  Task: {task_name}")
    print(f"  Target modules: {target_modules}")
    print(f"  Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # =========================================================================
    # Test 1: Sixt1c Mode
    # =========================================================================
    print_section("TEST 1: SIXT1C MODE (Analog LoRA)")

    print("\n  Creating model in sixt1c mode...")
    model_sixt1c = create_glue_model(task_name, device, target_modules,
                                     fp_lora=False, lora_alpha=lora_alpha)

    # Check trainability
    sixt1c_params, sixt1c_trainable = check_classifier_trainability(model_sixt1c, "Sixt1c")

    # Test learning
    sixt1c_learning, sixt1c_change, sixt1c_loss = test_classifier_learning(
        model_sixt1c, tokenizer, device, "Sixt1c"
    )

    # Clean up
    del model_sixt1c
    torch.cuda.empty_cache()

    # =========================================================================
    # Test 2: FP Mode
    # =========================================================================
    print_section("TEST 2: FP MODE (Digital LoRA)")

    print("\n  Creating model in FP mode...")
    model_fp = create_glue_model(task_name, device, target_modules,
                                 fp_lora=True, lora_alpha=lora_alpha)

    # Check trainability
    fp_params, fp_trainable = check_classifier_trainability(model_fp, "FP")

    # Test learning
    fp_learning, fp_change, fp_loss = test_classifier_learning(
        model_fp, tokenizer, device, "FP"
    )

    # Clean up
    del model_fp
    torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print_section("FINAL SUMMARY")

    print(f"\n  {'Mode':<15} {'Trainable':<12} {'Learning':<12} {'Max Change':<15} {'Final Loss':<12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*15} {'-'*12}")

    sixt1c_train_status = "✓ Yes" if sixt1c_trainable else "✗ No"
    sixt1c_learn_status = "✓ Yes" if sixt1c_learning else "✗ No"
    print(f"  {'Sixt1c':<15} {sixt1c_train_status:<12} {sixt1c_learn_status:<12} {sixt1c_change:<15.6f} {sixt1c_loss:<12.4f}")

    fp_train_status = "✓ Yes" if fp_trainable else "✗ No"
    fp_learn_status = "✓ Yes" if fp_learning else "✗ No"
    print(f"  {'FP':<15} {fp_train_status:<12} {fp_learn_status:<12} {fp_change:<15.6f} {fp_loss:<12.4f}")

    print("\n  Detailed Results:")
    print(f"    Sixt1c Mode:")
    print(f"      - Classifier trainable: {sixt1c_trainable}")
    print(f"      - Classifier learning: {sixt1c_learning}")
    print(f"      - Max weight change: {sixt1c_change:.6f}")
    print(f"      - Final loss: {sixt1c_loss:.4f}")

    print(f"\n    FP Mode:")
    print(f"      - Classifier trainable: {fp_trainable}")
    print(f"      - Classifier learning: {fp_learning}")
    print(f"      - Max weight change: {fp_change:.6f}")
    print(f"      - Final loss: {fp_loss:.4f}")

    # Overall verdict
    print("\n" + "="*80)

    all_good = sixt1c_trainable and sixt1c_learning and fp_trainable and fp_learning

    if all_good:
        print("  ✓✓✓ ALL TESTS PASSED!")
        print("  Classifier is properly trainable and learning in both modes!")
    else:
        print("  ✗✗✗ SOME TESTS FAILED!")
        if not sixt1c_trainable:
            print("    - Sixt1c classifier not trainable")
        if not sixt1c_learning:
            print("    - Sixt1c classifier not learning")
        if not fp_trainable:
            print("    - FP classifier not trainable")
        if not fp_learning:
            print("    - FP classifier not learning")

    print("="*80 + "\n")

    return all_good


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
