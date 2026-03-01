"""
Test Digital LoRA first (before analog conversion)
Check if PEFT LoRA itself works correctly
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model

def test_digital_lora():
    """Test pure digital LoRA (no analog conversion)."""
    print("=" * 80)
    print("  DIGITAL LORA TEST (No Analog Conversion)")
    print("=" * 80)

    device = torch.device("cuda")
    model_name = "google/mobilebert-uncased"

    # Load base model
    print("\n[1] Loading base model...")
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # Apply LoRA
    print("\n[2] Applying LoRA...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=0.01,
        lora_dropout=0.0,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model = model.to(device)

    print(model.print_trainable_parameters())

    # Check LoRA weights
    print("\n[3] Checking LoRA weights...")
    lora_a_weights = []
    lora_b_weights = []

    for name, param in model.named_parameters():
        if 'lora_A' in name and 'weight' in name:
            lora_a_weights.append((name, param.data))
        elif 'lora_B' in name and 'weight' in name:
            lora_b_weights.append((name, param.data))

    print(f"\nLoRA A layers: {len(lora_a_weights)}")
    if lora_a_weights:
        name, w = lora_a_weights[0]
        print(f"  {name}")
        print(f"  Shape: {w.shape}")
        print(f"  Mean: {w.mean().item():.6f}, Std: {w.std().item():.6f}")
        print(f"  Max: {w.abs().max().item():.6f}")

    print(f"\nLoRA B layers: {len(lora_b_weights)}")
    if lora_b_weights:
        name, w = lora_b_weights[0]
        print(f"  {name}")
        print(f"  Shape: {w.shape}")
        print(f"  Mean: {w.mean().item():.6f}, Std: {w.std().item():.6f}")
        print(f"  Max: {w.abs().max().item():.6f}")

    # Test forward pass
    print("\n[4] Testing forward pass...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "This movie is great!"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                      max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        output_with_lora = model(**inputs).logits

    print(f"\nOutput with digital LoRA:")
    print(f"  {output_with_lora}")

    # Disable LoRA
    for name, module in model.named_modules():
        if hasattr(module, 'scaling') and isinstance(module.scaling, dict):
            module.scaling['default'] = 0.0

    with torch.no_grad():
        output_without_lora = model(**inputs).logits

    # Restore
    for name, module in model.named_modules():
        if hasattr(module, 'scaling') and isinstance(module.scaling, dict):
            module.scaling['default'] = 0.01 / 8

    print(f"\nOutput WITHOUT LoRA:")
    print(f"  {output_without_lora}")

    diff = (output_with_lora - output_without_lora).abs()
    print(f"\nDifference:")
    print(f"  Max: {diff.max().item():.6f}")
    print(f"  Mean: {diff.mean().item():.6f}")

    # Test training
    print("\n[5] Testing training...")
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Get initial lora_B weight
    lora_b_name, lora_b_init = lora_b_weights[0]
    lora_b_init = lora_b_init.clone()

    for i in range(10):
        optimizer.zero_grad()
        labels = torch.tensor([1]).to(device)
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

    print(f"Final loss: {loss.item():.4f}")

    # Check if lora_B changed
    for name, param in model.named_parameters():
        if name == lora_b_name:
            lora_b_final = param.data
            break

    diff = (lora_b_final - lora_b_init).abs().max().item()
    print(f"\nLoRA B weight change: {diff:.6f}")

    print("\n" + "=" * 80)
    if diff > 1e-6:
        print("✓ Digital LoRA works correctly!")
    else:
        print("✗ Digital LoRA has issues!")
    print("=" * 80)


if __name__ == "__main__":
    test_digital_lora()
