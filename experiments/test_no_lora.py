"""
Test if explosion happens even without LoRA.
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def test_vanilla_model():
    """Test vanilla MobileBERT without any LoRA."""
    print(f"\n  Testing vanilla MobileBERT (no LoRA)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # Load vanilla model
    model_config = AutoConfig.from_pretrained("google/mobilebert-uncased", num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased", config=model_config)

    # Reinitialize classifier
    nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
    if model.classifier.bias is not None:
        nn.init.zeros_(model.classifier.bias)

    model = model.to(device)
    model.eval()

    # Prepare input
    text = "This movie is great!"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                      max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden_state = outputs.hidden_states[-1]
        logits = outputs.logits

    max_hidden = last_hidden_state.abs().max().item()
    max_logit = logits.abs().max().item()

    print(f"\n  Vanilla MobileBERT:")
    print(f"    Encoder max: {max_hidden:.2f}")
    print(f"    Logit max: {max_logit:.2f}")

    if max_hidden > 1000:
        print(f"    ✗ EXPLOSION in vanilla model!")
    else:
        print(f"    ✓ NO explosion - values are normal")

    return max_hidden, max_logit


def test_with_lora_no_conversion():
    """Test with LoRA but WITHOUT analog conversion."""
    print(f"\n  Testing with LoRA (no analog conversion)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    from peft import LoraConfig, get_peft_model

    # Load base model
    model_config = AutoConfig.from_pretrained("google/mobilebert-uncased", num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased", config=model_config)

    # Reinitialize classifier
    nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
    if model.classifier.bias is not None:
        nn.init.zeros_(model.classifier.bias)

    # Apply LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=1.0,
        lora_dropout=0.0,
        target_modules=["value"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    model = model.to(device)
    model.eval()

    # Prepare input
    text = "This movie is great!"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                      max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model.base_model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden_state = outputs.hidden_states[-1]
        logits = model(**inputs).logits

    max_hidden = last_hidden_state.abs().max().item()
    max_logit = logits.abs().max().item()

    print(f"\n  LoRA (no analog):")
    print(f"    Encoder max: {max_hidden:.2f}")
    print(f"    Logit max: {max_logit:.2f}")

    if max_hidden > 1000:
        print(f"    ✗ EXPLOSION with LoRA!")
    else:
        print(f"    ✓ NO explosion - values are normal")

    return max_hidden, max_logit


def main():
    print("="*80)
    print("  TESTING: Does explosion happen without analog conversion?")
    print("="*80)

    vanilla_h, vanilla_l = test_vanilla_model()

    print("\n" + "-"*80)

    lora_h, lora_l = test_with_lora_no_conversion()

    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)

    print(f"\n  {'Setup':<30} {'Encoder Max':<15} {'Logit Max':<15} {'Status':<10}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*10}")

    v_status = "OK" if vanilla_h < 1000 else "EXPLODED"
    print(f"  {'Vanilla MobileBERT':<30} {vanilla_h:<15.2f} {vanilla_l:<15.2f} {v_status:<10}")

    l_status = "OK" if lora_h < 1000 else "EXPLODED"
    print(f"  {'LoRA (no analog)':<30} {lora_h:<15.2f} {lora_l:<15.2f} {l_status:<10}")

    print("\n" + "="*80)

    if vanilla_h < 1000 and lora_h > 1000:
        print("\n  ⚠️  Explosion starts AFTER applying LoRA!")
        print("      Problem is likely in PEFT LoRA implementation or base_layer initialization")
    elif vanilla_h > 1000:
        print("\n  ⚠️  Explosion in vanilla model!")
        print("      Problem is in pretrained MobileBERT or how it's loaded")
    else:
        print("\n  ✓ No explosion in either setup - problem is analog-specific")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
