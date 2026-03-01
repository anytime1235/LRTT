"""
Test base MobileBERT model (no LoRA)
"""

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda")
model_name = "google/mobilebert-uncased"

print("Testing BASE MobileBERT model (no LoRA)")
print("=" * 60)

# Load model
config = AutoConfig.from_pretrained(model_name, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
model = model.to(device)
model.eval()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test input
text = "This movie is great!"
inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                  max_length=128, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"\nInput: {text}")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

print(f"\nLogits: {logits}")
print(f"Mean: {logits.mean().item():.4f}")
print(f"Std: {logits.std().item():.4f}")

# Check if values are reasonable
reasonable = logits.abs().max().item() < 100

print(f"\n{'✓' if reasonable else '✗'} Logits are reasonable: {reasonable}")

# Test with different input
text2 = "This movie is terrible!"
inputs2 = tokenizer(text2, return_tensors="pt", padding="max_length",
                   max_length=128, truncation=True)
inputs2 = {k: v.to(device) for k, v in inputs2.items()}

with torch.no_grad():
    outputs2 = model(**inputs2)
    logits2 = outputs2.logits

print(f"\nInput 2: {text2}")
print(f"Logits 2: {logits2}")

print("\n" + "=" * 60)
