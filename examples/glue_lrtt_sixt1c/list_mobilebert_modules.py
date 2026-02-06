#!/usr/bin/env python
"""List all MobileBERT module names for exclude_modules."""

from transformers import AutoConfig, AutoModelForSequenceClassification
import torch.nn as nn

model_name = "google/mobilebert-uncased"
model_config = AutoConfig.from_pretrained(model_name, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, config=model_config, use_safetensors=True
)

print("="*80)
print("LINEAR LAYERS (will be converted by LRTT)")
print("="*80)

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"{name}")
