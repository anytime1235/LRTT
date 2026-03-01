"""Debug STS-B: check model output and loss on first batch."""
import sys
sys.path.insert(0, '/data/LRTT_transformer/src')

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, default_data_collator
from datasets import load_dataset
from torch.utils.data import DataLoader

MODEL_NAME = "google/mobilebert-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Check raw digital model first
print("=== Digital Model (no analog) ===")
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=1)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
raw = load_dataset("nyu-mll/glue", "stsb")

def preprocess(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"],
                     padding="max_length", max_length=128, truncation=True)

tokenized = raw.map(preprocess, batched=True)
tokenized = tokenized.rename_column("label", "labels")
loader = DataLoader(tokenized["train"], batch_size=8, shuffle=False, collate_fn=default_data_collator)

batch = next(iter(loader))
input_ids = batch['input_ids'].to(device)
attention_mask = batch['attention_mask'].to(device)
labels = batch['labels'].to(device).float()

print(f"Labels dtype: {labels.dtype}, shape: {labels.shape}")
print(f"Labels values: {labels[:8]}")
print(f"Labels min/max: {labels.min():.2f} / {labels.max():.2f}")

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits.squeeze()
    print(f"Logits shape: {logits.shape}")
    print(f"Logits values: {logits[:8]}")
    print(f"Logits min/max: {logits.min():.4f} / {logits.max():.4f}")

    loss = nn.MSELoss()(logits, labels)
    print(f"MSELoss (digital): {loss.item():.4f}")

# 2. Now check with analog conversion
print("\n=== Analog Model (TikiTaka v2) ===")
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import UnitCellRPUConfig, IOParameters, UpdateParameters, NoiseManagementType, BoundManagementType
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsReferenceDevice

config2 = AutoConfig.from_pretrained(MODEL_NAME, num_labels=1)
model2 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config2)

# Same config as sweep script
sixt1c_device = LinearStepDevice(dw_min=0.001981, gamma_up=-0.1678, gamma_down=0.1410,
    dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
    gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3, write_noise_std=0.0,
    mult_noise=True, mean_bound_reference=True, lifetime=0.0)

softbounds_device = SoftBoundsReferenceDevice(w_max=1.0, w_min=-1.0, dw_min=0.001,
    dw_min_std=0.0, write_noise_std=0.0, diffusion=0.0, dw_min_dtod=0.0,
    w_max_dtod=0.0, w_min_dtod=0.0, up_down=0.0, up_down_dtod=0.0,
    lifetime=0.0, lifetime_dtod=0.0, slope_up_dtod=0.0, slope_down_dtod=0.0)

rpu_config = UnitCellRPUConfig(
    device=ChoppedTransferCompound(
        unit_cell_devices=[sixt1c_device, softbounds_device],
        transfer_every=491, units_in_mbatch=False, n_reads_per_transfer=1,
        transfer_columns=True, gamma=0.0, transfer_lr=3.986, fast_lr=3.0,
        scale_transfer_lr=True, auto_scale=True, auto_granularity=169.0,
        buffer_granularity=1.0, auto_momentum=0.99, in_chop_prob=0.061,
        in_chop_random=True,
        transfer_forward=IOParameters(noise_management=NoiseManagementType.NONE, bound_management=BoundManagementType.NONE),
        transfer_update=UpdateParameters(desired_bl=1, update_bl_management=False, update_management=False),
    ))
rpu_config.forward = IOParameters(is_perfect=False, inp_noise=0.0, out_noise=0.0, out_noise_std=0.0, w_noise=0.0,
    noise_management=NoiseManagementType.ABS_MAX, bound_management=BoundManagementType.ITERATIVE)
rpu_config.backward = IOParameters(is_perfect=False, inp_noise=0.0, out_noise=0.0, out_noise_std=0.0, w_noise=0.0,
    noise_management=NoiseManagementType.ABS_MAX, bound_management=BoundManagementType.ITERATIVE)
rpu_config.mapping.digital_bias = True
rpu_config.mapping.weight_scaling_omega = 1.0
rpu_config.mapping.weight_scaling_columnwise = True
rpu_config.mapping.learn_out_scaling = True
rpu_config.mapping.out_scaling_columnwise = True

all_linear = [name for name, m in model2.named_modules() if isinstance(m, nn.Linear)]
TARGET_MODULES = ["query", "key", "value"]
exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
exclude.append("classifier")

model2 = convert_to_analog(model2, rpu_config, exclude_modules=exclude)
model2 = model2.to(device)

with torch.no_grad():
    outputs2 = model2(input_ids=input_ids, attention_mask=attention_mask)
    logits2 = outputs2.logits.squeeze()
    print(f"Logits shape: {logits2.shape}")
    print(f"Logits values: {logits2[:8]}")
    print(f"Logits min/max: {logits2.min():.4f} / {logits2.max():.4f}")

    loss2 = nn.MSELoss()(logits2, labels)
    print(f"MSELoss (analog): {loss2.item():.4f}")

# 3. Check HF internal loss
print("\n=== HF Internal Loss ===")
with torch.no_grad():
    outputs3 = model2(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(f"HF loss: {outputs3.loss.item() if outputs3.loss is not None else 'None'}")
