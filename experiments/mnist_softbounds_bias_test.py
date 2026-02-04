#!/usr/bin/env python3
"""
SoftBounds with bias=True test
Same conditions as previous: rank=32, te=1, lifetime~46505
"""

import os
os.environ["LRTT_SILENT"] = "1"

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from time import time

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import FloatingPointRPUConfig, SoftBoundsDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset

RANK = 32
TE = 1
LR = 0.089054
TLR = 0.001277
DT_BATCH_SEC = 1.0  # lifetime ~ 46505

BATCH_SIZE = 64
EPOCHS = 30
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
}

print("=" * 70)
print("SoftBounds with BIAS=True")
print(f"rank={RANK}, te={TE}, lr={LR}, tlr={TLR}")
print(f"dt_batch_sec={DT_BATCH_SEC} (lifetime ~ 46505)")
print("=" * 70)

c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

device_config = PythonLRTTPreset.sixt1c_ab(
    rank=RANK,
    transfer_every=TE,
    lora_alpha=1.0,
    dt_batch_sec=DT_BATCH_SEC,
    include_retention=True,
    c_device=c_device,
    reinit_mode="decay",
    decay_factor=1.0,
)
device_config.transfer_lr = TLR
device_config.reinit_gain = 0.1
device_config.forward_inject = False
device_config.update_mode = "lora"
device_config.transfer_mode = "off"

rpu_config = PythonLRTTRPUConfig(device=device_config)

print(f"Actual lifetime: {device_config.unit_cell_devices[0].lifetime:.2f}")

# Model with BIAS=True for first layer
model = AnalogSequential(
    AnalogLinear(784, 256, bias=True, rpu_config=rpu_config),  # BIAS=True
    nn.ReLU(),
    AnalogLinear(256, 10, bias=True, rpu_config=FloatingPointRPUConfig()),
    nn.LogSoftmax(dim=1),
)
model.to(DEVICE)
print(f"Model created on {DEVICE}")
print(f"First layer bias: {model[0].bias is not None}")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST("/tmp/mnist", download=True, train=True, transform=transform)
val_set = datasets.MNIST("/tmp/mnist", download=True, train=False, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

optimizer = AnalogSGD(model.parameters(), lr=LR)
optimizer.regroup_param_groups(model)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.NLLLoss()

best_val_acc = 0.0
start_time = time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    for data, target in train_loader:
        data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
        target = target.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
            target = target.to(DEVICE, non_blocking=True)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    val_acc = 100.0 * correct / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    scheduler.step()
    print(f"Epoch {epoch}/{EPOCHS}: {val_acc:.2f}% (best={best_val_acc:.2f}%)")

print(f"\n>>> BIAS=True: best_acc={best_val_acc:.2f}%")
print(f"Total time: {time() - start_time:.1f}s")
print("\n비교: BIAS=False (이전 실험): 96.81%")
