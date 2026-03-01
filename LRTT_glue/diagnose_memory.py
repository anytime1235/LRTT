"""Diagnose GPU memory usage at each stage of model setup."""
import sys
sys.path.insert(0, '/data/LRTT_transformer/src')
import os, math
import torch, torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, set_seed
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.devices import (
    FloatingPointDevice, LinearStepDevice, SoftBoundsDevice)
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

MODEL_NAME = "google/mobilebert-uncased"
RANK = 8
TARGETS = ["query", "key", "value"]

def gpu_mem():
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        return f"alloc={a:.2f}GB reserved={r:.2f}GB"
    return "no CUDA"

def create_config(ab_type, alpha):
    if ab_type == "fp":
        ab = FloatingPointDevice()
    else:
        TAU = 46505.0; d = 1 - math.exp(-1.0/TAU); lt = 1.0/d
        ab = LinearStepDevice(
            dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
            gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
            dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
            gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3,
            write_noise_std=0.0, mean_bound_reference=True,
            lifetime=lt, lifetime_dtod=0.0, reset=0.0, reset_dtod=0.0)
    c = SoftBoundsDevice(dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0, up_down_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0, write_noise_std=0.0, mult_noise=True)
    dc = PythonLRTTDevice(rank=RANK, transfer_every=1000000,
        lora_alpha=alpha, reinit_gain=0.1, reinit_mode="hybrid",
        decay_factor=1.0, unit_cell_devices=[ab, ab, c])
    dc.transfer_lr=0.001; dc.units_in_mbatch=True; dc.forward_inject=True
    dc.transfer_method="onehot"; dc.update_mode="lora"; dc.a_init_mode="zero"
    rpu = PythonLRTTRPUConfig(device=dc)
    rpu.mapping.weight_scaling_omega=1.0; rpu.mapping.weight_scaling_columnwise=True
    rpu.mapping.learn_out_scaling=True; rpu.mapping.out_scaling_columnwise=True
    return rpu

ab_type = sys.argv[1] if len(sys.argv) > 1 else "sixt1c"
dev = torch.device('cuda')
set_seed(42)

print(f"[0] Before model load: {gpu_mem()}", flush=True)

cfg = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=cfg)

# Count linear layers
allL = [(n,m) for n,m in model.named_modules() if isinstance(m, nn.Linear)]
print(f"\nAll linear layers ({len(allL)}):")
total_params = 0
for n, m in allL:
    p = m.weight.numel()
    total_params += p
    converted = any(t in n for t in TARGETS) and "classifier" not in n
    print(f"  {'>>>' if converted else '   '} {n}: {m.weight.shape} = {p:,} params")
print(f"\nTotal linear params: {total_params:,}")
print(f"Params in converted layers: {sum(m.weight.numel() for n,m in allL if any(t in n for t in TARGETS) and 'classifier' not in n):,}")

model.to(dev)
print(f"\n[1] After model.to(cuda): {gpu_mem()}", flush=True)

exc = [n for n,_ in allL if not any(t in n for t in TARGETS)]; exc.append("classifier")
model = convert_to_analog(model, create_config(ab_type, 8.0), exclude_modules=exc)
print(f"\n[2] After convert_to_analog (on CPU): {gpu_mem()}", flush=True)

model.to(dev)
print(f"\n[3] After analog model.to(cuda): {gpu_mem()}", flush=True)
torch.cuda.empty_cache()
print(f"[3b] After empty_cache: {gpu_mem()}", flush=True)

# Count converted modules
n_analog = sum(1 for _,m in model.named_modules() if hasattr(m, 'analog_module'))
print(f"\nAnalog modules: {n_analog}")
print(f"Total model params: {sum(p.numel() for p in model.parameters()):,}")
