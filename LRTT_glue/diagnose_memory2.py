"""Diagnose GPU memory at CUDA level (nvidia-smi) at each stage."""
import sys
sys.path.insert(0, '/data/LRTT_transformer/src')
import os, math, subprocess
import torch, torch.nn as nn
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, default_data_collator, set_seed)
from datasets import load_dataset
from torch.utils.data import DataLoader
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import (
    FloatingPointDevice, LinearStepDevice, SoftBoundsDevice)
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

MODEL_NAME = "google/mobilebert-uncased"
RANK = 8
TARGETS = ["query", "key", "value"]

def gpu_mem():
    """Get CUDA-level memory from nvidia-smi for current process."""
    pid = os.getpid()
    torch.cuda.synchronize()
    r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True)
    for line in r.stdout.strip().split('\n'):
        if str(pid) in line:
            parts = line.split(',')
            return f"nvidia-smi={parts[1].strip()}MiB"
    # Also get pytorch level
    a = torch.cuda.memory_allocated() / 1024**2
    r2 = torch.cuda.memory_reserved() / 1024**2
    return f"nvidia-smi=0MiB (not visible), pytorch_alloc={a:.0f}MiB"

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
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 8
dev = torch.device('cuda')
set_seed(42)

# Warmup CUDA context
_ = torch.zeros(1, device=dev)
print(f"[0] CUDA context init: {gpu_mem()}", flush=True)

cfg = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=cfg)
model.to(dev)
print(f"[1] Digital model on GPU: {gpu_mem()}", flush=True)

exc_list = [n for n,m in model.named_modules() if isinstance(m, nn.Linear)
            and not any(t in n for t in TARGETS)]
exc_list.append("classifier")
model = convert_to_analog(model, create_config(ab_type, 8.0), exclude_modules=exc_list)
print(f"[2] After convert_to_analog (CPU tiles): {gpu_mem()}", flush=True)

model.to(dev)
torch.cuda.synchronize()
print(f"[3] Analog model on GPU: {gpu_mem()}", flush=True)

# Prepare data
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
raw = load_dataset("nyu-mll/glue", "mrpc")
def pp(ex): return tok(ex["sentence1"], ex["sentence2"], padding="max_length", max_length=128, truncation=True)
td = raw.map(pp, batched=True).rename_column("label", "labels")
dl = DataLoader(td["train"], batch_size=BATCH, shuffle=False, collate_fn=default_data_collator)
batch = next(iter(dl))

# Forward pass (eval, no grad)
model.eval()
with torch.no_grad():
    o = model(input_ids=batch['input_ids'].to(dev), attention_mask=batch['attention_mask'].to(dev))
torch.cuda.synchronize()
print(f"[4] After eval forward (batch={BATCH}, no_grad): {gpu_mem()}", flush=True)
del o

# Forward + backward (train)
model.train()
o = model(input_ids=batch['input_ids'].to(dev), attention_mask=batch['attention_mask'].to(dev))
torch.cuda.synchronize()
print(f"[5] After train forward (batch={BATCH}): {gpu_mem()}", flush=True)

loss = nn.CrossEntropyLoss()(o.logits, batch['labels'].to(dev))
loss.backward()
torch.cuda.synchronize()
print(f"[6] After backward: {gpu_mem()}", flush=True)

# Optimizer step
opt = AnalogAdam(model.parameters(), lr=0.001); opt.regroup_param_groups()
opt.step()
torch.cuda.synchronize()
print(f"[7] After optimizer step: {gpu_mem()}", flush=True)

print("\nDone!", flush=True)
