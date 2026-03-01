#!/usr/bin/env python3
"""Compare digital vs analog model forward outputs to check if conversion preserves behavior."""
import os, sys, torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, set_seed
from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.devices import SoftBoundsDevice, LinearStepDevice
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

MODEL_NAME = "albert/albert-base-v2"
SEED = 42

def _create_c_device():
    return SoftBoundsDevice(dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=False)

def _create_ab_device():
    return LinearStepDevice(dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True, dw_min_dtod=0.1,
        up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3,
        write_noise_std=0.0, mean_bound_reference=True)

def create_lrtt_config(rank=8, lora_alpha=0.061):
    ab_device = _create_ab_device()
    c_device = _create_c_device()
    device_config = PythonLRTTDevice(
        rank=rank, transfer_every=10000000, lora_alpha=lora_alpha,
        reinit_gain=1.0, reinit_mode="decay", decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device])
    device_config.transfer_lr = 0.1
    device_config.units_in_mbatch = True
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = True
    device_config.combined_out_scaling = False
    device_config.dynamic_te = False
    device_config.dynamic_te_power = 1.0
    device_config.dynamic_te_max = 10000000 * 20
    device_config.te_warmup_schedule = [10000000]
    device_config.te_warmup_steps = 0
    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = False
    rpu_config.mapping.out_scaling_columnwise = False
    return rpu_config

def _create_nontarget_rpu_config():
    from aihwkit.simulator.configs import SingleRPUConfig
    device = SoftBoundsDevice(dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=False)
    rpu_config = SingleRPUConfig(device=device)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = False
    rpu_config.mapping.out_scaling_columnwise = False
    return rpu_config

def list_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


# --- Digital model ---
set_seed(SEED)
digital_model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
digital_model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
inputs = tokenizer("What is artificial intelligence?",
                   "Artificial intelligence is a branch of computer science that aims to create intelligent machines.",
                   return_tensors="pt", max_length=128, truncation=True, padding="max_length")
inputs["start_positions"] = torch.tensor([10])
inputs["end_positions"] = torch.tensor([15])

with torch.no_grad():
    digital_out = digital_model(**{k: v for k, v in inputs.items() if k != 'start_positions' and k != 'end_positions'})
    digital_loss_out = digital_model(**inputs)

print("=" * 70)
print("Digital model output")
print("=" * 70)
print(f"  start_logits: mean={digital_out.start_logits.mean():.6f}, std={digital_out.start_logits.std():.6f}")
print(f"  end_logits:   mean={digital_out.end_logits.mean():.6f}, std={digital_out.end_logits.std():.6f}")
print(f"  loss: {digital_loss_out.loss.item():.4f}")
print(f"  start_logits[:10]: {digital_out.start_logits[0,:10].tolist()}")
print(f"  end_logits[:10]:   {digital_out.end_logits[0,:10].tolist()}")

# --- Analog model ---
set_seed(SEED)
analog_model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

lrtt_patterns = ["attention"]
always_digital = ["qa_outputs", "albert.encoder.embedding_hidden_mapping_in"]

def is_lrtt_target(layer_name):
    if any(d in layer_name for d in always_digital):
        return False
    if "encoder" not in layer_name:
        return False
    return any(p in layer_name for p in lrtt_patterns)

all_linear_names = list_linear_layers(analog_model)
exclude_modules = [n for n in all_linear_names if not is_lrtt_target(n)]
exclude_modules += always_digital
exclude_modules = list(set(exclude_modules))

# Pass 1: LRTT
lrtt_config = create_lrtt_config()
analog_model = convert_to_analog(analog_model, lrtt_config, exclude_modules=exclude_modules)

# Pass 2: NT (inplace)
nt_encoder_layers = [n for n in all_linear_names
    if not is_lrtt_target(n) and "encoder" in n
    and not any(d in n for d in always_digital)]
nontarget_config = _create_nontarget_rpu_config()
exclude_pass2 = [n for n in all_linear_names if n not in nt_encoder_layers]
analog_model = convert_to_analog(analog_model, nontarget_config, exclude_modules=exclude_pass2,
                                  inplace=True, ensure_analog_root=False)

analog_model.eval()

with torch.no_grad():
    analog_out = analog_model(**{k: v for k, v in inputs.items() if k != 'start_positions' and k != 'end_positions'})
    analog_loss_out = analog_model(**inputs)

print(f"\n{'=' * 70}")
print("Analog model output (LRTT attn + NT ffn)")
print("=" * 70)
print(f"  start_logits: mean={analog_out.start_logits.mean():.6f}, std={analog_out.start_logits.std():.6f}")
print(f"  end_logits:   mean={analog_out.end_logits.mean():.6f}, std={analog_out.end_logits.std():.6f}")
print(f"  loss: {analog_loss_out.loss.item():.4f}")
print(f"  start_logits[:10]: {analog_out.start_logits[0,:10].tolist()}")
print(f"  end_logits[:10]:   {analog_out.end_logits[0,:10].tolist()}")

# --- Comparison ---
print(f"\n{'=' * 70}")
print("Comparison")
print("=" * 70)
start_diff = (digital_out.start_logits - analog_out.start_logits).abs()
end_diff = (digital_out.end_logits - analog_out.end_logits).abs()
print(f"  start_logits max diff: {start_diff.max():.6f}, mean diff: {start_diff.mean():.6f}")
print(f"  end_logits   max diff: {end_diff.max():.6f}, mean diff: {end_diff.mean():.6f}")
print(f"  loss diff: {abs(digital_loss_out.loss.item() - analog_loss_out.loss.item()):.6f}")

corr_start = torch.corrcoef(torch.stack([digital_out.start_logits.flatten(), analog_out.start_logits.flatten()]))[0,1]
corr_end = torch.corrcoef(torch.stack([digital_out.end_logits.flatten(), analog_out.end_logits.flatten()]))[0,1]
print(f"  start_logits correlation: {corr_start:.6f}")
print(f"  end_logits correlation:   {corr_end:.6f}")

if corr_start > 0.99 and corr_end > 0.99:
    print("\n  PASS: Analog model preserves pretrained behavior")
else:
    print("\n  FAIL: Analog conversion broke pretrained behavior!")
