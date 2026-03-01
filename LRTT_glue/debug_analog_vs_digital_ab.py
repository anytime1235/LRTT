"""Compare analog vs digital (floating point) A/B tiles under same conditions."""
import sys
sys.path.insert(0, '/data/LRTT_transformer/src')

import torch
import torch.nn as nn
import math
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice, FloatingPointDevice
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.optim import AnalogSGD

MODEL_NAME = "google/mobilebert-uncased"
RANK = 8
NUM_STEPS = 10
LR = 0.001


def create_config(rank, lora_alpha, mode="analog"):
    """Create config. mode='analog' uses LinearStepDevice, mode='digital' uses FloatingPointDevice."""
    TAU_SEC = 46505.0
    delta = 1 - math.exp(-1.0 / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

    if mode == "analog":
        ab_device = LinearStepDevice(
            dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
            gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
            dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
            gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3,
            write_noise_std=0.0, mean_bound_reference=True,
            lifetime=lifetime, lifetime_dtod=0.0, reset=0.0, reset_dtod=0.0,
        )
    else:
        ab_device = FloatingPointDevice()

    c_device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=True,
    )
    device_config = PythonLRTTDevice(
        rank=rank, transfer_every=1000000,
        lora_alpha=lora_alpha, reinit_gain=0.1,
        reinit_mode="hybrid", decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.001
    device_config.units_in_mbatch = True
    device_config.forward_inject = True
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


def build_model(lora_alpha, mode, device_hw):
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    exclude = [n for n in all_linear if 'query' not in n]
    exclude.append('classifier')

    rpu_config = create_config(RANK, lora_alpha, mode)
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        if 'query' in name and 'bias' not in name:
            param.requires_grad = True
        elif 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model.to(device_hw)


def install_hooks(model, hook_data, lora_alpha):
    for name, module in model.named_modules():
        if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'controller'):
            ctrl = module.analog_module.controller
            orig_fn = ctrl._forward_inject_analog_unified
            lname = name

            def make_patched(ctrl, orig, ln, alpha):
                def patched(x, out_trans=False, in_trans=False):
                    x_bf = x.t() if in_trans else x
                    with torch.no_grad():
                        y_c = ctrl.tile_c.forward(x_bf)
                        y_b = ctrl.tile_b.forward(x_bf)
                        y_ab = ctrl.tile_a.forward(y_b)
                    hook_data[ln] = {'y_c': y_c.detach(), 'y_ab': y_ab.detach(), 'x': x_bf.detach()}
                    return orig(x, out_trans, in_trans)
                return patched

            ctrl._forward_inject_analog_unified = make_patched(ctrl, orig_fn, lname, lora_alpha)


def get_layer0_query(model):
    for name, module in model.named_modules():
        if 'layer.0' in name and 'query' in name and hasattr(module, 'analog_module'):
            return name, module.analog_module
    return None, None


def report(model, hook_data, batch, lora_alpha, device_hw, step_label):
    model.eval()
    hook_data.clear()
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device_hw)
        attention_mask = batch['attention_mask'].to(device_hw)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    lname, analog = get_layer0_query(model)
    if lname not in hook_data:
        return

    d = hook_data[lname]
    y_c = d['y_c']
    y_ab = d['y_ab']
    alpha = lora_alpha

    a_w = analog.tile_a.get_weights()[0]
    b_w = analog.tile_b.get_weights()[0]
    c_w = analog.tile_c.get_weights()[0]
    ab_w = a_w @ b_w

    ratio = (alpha * y_ab).norm().item() / max(y_c.norm().item(), 1e-10)

    print(f"  [{step_label}]")
    print(f"    Forward:  C_norm={y_c.norm():.2f}  AB_norm={y_ab.norm():.2f}  "
          f"α*AB_norm={(alpha*y_ab).norm():.2f}  total={(y_c+alpha*y_ab).norm():.2f}  "
          f"ratio={ratio:.4f}")
    print(f"    Weights:  A_norm={a_w.norm():.4f} A_max={a_w.abs().max():.4f} A_std={a_w.std():.6f}  "
          f"B_norm={b_w.norm():.4f} B_max={b_w.abs().max():.4f} B_std={b_w.std():.6f}")
    print(f"              C_norm={c_w.norm():.4f}  AB_product_norm={ab_w.norm():.4f}  "
          f"AB_max={ab_w.abs().max():.4f}  AB_std={ab_w.std():.6f}")
    print(f"    Logits:   norm={logits.norm():.4f}  mean={logits.mean():.4f}  std={logits.std():.4f}")

    model.train()
    return {
        'ratio': ratio,
        'a_norm': a_w.norm().item(),
        'b_norm': b_w.norm().item(),
        'ab_norm': ab_w.norm().item(),
        'a_max': a_w.abs().max().item(),
        'b_max': b_w.abs().max().item(),
    }


def run_experiment(lora_alpha, mode, device_hw, loader, first_batch):
    print(f"\n  --- {mode.upper()} AB, alpha={lora_alpha} ---")

    model = build_model(lora_alpha, mode, device_hw)
    hook_data = {}
    install_hooks(model, hook_data, lora_alpha)

    optimizer = AnalogSGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    report(model, hook_data, first_batch, lora_alpha, device_hw, "INIT")

    # Store initial weights
    _, analog = get_layer0_query(model)
    init_a = analog.tile_a.get_weights()[0].clone()
    init_b = analog.tile_b.get_weights()[0].clone()

    model.train()
    losses = []
    for step, batch in enumerate(loader):
        if step >= NUM_STEPS:
            break
        input_ids = batch['input_ids'].to(device_hw)
        attention_mask = batch['attention_mask'].to(device_hw)
        labels = batch['labels'].to(device_hw)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if step in [0, 4, 9]:
            r = report(model, hook_data, first_batch, lora_alpha, device_hw, f"Step {step+1}")

    # Final weight change
    cur_a = analog.tile_a.get_weights()[0]
    cur_b = analog.tile_b.get_weights()[0]
    print(f"    Weight Δ: A_change={( cur_a - init_a).norm():.4f}  B_change={(cur_b - init_b).norm():.4f}")
    print(f"    Loss:     start={losses[0]:.4f}  end={losses[-1]:.4f}  "
          f"change={losses[-1]-losses[0]:.4f}")


def main():
    device_hw = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    raw = load_dataset("glue", "rte", split="train[:64]")
    def preprocess(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"],
                        truncation=True, padding="max_length", max_length=128)
    tokenized = raw.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    loader = DataLoader(tokenized, batch_size=16, shuffle=False, collate_fn=default_data_collator)
    first_batch = next(iter(loader))

    for alpha in [0.01, 0.1, 1.0, 5.0]:
        print(f"\n{'='*70}")
        print(f"LORA_ALPHA = {alpha}")
        print(f"{'='*70}")
        for mode in ["analog", "digital"]:
            run_experiment(alpha, mode, device_hw, loader, first_batch)


if __name__ == "__main__":
    main()
