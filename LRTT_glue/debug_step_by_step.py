"""Step-by-step comparison: analog vs digital AB tile evolution."""
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
NUM_STEPS = 50
LR = 0.001


def create_config(rank, lora_alpha, mode="analog"):
    TAU_SEC = 46505.0
    delta = 1 - math.exp(-1.0 / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

    if mode == "analog":
        ab_device = LinearStepDevice(
            dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
            gamma_up=-0.1678, gamma_down=0.1410, mult_noise=False,
            dw_min_dtod=0.0, up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
            gamma_up_dtod=0.0, gamma_down_dtod=0.0, dw_min_std=0.0,
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
    rpu_config.backward.out_noise = 0.0
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

            def make_patched(ctrl, orig, ln):
                def patched(x, out_trans=False, in_trans=False):
                    x_bf = x.t() if in_trans else x
                    with torch.no_grad():
                        y_c = ctrl.tile_c.forward(x_bf)
                        y_b = ctrl.tile_b.forward(x_bf)
                        y_ab = ctrl.tile_a.forward(y_b)
                    hook_data[ln] = {'y_c': y_c.detach(), 'y_ab': y_ab.detach()}
                    return orig(x, out_trans, in_trans)
                return patched

            ctrl._forward_inject_analog_unified = make_patched(ctrl, orig_fn, lname)


def get_layer0(model):
    for name, module in model.named_modules():
        if 'layer.0' in name and 'query' in name and hasattr(module, 'analog_module'):
            return name, module.analog_module
    return None, None


def collect_metrics(model, hook_data, batch, lora_alpha, device_hw):
    model.eval()
    hook_data.clear()
    with torch.no_grad():
        ids = batch['input_ids'].to(device_hw)
        mask = batch['attention_mask'].to(device_hw)
        model(input_ids=ids, attention_mask=mask)

    lname, analog = get_layer0(model)
    if lname not in hook_data:
        return None

    d = hook_data[lname]
    y_c = d['y_c']
    y_ab = d['y_ab']
    a_w = analog.tile_a.get_weights()[0]
    b_w = analog.tile_b.get_weights()[0]
    c_w = analog.tile_c.get_weights()[0]

    model.train()
    return {
        'a_norm': a_w.norm().item(),
        'b_norm': b_w.norm().item(),
        'c_norm': c_w.norm().item(),
        'ab_prod_norm': (a_w @ b_w).norm().item(),
        'a_max': a_w.abs().max().item(),
        'b_max': b_w.abs().max().item(),
        'y_c_norm': y_c.norm().item(),
        'y_ab_norm': y_ab.norm().item(),
        'alpha_y_ab_norm': (lora_alpha * y_ab).norm().item(),
        'total_norm': (y_c + lora_alpha * y_ab).norm().item(),
        'ratio': (lora_alpha * y_ab).norm().item() / max(y_c.norm().item(), 1e-10),
    }


def run_one(lora_alpha, mode, device_hw, loader, first_batch):
    model = build_model(lora_alpha, mode, device_hw)
    hook_data = {}
    install_hooks(model, hook_data, lora_alpha)
    optimizer = AnalogSGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Collect init
    _, analog = get_layer0(model)
    init_a = analog.tile_a.get_weights()[0].clone()
    init_b = analog.tile_b.get_weights()[0].clone()

    rows = []
    m = collect_metrics(model, hook_data, first_batch, lora_alpha, device_hw)
    if m:
        m['step'] = 0
        m['a_change'] = 0.0
        m['b_change'] = 0.0
        m['loss'] = None
        rows.append(m)

    model.train()
    for step, batch in enumerate(loader):
        if step >= NUM_STEPS:
            break
        ids = batch['input_ids'].to(device_hw)
        mask = batch['attention_mask'].to(device_hw)
        labels = batch['labels'].to(device_hw)

        optimizer.zero_grad()
        out = model(input_ids=ids, attention_mask=mask)
        loss = criterion(out.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        m = collect_metrics(model, hook_data, first_batch, lora_alpha, device_hw)
        if m:
            cur_a = analog.tile_a.get_weights()[0]
            cur_b = analog.tile_b.get_weights()[0]
            m['step'] = step + 1
            m['a_change'] = (cur_a - init_a).norm().item()
            m['b_change'] = (cur_b - init_b).norm().item()
            m['loss'] = loss.item()
            rows.append(m)

    return rows


def print_table(alpha, analog_rows, digital_rows):
    print(f"\n{'='*120}")
    print(f"LORA_ALPHA = {alpha}")
    print(f"{'='*120}")

    header = f"{'Step':>4} | {'Mode':>7} | {'A_norm':>8} {'B_norm':>8} {'A_chg':>8} {'B_chg':>8} | " \
             f"{'C_out':>8} {'AB_out':>8} {'α*AB':>8} {'Total':>8} {'Ratio':>7} | " \
             f"{'AB_prod':>8} {'Loss':>12}"
    print(header)
    print("-" * 120)

    show_steps = {0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50}
    for step in range(NUM_STEPS + 1):
        if step not in show_steps:
            continue
        a = analog_rows[step] if step < len(analog_rows) else None
        d = digital_rows[step] if step < len(digital_rows) else None

        for tag, r in [("ANALOG", a), ("DIGITAL", d)]:
            if r is None:
                continue
            loss_str = f"{r['loss']:.1f}" if r['loss'] is not None else "-"
            print(f"{r['step']:>4} | {tag:>7} | "
                  f"{r['a_norm']:>8.4f} {r['b_norm']:>8.4f} {r['a_change']:>8.4f} {r['b_change']:>8.4f} | "
                  f"{r['y_c_norm']:>8.2f} {r['y_ab_norm']:>8.2f} {r['alpha_y_ab_norm']:>8.2f} {r['total_norm']:>8.2f} {r['ratio']:>7.4f} | "
                  f"{r['ab_prod_norm']:>8.4f} {loss_str:>12}")
        if a or d:
            print("-" * 120)


def main():
    device_hw = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    raw = load_dataset("glue", "rte", split="train[:1024]")
    def preprocess(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"],
                        truncation=True, padding="max_length", max_length=128)
    tokenized = raw.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    loader = list(DataLoader(tokenized, batch_size=16, shuffle=False, collate_fn=default_data_collator))
    first_batch = loader[0]

    for alpha in [0.1, 1.0, 5.0]:
        analog_rows = run_one(alpha, "analog", device_hw, loader, first_batch)
        digital_rows = run_one(alpha, "digital", device_hw, loader, first_batch)
        print_table(alpha, analog_rows, digital_rows)


if __name__ == "__main__":
    main()
