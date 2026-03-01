"""Compare FP A/B vs sixt1c A/B on MRPC: contribution ratio analysis.

Goal:
  1) FP A/B, alpha=8, lr=1e-3: measure ||C(x)|| vs ||alpha*A(B(x))||
  2) sixt1c A/B, various alpha, lr=1e-3: find alpha giving similar ratio
"""
import os
import sys
sys.path.insert(0, '/data/LRTT_transformer/src')

import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    default_data_collator, set_seed,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import (
    FloatingPointDevice, LinearStepDevice, SoftBoundsDevice,
)
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

MODEL_NAME = "google/mobilebert-uncased"
RANK = 8
NUM_EPOCHS = 3
BATCH_SIZE = 128
MAX_SEQ_LENGTH = 128
SEED = 42
WARMUP_STEPS = 0
MIN_LR_RATIO = 0.05
LR = 0.001
TARGET_MODULES = ["query", "key", "value"]


# ============================================================
# Config builders
# ============================================================
def create_config(ab_type, lora_alpha):
    if ab_type == "fp":
        ab_device = FloatingPointDevice()
    else:
        TAU_SEC = 46505.0
        delta = 1 - math.exp(-1.0 / TAU_SEC)
        lifetime = 1.0 / delta if delta > 0 else 0.0
        ab_device = LinearStepDevice(
            dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
            gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
            dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
            gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3,
            write_noise_std=0.0, mean_bound_reference=True,
            lifetime=lifetime, lifetime_dtod=0.0, reset=0.0, reset_dtod=0.0,
        )

    c_device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=True,
    )
    device_config = PythonLRTTDevice(
        rank=RANK, transfer_every=1000000,
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


def create_model(ab_type, lora_alpha, device_hw):
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    exclude = [n for n in all_linear if not any(t in n for t in TARGET_MODULES)]
    exclude.append("classifier")

    rpu_config = create_config(ab_type, lora_alpha)
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES) and "bias" not in name
        param.requires_grad = is_target or "classifier" in name

    return model.to(device_hw)


# ============================================================
# Contribution measurement (hooks into forward_inject)
# ============================================================
def measure_contributions(model, sample_batch, device_hw):
    """Measure actual ||C(x)|| and ||alpha*A(B(x))|| via monkey-patch."""
    c_norms, ab_norms = {}, {}

    # Collect controllers
    controllers = []
    for name, mod in model.named_modules():
        if not hasattr(mod, 'analog_module'):
            continue
        am = mod.analog_module
        if hasattr(am, 'controller') and am.controller.forward_inject_enabled:
            controllers.append((name, am.controller))

    # Patch to capture
    originals = {}
    for lname, ctrl in controllers:
        orig = ctrl._forward_inject_analog_unified

        def make_hook(ln, c, orig_fn):
            def hooked(x, in_trans=False, out_trans=False):
                x_bf = x.t() if in_trans else x
                g = c.tile_b.forward(x_bf)
                y_ab = c.tile_a.forward(g)
                y_c = c.tile_c.forward(x_bf)
                y = y_c + c.lora_alpha * y_ab
                c_norms[ln] = y_c.norm().item()
                ab_norms[ln] = (c.lora_alpha * y_ab).norm().item()
                return y.t() if out_trans else y
            return hooked
        ctrl._forward_inject_analog_unified = make_hook(lname, ctrl, orig)
        originals[(lname, ctrl)] = orig

    model.eval()
    with torch.no_grad():
        ids = sample_batch['input_ids'].to(device_hw)
        mask = sample_batch['attention_mask'].to(device_hw)
        model(input_ids=ids, attention_mask=mask)
    model.train()

    # Restore
    for (ln, ctrl), orig in originals.items():
        ctrl._forward_inject_analog_unified = orig

    if not c_norms:
        return {'c_norm': 0, 'ab_norm': 0, 'ratio': 0}

    c_vals = list(c_norms.values())
    ab_vals = list(ab_norms.values())
    ratios = [ab / max(c, 1e-10) for ab, c in zip(ab_vals, c_vals)]
    return {
        'c_norm': np.mean(c_vals),
        'ab_norm': np.mean(ab_vals),
        'ratio': np.mean(ratios),
    }


# ============================================================
# Evaluate
# ============================================================
def evaluate(model, eval_loader, device_hw):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in eval_loader:
            ids = batch['input_ids'].to(device_hw)
            mask = batch['attention_mask'].to(device_hw)
            labels = batch['labels'].to(device_hw)
            outputs = model(input_ids=ids, attention_mask=mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    model.train()
    f1 = f1_score(all_labels, all_preds)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return f1, acc, total_loss / len(all_labels)


# ============================================================
# Single run
# ============================================================
def run_single(ab_type, lora_alpha, lr, device_hw,
               train_loader, eval_loader, sample_batch):
    label = f"{ab_type} A/B, alpha={lora_alpha}, lr={lr}"
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    set_seed(SEED)
    model = create_model(ab_type, lora_alpha, device_hw)

    optimizer = AnalogAdam(model.parameters(), lr=lr)
    optimizer.regroup_param_groups()

    num_steps = len(train_loader) * NUM_EPOCHS
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step) / float(max(1, WARMUP_STEPS))
        progress = float(step - WARMUP_STEPS) / float(max(1, num_steps - WARMUP_STEPS))
        return max(MIN_LR_RATIO, 1.0 - (1.0 - MIN_LR_RATIO) * progress)

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    # ---- Init ----
    c0 = measure_contributions(model, sample_batch, device_hw)
    f1, acc, loss = evaluate(model, eval_loader, device_hw)
    print(f"  Ep0: F1={f1:.4f} Acc={acc:.4f} Loss={loss:.4f}  "
          f"C={c0['c_norm']:.2f} AB={c0['ab_norm']:.4f} ratio={c0['ratio']:.6f}")

    best_f1 = f1
    contrib_log = [c0]

    # ---- Train ----
    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss, n_batches = 0.0, 0
        for batch in tqdm(train_loader, desc=f"  Ep{epoch}", leave=False):
            ids = batch['input_ids'].to(device_hw)
            mask = batch['attention_mask'].to(device_hw)
            labels = batch['labels'].to(device_hw)

            optimizer.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask)
            loss = criterion(outputs.logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  *** NaN/Inf at epoch {epoch}, aborting ***")
                del model; torch.cuda.empty_cache()
                return {
                    'ab_type': ab_type, 'alpha': lora_alpha, 'lr': lr,
                    'best_f1': best_f1, 'acc': acc,
                    'contrib': contrib_log, 'status': 'diverged',
                }

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        ct = measure_contributions(model, sample_batch, device_hw)
        contrib_log.append(ct)
        f1, acc, eval_loss = evaluate(model, eval_loader, device_hw)
        train_avg = total_loss / n_batches
        print(f"  Ep{epoch}: F1={f1:.4f} Acc={acc:.4f} TrL={train_avg:.4f} EvL={eval_loss:.4f}  "
              f"C={ct['c_norm']:.2f} AB={ct['ab_norm']:.4f} ratio={ct['ratio']:.6f}")
        if f1 > best_f1:
            best_f1 = f1

    print(f"  Best F1: {best_f1:.4f}")
    del model; torch.cuda.empty_cache()
    return {
        'ab_type': ab_type, 'alpha': lora_alpha, 'lr': lr,
        'best_f1': best_f1, 'acc': acc,
        'contrib': contrib_log, 'status': 'ok',
    }


# ============================================================
# Main
# ============================================================
def main():
    device_hw = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    raw = load_dataset("nyu-mll/glue", "mrpc")
    def preprocess(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"],
                        padding="max_length", max_length=MAX_SEQ_LENGTH, truncation=True)
    tokenized = raw.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=default_data_collator)
    eval_loader = DataLoader(tokenized["validation"], batch_size=128,
                             shuffle=False, collate_fn=default_data_collator)
    sample_batch = next(iter(eval_loader))

    print(f"MRPC: Train={len(tokenized['train'])}, Val={len(tokenized['validation'])}")
    print(f"Batches: train={len(train_loader)}, eval={len(eval_loader)}")
    print(f"LR={LR}, RANK={RANK}, EPOCHS={NUM_EPOCHS}, BATCH={BATCH_SIZE}")

    # ---- Experiment configs ----
    # (ab_type, alpha, lr)
    # Key: controller does lr_eff = lr * alpha for tile.update()
    # Previous best: sixt1c alpha=0.02, lr=0.002 → lr_eff = 0.00004
    # To match lr_eff=0.00004 with alpha=8: lr = 0.000005
    # But that's too small for AnalogAdam (digital params).
    # Strategy: keep lr=1e-3 for optimizer, let lr_eff vary.
    # Just see what ratio emerges at each alpha.
    LR_EFF_REF = 0.00004  # reference lr_eff from best config
    configs = [
        # FP: alpha=8, lr adjusted so lr_eff matches reference
        ("fp",     8.0,   LR_EFF_REF / 8.0),   # lr=5e-6, lr_eff=4e-5
        ("fp",     8.0,   5e-5),                 # lr=5e-5, lr_eff=4e-4
        ("fp",     8.0,   5e-4),                 # lr=5e-4, lr_eff=4e-3 (may diverge)
        # FP: smaller alpha for comparison
        ("fp",     1.0,   LR_EFF_REF / 1.0),    # lr=4e-5, lr_eff=4e-5
        ("fp",     0.02,  0.002),                 # lr=2e-3, lr_eff=4e-5 (same as prev best)
        # sixt1c: same lr_eff reference
        ("sixt1c", 0.02,  0.002),                 # lr_eff=4e-5 (prev best)
        ("sixt1c", 0.1,   LR_EFF_REF / 0.1),     # lr=4e-4, lr_eff=4e-5
        ("sixt1c", 0.5,   LR_EFF_REF / 0.5),     # lr=8e-5, lr_eff=4e-5
        ("sixt1c", 1.0,   LR_EFF_REF / 1.0),     # lr=4e-5, lr_eff=4e-5
        ("sixt1c", 8.0,   LR_EFF_REF / 8.0),     # lr=5e-6, lr_eff=4e-5
    ]

    results = []
    for ab_type, alpha, lr in configs:
        import gc; gc.collect(); torch.cuda.empty_cache()
        try:
            r = run_single(ab_type, alpha, lr, device_hw,
                           train_loader, eval_loader, sample_batch)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            r = {'ab_type': ab_type, 'alpha': alpha, 'lr': lr,
                 'best_f1': 0.0, 'acc': 0.0, 'contrib': [], 'status': 'error'}
        results.append(r)

    # ---- Summary ----
    print(f"\n{'='*95}")
    print(f"  MRPC: FP A/B vs sixt1c A/B  —  Contribution Ratio & F1  (lr={LR})")
    print(f"{'='*95}")
    print(f"  {'Type':>7} {'Alpha':>6} {'LR':>8} {'lr_eff':>8} | {'F1':>7} {'Stat':>8} | "
          f"{'InitR':>10} {'Ep1R':>10} {'Ep2R':>10} {'Ep3R':>10} | "
          f"{'FinalC':>8} {'FinalAB':>8}")
    print(f"  {'-'*110}")

    fp_best = None
    for r in results:
        cl = r.get('contrib', [])
        ratios = [c['ratio'] for c in cl] if cl else [0]*4
        while len(ratios) < 4:
            ratios.append(0)

        c_final = cl[-1]['c_norm'] if cl else 0
        ab_final = cl[-1]['ab_norm'] if cl else 0
        lr_eff = r['lr'] * r['alpha']

        marker = ""
        if r['ab_type'] == 'fp' and r.get('status') == 'ok':
            if fp_best is None or r['best_f1'] > fp_best['best_f1']:
                fp_best = r
                marker = " ***"

        print(f"  {r['ab_type']:>7} {r['alpha']:>6.2f} {r['lr']:>8.4f} {lr_eff:>8.4f} | "
              f"{r['best_f1']:>7.4f} {r.get('status','?'):>8} | "
              f"{ratios[0]:>10.6f} {ratios[1]:>10.6f} {ratios[2]:>10.6f} {ratios[3]:>10.6f} | "
              f"{c_final:>8.2f} {ab_final:>8.4f}{marker}")

    if fp_best:
        fp_cl = fp_best.get('contrib', [])
        fp_r = fp_cl[-1]['ratio'] if fp_cl else 0
        print(f"\n  *** Best FP: alpha={fp_best['alpha']}, lr={fp_best['lr']}, "
              f"F1={fp_best['best_f1']:.4f}, final ratio={fp_r:.6f}")
    print(f"{'='*95}")


if __name__ == "__main__":
    main()
