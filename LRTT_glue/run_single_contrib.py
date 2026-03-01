"""Single-run contribution measurement. Called as subprocess."""
import sys
sys.path.insert(0, '/data/LRTT_transformer/src')
import os, json, math
import torch, torch.nn as nn, numpy as np
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, default_data_collator, set_seed)
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import (
    FloatingPointDevice, LinearStepDevice, SoftBoundsDevice)
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

MODEL_NAME = "google/mobilebert-uncased"
RANK, EPOCHS, BATCH, MAXLEN, SEED_VAL = 8, 3, 256, 128, 42
TARGETS = ["query", "key", "value"]
MIN_LR_RATIO = 0.05
WARMUP_STEPS = 30


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


def measure(model, batch, dev):
    """Measure C vs AB contribution using tile.forward() (captures actual analog output)."""
    cn, abn = {}, {}
    ctrls = []
    for n, m in model.named_modules():
        if hasattr(m, 'analog_module'):
            am = m.analog_module
            if hasattr(am, 'controller') and am.controller.forward_inject_enabled:
                ctrls.append((n, am.controller))
    origs = {}
    for ln, ctrl in ctrls:
        orig = ctrl._forward_inject_analog_unified
        def mk(l, c):
            def h(x, in_trans=False, out_trans=False):
                xb = x.t() if in_trans else x
                g = c.tile_b.forward(xb)
                ya = c.tile_a.forward(g)
                yc = c.tile_c.forward(xb)
                y = yc + c.lora_alpha * ya
                cn[l] = yc.norm().item()
                abn[l] = (c.lora_alpha * ya).norm().item()
                return y.t() if out_trans else y
            return h
        ctrl._forward_inject_analog_unified = mk(ln, ctrl)
        origs[(ln, ctrl)] = orig
    model.eval()
    with torch.no_grad():
        model(input_ids=batch['input_ids'].to(dev), attention_mask=batch['attention_mask'].to(dev))
    model.train()
    for (l,c), o in origs.items(): c._forward_inject_analog_unified = o
    if not cn: return 0, 0, 0
    cv, av = list(cn.values()), list(abn.values())
    rs = [a/max(c,1e-10) for a,c in zip(av, cv)]
    return float(np.mean(cv)), float(np.mean(av)), float(np.mean(rs))


def evaluate(model, loader, dev):
    model.eval(); preds, labels = [], []; tl = 0; crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for b in loader:
            o = model(input_ids=b['input_ids'].to(dev), attention_mask=b['attention_mask'].to(dev))
            l = crit(o.logits, b['labels'].to(dev)); tl += l.item()*b['labels'].size(0)
            preds.extend(o.logits.argmax(-1).cpu().numpy()); labels.extend(b['labels'].numpy())
    model.train()
    return float(f1_score(labels, preds)), float(np.mean(np.array(preds)==np.array(labels))), tl/len(labels)


def run(ab_type, alpha, lr, out_file):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw = load_dataset("nyu-mll/glue", "mrpc")
    def pp(ex): return tok(ex["sentence1"], ex["sentence2"], padding="max_length", max_length=MAXLEN, truncation=True)
    td = raw.map(pp, batched=True).rename_column("label", "labels")
    tl = DataLoader(td["train"], batch_size=BATCH, shuffle=True, collate_fn=default_data_collator)
    el = DataLoader(td["validation"], batch_size=8, shuffle=False, collate_fn=default_data_collator)
    sb = next(iter(el))

    set_seed(SEED_VAL)
    cfg = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=cfg)
    allL = [n for n,m in model.named_modules() if isinstance(m, nn.Linear)]
    exc = [n for n in allL if not any(t in n for t in TARGETS)]; exc.append("classifier")
    model = convert_to_analog(model, create_config(ab_type, alpha), exclude_modules=exc)
    for n,p in model.named_parameters():
        p.requires_grad = (any(t in n for t in TARGETS) and "bias" not in n) or "classifier" in n
    model.to(dev)
    torch.cuda.empty_cache()

    opt = AnalogAdam(model.parameters(), lr=lr); opt.regroup_param_groups()
    ns = len(tl) * EPOCHS
    def lrl(s):
        if s < WARMUP_STEPS:
            return float(s) / float(max(1, WARMUP_STEPS))
        p = float(s - WARMUP_STEPS)/float(max(1, ns - WARMUP_STEPS))
        return max(MIN_LR_RATIO, 1.0 - (1.0-MIN_LR_RATIO)*p)
    from torch.optim.lr_scheduler import LambdaLR
    sch = LambdaLR(opt, lrl); crit = nn.CrossEntropyLoss()

    result = {'ab_type': ab_type, 'alpha': alpha, 'lr': lr,
              'lr_eff': lr*alpha, 'status': 'ok', 'epochs': []}

    cn, abn, ratio = measure(model, sb, dev)
    f1, acc, loss = evaluate(model, el, dev)
    result['epochs'].append({'ep':0, 'f1':f1, 'acc':acc, 'loss':loss,
                             'c_norm':cn, 'ab_norm':abn, 'ratio':ratio})
    print(f"Ep0: F1={f1:.4f} C={cn:.2f} AB={abn:.4f} ratio={ratio:.6f}", flush=True)
    # Save after each epoch
    with open(out_file, 'w') as f: json.dump(result, f, indent=2)

    best_f1 = f1; model.train()
    for ep in range(1, EPOCHS+1):
        tls, nb = 0, 0
        for i, batch in enumerate(tl):
            opt.zero_grad()
            o = model(input_ids=batch['input_ids'].to(dev), attention_mask=batch['attention_mask'].to(dev))
            loss = crit(o.logits, batch['labels'].to(dev))
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN at ep{ep} step{i}", flush=True)
                result['status'] = 'diverged'
                with open(out_file, 'w') as f: json.dump(result, f, indent=2)
                return result
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); sch.step()
            tls += loss.item(); nb += 1

        cn, abn, ratio = measure(model, sb, dev)
        f1, acc, evl = evaluate(model, el, dev)
        if f1 > best_f1: best_f1 = f1
        result['epochs'].append({'ep':ep, 'f1':f1, 'acc':acc, 'loss':evl,
                                  'train_loss':tls/nb, 'c_norm':cn, 'ab_norm':abn, 'ratio':ratio})
        print(f"Ep{ep}: F1={f1:.4f} TrL={tls/nb:.4f} C={cn:.2f} AB={abn:.4f} ratio={ratio:.6f}", flush=True)
        result['best_f1'] = best_f1
        with open(out_file, 'w') as f: json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    ab_type = sys.argv[1]
    alpha = float(sys.argv[2])
    lr = float(sys.argv[3])
    out_file = sys.argv[4]
    print(f"=== {ab_type} alpha={alpha} lr={lr} lr_eff={lr*alpha:.6f} ===", flush=True)
    r = run(ab_type, alpha, lr, out_file)
    print(f"Done: {r.get('status','?')} best_f1={r.get('best_f1',0):.4f}", flush=True)
