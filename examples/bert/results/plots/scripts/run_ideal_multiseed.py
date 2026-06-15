#!/usr/bin/env python3
"""constantstepideal multi-seed analysis: bilinear cascade verification on truly ideal device.

GPU 0/1/2/3 × seeds 42/43/44/45, all FI=False, minimal diag, same other hyperparams.
"""
import datetime, json, os, re, subprocess, sys, shutil, time
from pathlib import Path

REPO = Path("/root/LRTT")
SRC = REPO / "examples/bert/fine_bert_squad_lrtt.py"
RESULTS_DIR = REPO / "examples/bert/results/BERT_SQUAD_LRTT_FINE"
OUT_DIR = REPO / "examples/bert/results/plots/diag_ideal_multiseed"
RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

VARIANTS = [
    dict(tag=f"ideal_seed{seed}", gpu=gpu, seed=seed)
    for gpu, seed in [(0, 42), (1, 43), (2, 44), (3, 45)]
]

MINIMAL_DIAG_GROUPS = """{
    "g1_norms":        True, "g2_minmax": True, "g3_mean": False, "g3b_mean_abs": False,
    "g3c_weight_hist": False, "g4_deltas": True, "g5a_erank_ab": False, "g5b_erank_c": False,
    "g6a_cells": False, "g6b_cell_deltas": False, "g7_cosines": True, "g8_signal_abs": True,
    "g10_signal_hist": False, "g11a_xc_dc_abs": True, "g11c_xc_dc_hist": False, "g11d_xfer_meta": True,
}"""


def patch(c, k, v):
    p = re.compile(rf"^{k}\s*=.*$", re.MULTILINE)
    out, n = p.subn(f"{k} = {v}", c, count=1)
    if n == 0: raise RuntimeError(f"Failed patch {k}")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = SRC.read_text()
    procs = []
    for v in VARIANTS:
        unique = f"ideal_{RUN_STAMP}_{v['tag']}"
        src = base
        src = patch(src, "N_EPOCHS", "5"); src = patch(src, "SCHEDULE_EPOCHS", "5")
        src = patch(src, "DIAG_EPOCHS", "5"); src = patch(src, "BATCH_SIZE", "48")
        src = patch(src, "WARMUP_STEPS", "365"); src = patch(src, "SEED", str(v["seed"]))
        src = patch(src, "TRAIN_SUBSET_SIZE", "0"); src = patch(src, "EVAL_SUBSET_SIZE", "0")
        src = patch(src, "LRTT_RANK", "32"); src = patch(src, "LEARNING_RATE", "0.0038")
        src = patch(src, "TRANSFER_LR", "0.095"); src = patch(src, "TRANSFER_EVERY", "1")
        src = patch(src, "FAST_LR", "0.474"); src = patch(src, "AB_DW_MIN", "0.0004883")
        src = patch(src, "C_DW_MIN", "0.001953"); src = patch(src, "AB_MULTILEVEL", "None")
        # KEY: ideal device for A and B (no 6T1C dynamics)
        src = patch(src, "AB_DEVICE", '"constantstepideal"')
        src = patch(src, "A_DEVICE", '"constantstepideal"')
        src = patch(src, "B_DEVICE", '"constantstepideal"')
        src = patch(src, "REINIT_MODE", '"decay"'); src = patch(src, "TRANSFER_METHOD", '"onehot"')
        src = patch(src, "C_DEVICE", '"constantstepideal"'); src = patch(src, "LORA_TARGET", '"qkvo"')
        src = patch(src, "FORWARD_INJECT", "False"); src = patch(src, "FI_CONTINUOUS_ALPHA", "False")
        src = patch(src, "LEARN_OUT_SCALING", "False"); src = patch(src, "IS_PERFECT", "True")
        src = patch(src, "OUT_NOISE", "0.0"); src = patch(src, "AUTO_SCALE_MODE", '"none"')
        src = patch(src, "MIN_LR_RATE", "0.0"); src = patch(src, "ENABLE_DIAGNOSTIC", "True")
        src = patch(src, "ERANK_RATE_LIMIT_STEPS", "0"); src = patch(src, "HIST_RATE_STEPS", "1844")
        src = patch(src, "DIAG_TILES", '"first_last"')
        src = re.sub(r"DIAG_GROUPS\s*=\s*\{[^}]+\}", f"DIAG_GROUPS = {MINIMAL_DIAG_GROUPS}", src, count=1)
        src = src.replace(
            'stamp = f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"',
            f'stamp = f"{unique}"'
        )
        tmp = SRC.parent / f"_tmp_{unique}.py"
        tmp.write_text(src)
        log = RESULTS_DIR / f"runlog_{unique}.txt"
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(v["gpu"])
        logf = open(log, "wb")
        proc = subprocess.Popen([sys.executable, "-u", str(tmp)], cwd=SRC.parent,
                                stdout=logf, stderr=subprocess.STDOUT, env=env)
        print(f"  [GPU {v['gpu']}] launched: {v['tag']} (seed={v['seed']}, ALL IDEAL)  pid={proc.pid}",
              flush=True)
        procs.append({"variant": v, "proc": proc, "tmp": tmp, "log": log, "unique": unique, "logf": logf})

    print(f"\nAll {len(procs)} launched. Waiting...", flush=True)
    while procs:
        time.sleep(60)
        still = []
        for p in procs:
            if p["proc"].poll() is not None:
                p["logf"].close()
                tag = p["variant"]["tag"]
                jp = RESULTS_DIR / f"squad_diagnostic_log_{p['unique']}.json"
                if jp.exists():
                    dst = OUT_DIR / f"diag_{p['unique']}.json"
                    shutil.copy(jp, dst)
                    d = json.loads(dst.read_text())
                    f1_best = d.get("best_f1", -1)
                    f1_e5 = d["epoch_history"][-1]["f1"] if d.get("epoch_history") else None
                    print(f"  [GPU {p['variant']['gpu']}] Done {tag} exit={p['proc'].returncode}, "
                          f"F1_best={f1_best:.2f}, F1_e5={f1_e5}", flush=True)
                else:
                    print(f"  [GPU {p['variant']['gpu']}] Done {tag} exit={p['proc'].returncode}, NO JSON",
                          flush=True)
                p["tmp"].unlink(missing_ok=True)
            else:
                still.append(p)
        procs = still
    print("All done.", flush=True)


if __name__ == "__main__":
    main()
