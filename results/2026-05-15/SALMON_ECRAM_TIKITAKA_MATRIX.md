# SALMON — EcRam Tikitaka precision matrix (cross-GPU)

- **Generated:** 2026-05-19
- **Task:** CIFAR-10, ResNet18, 300 epochs, seed=12345, AnalogSGD + CosineAnnealing(T_max=300), momentum=0.9, **backbone = EcRamPreset**
- **Tikitaka:** analog fast tile + analog transfer (`make_tikitaka_ecram`), digital opt.lr pinned at 0.01
- **Why two source DBs:** per `SALMON/CORE8_4GPU_SPLIT.md` the EcRam precision lanes are split across hosts —
  - **gpu2** lane = EcRam **OFF + INT8** → `GPU2_SALMON_RESULTS_SUMMARY.{json,md}`
  - **gpu3** lane = EcRam **FP32 + FP16** → `GPU3_SALMON_RESULTS_SUMMARY.{json,md}`
  - Tikitaka sweeps were built on top of each host's own baseline cells, so EcRam-tikitaka data is physically distributed across the two DBs. This file consolidates them.

## Master matrix — test/backbone_acc (%)

`fast_lr × transfer_lr` 2×2 per (dpu_precision, SA) cell. **Bold** = cell-best.

| dpu_precision | SA | f0.1/t0.1 | f0.1/t1.0 | f1.0/t0.1 | f1.0/t1.0 | matched baseline (opt.lr=0.01) | source DB / run prefix |
|---|---|---|---|---|---|---|---|
| off       | off | 66.10 | 51.20 | **68.94** | 59.52 | 52.34 (`core8_ecram_off_lr01`)       | **GPU2** / `core8_tikitaka_ecram_off_*` |
| int8_qat  | on  | 70.83 | 58.81 | **74.12** | 67.00 | 57.37 (`core8_ecram_int8_qat_lr01`)  | **GPU2** / `core8_tikitaka_ecram_fast*` |
| fp32      | on  | 70.52 | 59.04 | **74.75** | 65.77 | 58.40 (`core8_ecram_fp32_lr01`)      | **GPU3** / `core8_tikitaka_ecram_f*`   |
| fp16      | on  |  —    |  —    | **73.45** |  —    | 57.56 (`core8_ecram_fp16_lr01`)      | **GPU3** / `core8_tikitaka_ecram_f10_t01_fp16` |
| int4      | on  |  —    |  —    |  —    |  —    | — | not implemented (see gaps) |

## Best-cell summary (all are fast_lr=1.0 / transfer_lr=0.1)

| precision / SA | best test acc | Δ vs matched baseline |
|---|---|---|
| fp32, SA on  | **74.75%** | +16.35 pp |
| int8_qat, SA on | 74.12% | +16.75 pp |
| fp16, SA on  | 73.45% | +15.89 pp |
| off, SA off  | 68.94% | +16.60 pp |

→ Tikitaka fast-tile gain is ~+16 pp and **largely precision-independent** at the best cell; SA/DPU precision (off→int8→fp16→fp32) only moves the absolute level ~+5–6 pp on top. `fast_lr=1.0, transfer_lr=0.1` is the best corner in every precision; `transfer_lr=1.0` collapses training (early-best) across all.

## Gaps (empty / incomplete cells)

1. **fp16 tikitaka 2×2 incomplete** — only `f1.0/t0.1` exists (GPU3). Missing: `f0.1/t0.1`, `f0.1/t1.0`, `f1.0/t1.0` (dpu=fp16, SA on).
2. **int4 tikitaka — not runnable on the EcRam tree.** `dpu_precision` only applies to the SA/DPU branch; `exp7_1_module.py::run_dpu_attention` supports only `fp32/fp16/int8_qat` (else `ValueError`), and the gpu2 tree imports only `prepare_dpu_int8_qat` (no `prepare_dpu_int4_qat`). INT4 also has no EcRam lane in `CORE8_4GPU_SPLIT.md`. Requires SA-branch INT4-QAT implementation first.
3. **No-SA (off) only on the gpu2 lane.** fp16/fp32 lanes (gpu3) ran SA-on only — there is no fp16/fp32 "without SA" contrast, and conceptually `dpu_precision` is a no-op when `sa_enabled=false` (no DPU branch), so `off/off` is the only meaningful no-SA point.

## Provenance

- GPU2: `/root/SALMON/logs/core8_tikitaka_ecram_off_*` (2026-05-15→16, `_tikitaka_off_dispatch.log`), `core8_tikitaka_ecram_fast*` (2026-05-12→14, `_tikitaka_dispatch.log`).
- GPU3: `core8_tikitaka_ecram_f*` (fp32), `core8_tikitaka_ecram_f10_t01_fp16` (fp16, added in commit `24455fc3`).
- Per-run metrics: each run's `csv/version_0/metrics.csv`; best_val/test extracted, see the two GPU{2,3} SALMON JSON DBs for full fields (timestamps, best_val_epoch, loss).
