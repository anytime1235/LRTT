# SALMON — core-8 results summary

- **GPU:** **gpu2** (EcRamPreset machine per `CORE8_4GPU_SPLIT.md`; all runs here ran on this host, A100-SXM4-40GB MIG 3g.20gb)
- **Generated:** 2026-05-15 · **Updated:** 2026-05-19 (added GPU2 SA-OFF/dpu_off tikitaka sweep, 4 runs)
- **Source:** `/root/SALMON/logs/core8_*/runs/*/csv/version_0/metrics.csv`
- **Task:** CIFAR-10, ResNet18, 300 epochs, seed=12345, AnalogSGD + CosineAnnealing (T_max=300, eta_min=0.001), momentum=0.9
- **Backbone analog tile preset:** `EcRamPreset` (all runs)
- **Note:** `results/core8_dpu_precision_summary.{csv,md}` on disk is an unfilled template — actual metrics live in each run's `csv/version_0/metrics.csv`. This file consolidates them.

## Results table

| run_id | sa_enabled | dpu_precision | opt.lr | fast_lr | transfer_lr | best_val/backbone_acc | test/backbone_acc | test/loss |
|---|---|---|---|---|---|---|---|---|
| core8_ecram_off                  | false | off       | 0.10 |  —  |  —  | 0.4576 | **0.4430** | 1.4957 |
| core8_ecram_off_lr01             | false | off       | 0.01 |  —  |  —  | 0.5326 | **0.5234** | 1.3128 |
| core8_ecram_int8_qat             | true  | int8_qat  | 0.10 |  —  |  —  | 0.5422 | **0.5224** | 1.3029 |
| core8_ecram_int8_qat_lr01        | true  | int8_qat  | 0.01 |  —  |  —  | 0.5880 | **0.5737** | 1.1822 |
| core8_tikitaka_ecram_fast01_tr01 | true  | int8_qat  | 0.01 | 0.1 | 0.1 | 0.7156 | **0.7083** | 0.8183 |
| core8_tikitaka_ecram_fast01_tr10 | true  | int8_qat  | 0.01 | 0.1 | 1.0 | 0.6042 | **0.5881** | 1.1414 |
| core8_tikitaka_ecram_fast10_tr01 | true  | int8_qat  | 0.01 | 1.0 | 0.1 | 0.7504 | **0.7412** | 0.7328 |
| core8_tikitaka_ecram_fast10_tr10 | true  | int8_qat  | 0.01 | 1.0 | 1.0 | 0.6786 | **0.6700** | 0.9306 |
| core8_tikitaka_ecram_off_fast01_tr01 | false | off | 0.01 | 0.1 | 0.1 | 0.6720 | **0.6610** | 0.9594 |
| core8_tikitaka_ecram_off_fast01_tr10 | false | off | 0.01 | 0.1 | 1.0 | 0.5168 | **0.5120** | 1.3435 |
| core8_tikitaka_ecram_off_fast10_tr01 | false | off | 0.01 | 1.0 | 0.1 | 0.6926 | **0.6894** | 0.8832 |
| core8_tikitaka_ecram_off_fast10_tr10 | false | off | 0.01 | 1.0 | 1.0 | 0.6014 | **0.5952** | 1.1523 |

## Three experiment groups

### 1) Baselines — `sa_enabled=false`, no DPU
- `core8_ecram_off` (lr=0.10): test/backbone_acc = **44.30%**
- `core8_ecram_off_lr01` (lr=0.01): test/backbone_acc = **52.34%**

### 2) DPU on (int8_qat), no tikitaka — `sa_enabled=true`
- `core8_ecram_int8_qat` (lr=0.10): test/backbone_acc = **52.24%**
- `core8_ecram_int8_qat_lr01` (lr=0.01): test/backbone_acc = **57.37%**  *(+5.0pp over matched off baseline)*

### 3) Tikitaka + DPU int8_qat — `fast_lr × transfer_lr` sweep at opt.lr=0.01

| | transfer_lr=0.1 | transfer_lr=1.0 |
|---|---|---|
| **fast_lr=0.1** | 70.83% | 58.81% |
| **fast_lr=1.0** | **74.12%** | 67.00% |

Best cell: `fast_lr=1.0, transfer_lr=0.1`.

### 4) Tikitaka, SA off + DPU off — `fast_lr × transfer_lr` sweep at opt.lr=0.01  *(added 2026-05-19, GPU2)*

Companion to group 3 (`scripts/run_core8_tikitaka_ecram_off.sh`): same analog fast tile, but **no SA and no DPU**. Isolates the tikitaka fast-tile contribution alone.

| | transfer_lr=0.1 | transfer_lr=1.0 |
|---|---|---|
| **fast_lr=0.1** | 66.10% | 51.20% |
| **fast_lr=1.0** | **68.94%** | 59.52% |

Best cell: `fast_lr=1.0, transfer_lr=0.1` → test **68.94%** vs matched `core8_ecram_off_lr01` baseline 52.34% = **+16.60 pp** from the analog fast tile alone. transfer_lr=1.0 collapses training (early-best ep67/105) — same pattern as group 3.

## Takeaways
- `sa_enabled` alone gives a small bump (~5 pp at matched lr); the big jump comes from adding the tikitaka analog fast tile (+22 pp baseline → tikitaka).
- Lower optimizer.lr (0.01) wins every matched pair.
- Within the tikitaka 2×2, higher fast_lr and lower transfer_lr are both better — high transfer_lr (1.0) costs 7–10 pp.
- Best run end-to-end: `core8_tikitaka_ecram_fast10_tr01` — **test/backbone_acc 74.12%, val 75.04% at epoch 171 (early-best ckpt).**

## Dispatch logs (timeline)
- `logs/_gpu2_dispatch.log` — first wave (off / int8_qat, lr=0.10), finished 2026-05-09 12:00.
- `logs/_gpu2_dispatch_lr01.log` — lr=0.01 rerun, finished 2026-05-10 09:30.
- `logs/_tikitaka_dispatch.log` — 2×2 tikitaka sweep (int8_qat), last run finished 2026-05-14 10:46.
- `logs/_tikitaka_off_dispatch.log` — 2×2 tikitaka SA-OFF/dpu_off sweep (GPU2), 2026-05-15 06:45 → 2026-05-16 ~21:30.
