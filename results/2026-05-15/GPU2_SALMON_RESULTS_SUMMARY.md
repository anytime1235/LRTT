# SALMON — core-8 results summary

- **Generated:** 2026-05-15
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

## Takeaways
- `sa_enabled` alone gives a small bump (~5 pp at matched lr); the big jump comes from adding the tikitaka analog fast tile (+22 pp baseline → tikitaka).
- Lower optimizer.lr (0.01) wins every matched pair.
- Within the tikitaka 2×2, higher fast_lr and lower transfer_lr are both better — high transfer_lr (1.0) costs 7–10 pp.
- Best run end-to-end: `core8_tikitaka_ecram_fast10_tr01` — **test/backbone_acc 74.12%, val 75.04% at epoch 171 (early-best ckpt).**

## Dispatch logs (timeline)
- `logs/_gpu2_dispatch.log` — first wave (off / int8_qat, lr=0.10), finished 2026-05-09 12:00.
- `logs/_gpu2_dispatch_lr01.log` — lr=0.01 rerun, finished 2026-05-10 09:30.
- `logs/_tikitaka_dispatch.log` — 2×2 tikitaka sweep, last run finished 2026-05-14 10:46.
