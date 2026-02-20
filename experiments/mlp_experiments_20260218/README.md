# MLP Experiments - February 2026

**Date**: 2026-02-18
**Branch**: MLP
**GPU Status**: Experiments completed before GPU disconnect

## 📁 Directory Structure

```
mlp_experiments_20260218/
├── tikitaka/              # Tikitaka algorithm experiments
├── lrtt_decay/            # LRTT with decay transfer scheme
├── lrtt_loraalpha/        # LRTT LoRA alpha parameter sweeps
├── rank8_experiments/     # Rank-8 NN experiments (hybrid & decay)
├── noise_asymmetry/       # Noise asymmetry experiments
├── analysis_scripts/      # Analysis and sweep generation scripts
├── logs/                  # Experiment execution logs
└── configs/               # Configuration files for sweeps
```

## 🎯 Experiment Summary

### 1. Tikitaka Experiments
**Directory**: `tikitaka/`

- **tikitaka_all_analog/**: Full analog tikitaka sweep (30 trials)
  - Results: `sweep_results.json`
- **tikitaka_all_analog_top5_200ep/**: Top 5 configurations run for 200 epochs
  - Trials: 0010, 0011, 0012, 0013, 0014
- **tikitaka_optimal_all_analog/**: Optimal configuration experiment
- **sweep_tikitaka_all_analog_v2/**: Version 2 of tikitaka sweep (30 trials)

**Key Files**:
- `tikitaka/sweep_tikitaka_all_analog/sweep_results.json` (343 lines, 30 trials)

### 2. LRTT Decay Experiments
**Directory**: `lrtt_decay/`

Systematic sweep of LRTT with decay transfer scheme over different transfer intervals.

- **Transfer Every (TE) values**: 50, 100, 500, 1000
- **Trials per TE**: 10 trials each
- **Total trials**: 40

**Key Files**:
- `lrtt_decay/sweep_lrtt_decay_200ep/sweep_results.json`
- Individual trial directories: `te{50,100,500,1000}_trial_000{0-9}/`

**Results**: 200 epochs, comprehensive hyperparameter optimization

### 3. LRTT LoRA Alpha Experiments
**Directory**: `lrtt_loraalpha/`

Exploration of LoRA alpha parameter effects on LRTT performance.

- **LoRA Alpha values**: 0.01, 0.1, 0.5, 1.0
- **Transfer Every values**: 50, 100, 500, 1000
- **Trials**: 4 trials per combination
- **Total combinations**: 64 trials

**Key Files**:
- `lrtt_loraalpha/sweep_lrtt_loraalpha_200ep/sweep_results.json` (149 lines)
- `lrtt_loraalpha/sweep_lrtt_loraalpha_200ep/sequential_results.json`
- `lrtt_loraalpha/sweep_lrtt_loraalpha_analysis.png` (visualization)

### 4. Rank8 NN Experiments
**Directory**: `rank8_experiments/`

Experiments exploring rank-8 low-rank adaptations with nearest-neighbor initialization.

#### a) Rank8 NN Hybrid
- **Mode**: Hybrid transfer (A=0 hard reset, B unchanged)
- **Results**: `rank8_nn_hybrid/results_final.json` (897 lines)
- **Best accuracy**: 93.49% (TE=2)

#### b) Rank8 NN Decay
- **Mode**: Decay transfer scheme
- **Results**: `rank8_nn_decay/results_final.json` (897 lines)
- **Best accuracy**: 96.89% (TE=2)

#### c) Rank8 TE20 Hybrid
- **Mode**: Hybrid with TE=20
- **Results**: `rank8_te20_hybrid/` directory

**Analysis**:
- `te_trend_analysis.png`: Trend analysis visualization

### 5. Noise Asymmetry Experiments
**Directory**: `noise_asymmetry/`

Investigation of asymmetric noise effects in analog training.

- **Modes**: hybrid
- **Ranks tested**: 1
- **TE values**: 500
- **Noise scales**: 0%, 10%, 50%
- **Up/down asymmetry**: 0%

**Key Files**:
- `noise_asymmetry/noise_asymmetry/results.json` (3241 lines)
- **Total configurations**: ~200+ different settings

**Results highlights**:
- Best accuracy with 50% noise: 95.63%
- Best accuracy with 10% noise: 95.12%

## 📊 Analysis Scripts
**Directory**: `analysis_scripts/`

| Script | Purpose |
|--------|---------|
| `analysis_rank8_lrtt.py` | Rank-8 LRTT analysis |
| `analyze_sweep_results.py` | General sweep result analyzer |
| `analyze_te_lr_correlation.py` | TE-LR correlation analysis |
| `analyze_te_trend.py` | Transfer every trend analysis |
| `sweep_rank8_*.py` | Various rank-8 sweep generators |

## 📝 Configuration Files
**Directory**: `configs/`

- `rank8_newte_sweep_configs.json` (25K)
- `rank8_nn_sweep_configs.json` (25K)
- `rank8_te16_configs.json` (17K)
- `rank8_te16_sweep_configs.json` (15K)
- `rank8_te20_sweep_configs.json` (17K)

## 📋 Execution Logs
**Directory**: `logs/`

Contains execution logs for all experiments:
- `tikitaka_*.log`
- `sweep_lrtt_*.log`
- `sweep_rank8_*.log`

Notable logs:
- `tikitaka_top5_200ep_nohup.log` (1.1M) - Complete 200 epoch runs
- `sweep_lrtt_decay_50ep.log` (347K) - 50 epoch decay sweep

## 🔬 Key Findings

### Best Configurations by Experiment Type

1. **Rank8 NN Decay**: 96.89% accuracy (best overall)
   - LR: 0.1158, TLR: 0.0009, TE: 2

2. **Rank8 NN Hybrid**: 93.49% accuracy
   - LR: 0.1476, TLR: 0.0011, TE: 2

3. **Noise Asymmetry**: 95.63% accuracy
   - 50% noise, hybrid mode, rank 1, TE: 500

## 📈 Data Statistics

- **Total files**: 167 JSON/log files
- **Total size**: ~8.8 GB
- **Experiment types**: 5 major categories
- **Total trials**: 200+ individual runs
- **Epochs range**: 30-200 epochs

## 🔧 LRTT Best Configs Update

The best configurations from these experiments have been saved to:
`/root/LRTT/experiments/mnist_sweep_analysis/lrtt_sweep_best_configs.json`

Updated: 2026-02-16 16:18

## ⚠️ Notes

- All experiments were completed before GPU disconnect (nvidia-smi failure)
- Results represent the final state of training when processes were terminated
- Some trials may have been interrupted; check individual trial logs for completion status

## 📚 Related Files

Outside this directory but part of the same experimental batch:
- `/root/LRTT/experiments/mnist_sweep_analysis/lrtt_sweep_best_configs.json`
- Various reoptimization results in `mnist_sweep_analysis/`

---

**Generated**: 2026-02-18
**Author**: Automated experiment collection
**Status**: Complete, ready for analysis
