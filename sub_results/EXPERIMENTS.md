# Analog Fine-Tuning Optuna Experiments

> Comprehensive documentation of all Optuna hyperparameter search experiments for analog hardware-aware fine-tuning.

> **Total DB files:** 336 | **Studies with data:** 265 | **Generated:** 2026-03-01

---
## 1. 실험 개요

### 목적
IBM Analog Hardware Kit (AIHWKit)을 활용하여 BERT 계열 모델의 analog hardware-aware fine-tuning 최적 하이퍼파라미터를 Optuna로 탐색.

### 하드웨어 설정
- **GPU:** NVIDIA H200
- **Analog Simulation:** IBM AIHWKit (aihwkit)
- **Frameworks:** PyTorch 2.3.1+cu121, Transformers 4.44.0

### 실험 매트릭스

| | ALBERT | BERT | MobileBERT |
|---|:---:|:---:|:---:|
| **Baseline** | - | - | - |
| **LoRA-LRTT** | GLUE (44), GLUE-2S (21), SQuAD (5) | - | GLUE (42), GLUE-TS (1), SQuAD (9) |
| **TikiTaka v1** | GLUE (54), SQuAD (2) | SQuAD (47) | GLUE (22) |
| **TikiTaka v2** | - | - | GLUE (10), SQuAD (7) |

*괄호 안 숫자: Optuna study 수*

---
## 2. 디렉토리 구조

```
sub_results/
├── EXPERIMENTS.md                    ← 이 문서
├── docs/                             5개 setup 문서
│   ├── Albert_setup.txt
│   ├── AnalogLora_setup.txt
│   ├── LRTT_setup.txt
│   ├── Setup.txt
│   └── Tikitaka_setup.txt
├── scripts/                          Python 실험 스크립트
│   ├── baseline/
│   ├── albert/{lora,tiki}/
│   ├── bert/tiki/
│   └── mobilebert/{lora,tiki}/
├── shell_scripts/                    Shell 실행 스크립트
│   ├── baseline/
│   ├── albert/{lora/{glue,squad}, tikitaka/glue}/
│   └── mobilebert/{lora/glue, tikitaka/glue}/
├── results/                          Optuna DB 파일 (336개)
│   ├── albert/
│   │   ├── lora/{glue, glue_2stage, squad}/
│   │   └── tikitaka/{glue, squad}/
│   ├── bert/tikitaka/squad/
│   ├── mobilebert/
│   │   ├── lora/{glue, glue_trainscale, squad}/
│   │   └── tikitaka/{glue_v1, glue_v2, squad_v2}/
│   └── _misc/{analoglora_alllayer, tikitaka_legacy}/
├── logs/                             학습 로그
│   ├── baseline/
│   ├── albert/{lora/glue, tikitaka/glue}/
│   ├── bert/tikitaka/squad/
│   └── mobilebert/{lora/glue, tikitaka/glue}/
└── tikitaka_sweep/                   TikiTaka v2 sweep 분석
    ├── analysis_report.md
    ├── best_params_summary.json
    ├── sweep_results_20260204.json
    └── sweep_remaining_tasks.log
```

---
## 3. 방법론 (Method) 설명

### 3.1 Baseline (Frozen Analog QKV)

- Analog tile에 QKV 가중치를 로드한 뒤 **freeze** (학습 없음)
- Classification head만 digital로 학습
- 비교 기준선으로 사용

### 3.2 LoRA-LRTT (Low-Rank Adaptation + Transfer Tiles)

- LoRA의 low-rank A/B 행렬을 analog transfer tile로 구현
- `AnalogLinear`에 `InferenceRPUConfig` + TTv2 기반 학습
- **핵심 파라미터:**
  - `learning_rate`: 전체 학습률 (0.001~1.0, log scale)
  - `lora_alpha`: LoRA scaling factor
  - `target_ab_lr`: A/B transfer tile 학습률
  - `lora_rank`: Low-rank dimension
- **변형:**
  - 2-Stage: Stage 1 (head freeze) → Stage 2 (full)
  - TrainScale: output scaling도 학습
  - Warm Alpha: alpha를 warm-up 후 적용

### 3.3 TikiTaka v1 (ChoppedTransferCompound)

- 3-weight matrix scheme: A (fast), C (slow/main), χ (chi, hidden)
- `ChoppedTransferCompound` RPU config
- **핵심 파라미터:**
  - `learning_rate`: SGD/Adam base LR
  - `transfer_lr`: A→χ transfer 학습률
  - `fast_lr`: fast matrix 학습률
  - `transfer_every`: transfer 주기 (steps)
  - `auto_granularity`: granularity 자동 설정
  - `in_chop_prob`: chopping probability

### 3.4 TikiTaka v2

- TikiTaka v1의 개선 버전
- `units_in_mbatch` (UIMB) 파라미터 추가
- 더 안정적인 학습 dynamics
- MobileBERT GLUE/SQuAD에서 주로 실험

---
## 4. 실험 결과 (모델별)

### 4.1 ALBERT

#### 4.1.1 LoRA GLUE (44 studies)

**Best per Task:**

| Task | Best Value | Study | Trials |
|------|-----------|-------|--------|
| SST2 | **0.9140** | `albert_sst2_lrtt_bs32_sgd_decay_nowd_nomom_nonest_combo` | 13/25 |
| MRPC | **0.9151** | `albert_mrpc_lrtt_bs32_sgd_decay_nowd_nomom_nonest_combo` | 16/30 |
| STSB | **0.8851** | `albert_stsb_lrtt_bs16_sgd_decay_nowd_nomom_nonest_combo` | 22/30 |
| COLA | **0.4887** | `albert_cola_lrtt_bs16_sgd_decay_nowd_nomom_nonest_combo` | 7/30 |
| RTE | **0.7401** | `albert_rte_lrtt_bs32_sgd_decay_nowd_nomom_nonest_combos` | 1/1 |
| QQP | **0.7315** | `albert_qqp_lrtt_bs128_sgd_decay_nowd_nomom_nonest_combo` | 0/2 |

**Top 10 Studies:**

| # | Study Name | Trials (Complete/Total) | Best Value | Key Parameters |
|---|-----------|------------------------|------------|----------------|
| 1 | `albert_mrpc_lrtt_bs32_sgd_decay_nowd_nomom_nonest_combos_...` | 16/30 | **0.9151** | learning_rate=0.0198, lora_alpha=0.0146, target_ab_lr=0.0685 |
| 2 | `albert_mrpc_lrtt_bs32_sgd_decay_nowd_nomom_nonest_combos_...` | 16/30 | **0.9151** | learning_rate=0.0198, lora_alpha=0.0146, target_ab_lr=0.0685 |
| 3 | `albert_sst2_lrtt_bs32_sgd_decay_nowd_nomom_nonest_combos_...` | 13/25 | **0.9140** | learning_rate=0.00116, lora_alpha=0.0191, target_ab_lr=0.0370 |
| 4 | `albert_sst2_lrtt_bs32_sgd_decay_nowd_nomom_nonest_combos_...` | 11/50 | **0.9140** | learning_rate=0.00116, lora_alpha=0.0191, target_ab_lr=0.0370 |
| 5 | `albert_sst2_lrtt_bs32_sgd_decay_nowd_nomom_nonest_combos_...` | 0/2 | **0.9094** | learning_rate=0.0233, lora_alpha=0.0100, target_ab_lr=0.0300 |
| 6 | `albert_sst2_frozen_no_outscaling` | 2/3 | **0.9083** | learning_rate=0.0233, lora_alpha=0.0100, target_ab_lr=0.0300 |
| 7 | `albert_mrpc_frozen_no_outscaling` | 1/2 | **0.8924** | learning_rate=0.0625, lora_alpha=0.0100, target_ab_lr=0.0300 |
| 8 | `albert_mrpc_lrtt_bs32_sgd_decay_nowd_nomom_nonest_combos_...` | 1/2 | **0.8859** | learning_rate=0.0625, lora_alpha=0.0100, target_ab_lr=0.0300 |
| 9 | `albert_stsb_lrtt_bs16_sgd_decay_nowd_nomom_nonest_combos_...` | 22/30 | **0.8851** | learning_rate=0.00212, lora_alpha=0.0330, target_ab_lr=0.00591 |
| 10 | `albert_mrpc_lrtt_bs32_sgd_decay_nowd_nomom_nonest_combos_...` | 1/2 | **0.8812** | learning_rate=0.0198, lora_alpha=0.0100, target_ab_lr=0.0300 |

#### 4.1.2 LoRA GLUE 2-Stage (21 studies)

2-Stage 학습: Stage 1에서 head를 freeze하고 LoRA만 학습 → Stage 2에서 전체 fine-tune.

| Task | Best Value | Study | Trials |
|------|-----------|-------|--------|
| MRPC | **0.9065** | `albert_mrpc_lora2s_bs32_sgd_decay_nowd_nomom_nonest_com` | 1/13 |
| RTE | **0.7040** | `albert_rte_lora2s_bs32_learncos_attn` | 1/14 |
| OTHER | **0.6787** | `diag_stepwise_ablr001_v1` | 1/1 |

**All Studies:**

| # | Study Name | Trials (Complete/Total) | Best Value | Key Parameters |
|---|-----------|------------------------|------------|----------------|
| 1 | `albert_mrpc_lora2s_bs32_sgd_decay_nowd_nomom_nonest_combo...` | 1/13 | **0.9065** | lora_alpha=0.0114, target_ab_lr=0.00889 |
| 2 | `albert_mrpc_lora2s_bs32_sgd_decay_nowd_nomom_nonest_combo...` | 2/5 | **0.9034** | lora_alpha=0.00680, target_ab_lr=0.1221 |
| 3 | `albert_mrpc_lora2s_bs32_sgd_decay_nowd_nomom_nonest_combo...` | 3/18 | **0.9029** | lora_alpha=0.1208, target_ab_lr=0.0681 |
| 4 | `albert_mrpc_lora2s_trainln_adam_all` | 0/10 | **0.8785** | - |
| 5 | `albert_mrpc_lora2s_traincls_adam_all_v1` | 0/6 | **0.8158** | - |
| 6 | `albert_rte_lora2s_bs32_learncos_attn` | 1/14 | **0.7040** | lora_alpha=0.0364, target_ab_lr=0.1668 |
| 7 | `albert_rte_lora2s_bs32_learncos01_attn` | 0/8 | **0.6968** | lora_alpha=0.0364, target_ab_lr=0.1668 |
| 8 | `albert_rte_lora2s_bs32_learncos1e3_all_walpha` | 0/5 | **0.6859** | lora_alpha=0.0114, target_ab_lr=0.00889 |
| 9 | `albert_rte_lora2s_bs32_learncos1e3_all` | 0/7 | **0.6823** | lora_alpha=0.0114, target_ab_lr=0.00889 |
| 10 | `diag_stepwise_ablr001_v1` | 1/1 | **0.6787** | - |
| 11 | `albert_rte_lora2s_bs32_sgd_decay_nowd_nomom_nonest_combos...` | 0/1 | **0.6751** | lora_alpha=0.0360, target_ab_lr=0.1670 |
| 12 | `albert_rte_lora2s_bs32_learncos01_all` | 0/3 | **0.6498** | lora_alpha=0.0360, target_ab_lr=0.1670 |
| 13 | `albert_rte_lora2s_trainln_adam_all_v2` | 0/5 | **0.6318** | - |
| 14 | `diag_stepwise_alpha036_v1` | 1/1 | **0.5957** | - |
| 15 | `diag_noiseless_alpha036_rte_v1` | 1/1 | **0.5415** | - |
| 16 | `diag_noiseless_rte_v1` | 1/1 | **0.5379** | lora_alpha=1.0000, target_ab_lr=0.0100 |
| 17 | `albert_rte_lora2s_traincls_adam_all_v3` | 0/5 | **0.5307** | - |
| 18 | `albert_rte_lora2s_traincls_adam_all_v2` | 0/5 | **N/A** | - |
| 19 | `albert_rte_lora2s_bs32_sgd_decay_nowd_nomom_nonest_combos...` | 0/1 | **N/A** | - |
| 20 | `albert_rte_lora2s_trainln_adam_all` | 0/5 | **N/A** | - |
| 21 | `albert_rte_lora2s_traincls_adam_all_v1` | 0/5 | **N/A** | - |

#### 4.1.3 LoRA SQuAD (5 studies)

| # | Study Name | Trials (Complete/Total) | Best Value | Key Parameters |
|---|-----------|------------------------|------------|----------------|
| 1 | `albert_squad_lrtt_bs48_sgd_decay_nowd_nomom_nonest_combos...` | 13/16 | **85.00** | learning_rate=0.2858, lora_alpha=0.0610, target_ab_lr=0.00909 |
| 2 | `albert_squad_lrtt_bs48_sgd_decay_nowd_nomom_nonest_combos...` | 2/5 | **85.00** | learning_rate=0.2858, lora_alpha=0.0610, target_ab_lr=0.00909 |
| 3 | `albert_squad_lrtt_bs64_sgd_decay_nowd_nomom_nonest_combos...` | 0/1 | **79.23** | learning_rate=0.2858, lora_alpha=0.0610, target_ab_lr=0.00909 |
| 4 | `albert_squad_lrtt_bs64_sgd_decay_nowd_nomom_nonest_convnt...` | 0/1 | **5.02** | learning_rate=0.2858, lora_alpha=0.0610, target_ab_lr=0.00909 |
| 5 | `albert_squad_lrtt2s_bs48_sgd_decay_nowd_nomom_nonest_comb...` | 0/1 | **N/A** | - |

> Best F1: **85.00** (bs48, all layers, combos config, 13/16 trials complete)

#### 4.1.4 TikiTaka GLUE (54 studies)

다양한 GLUE 태스크에서 TikiTaka v1 탐색. Num layers (nl3~nl10), frozen target 등 다양한 변형 포함.

| Task | Best Value | Study | Trials |
|------|-----------|-------|--------|
| SST2 | **0.9174** | `albert_glue_tiki_sst2_bs32_adam_nowd_nomom_nonest_attn` | 2/3 |
| MRPC | **0.9122** | `albert_glue_tiki_mrpc_bs32_adam_nowd_nomom_nonest_attn_` | 6/7 |
| STSB | **0.8675** | `albert_glue_tiki_stsb_bs16_adam_nowd_nomom_nonest_frtgt` | 1/1 |
| COLA | **0.5139** | `albert_glue_tiki_cola_bs16_adam_nowd_nomom_nonest_attn` | 12/20 |
| RTE | **0.7329** | `albert_glue_tiki_rte_bs32_adam_nowd_nomom_nonest_attn_v` | 10/50 |
| QNLI | **0.8733** | `albert_glue_tiki_qnli_bs32_sgd_nowd_nomom_nonest_attn` | 0/1 |

**Top 15 Studies:**

| # | Study Name | Trials (Complete/Total) | Best Value | Key Parameters |
|---|-----------|------------------------|------------|----------------|
| 1 | `albert_glue_tiki_sst2_bs32_adam_nowd_nomom_nonest_attn` | 2/3 | **0.9174** | learning_rate=0.00056 |
| 2 | `albert_glue_tiki_mrpc_bs32_adam_nowd_nomom_nonest_attn_v1...` | 6/7 | **0.9122** | learning_rate=0.00167, transfer_lr=0.0540 |
| 3 | `albert_glue_tiki_mrpc_bs32_adam_nowd_nomom_nonest_attn` | 13/22 | **0.9119** | learning_rate=0.00185 |
| 4 | `albert_glue_tiki_sst2_bs32_adam_nowd_nomom_nonest_frtgt_a...` | 1/1 | **0.9083** | - |
| 5 | `albert_glue_tiki2s_mrpc_bs32_sgd_nomom_nonest_attn` | 3/24 | **0.9066** | transfer_lr=0.5399, fast_lr=0.0149, transfer_every=0.00000 |
| 6 | `albert_glue_tiki_mrpc_bs32_adam_nowd_nomom_nonest_frtgt_a...` | 1/1 | **0.9063** | - |
| 7 | `albert_glue_tiki_sst2_bs32_adam_nowd_nomom_nonest_frtgt_a...` | 0/1 | **0.9014** | - |
| 8 | `albert_glue_tiki_sst2_bs32_adam_nowd_nomom_nonest_frtgt_a...` | 1/1 | **0.9002** | - |
| 9 | `albert_glue_tiki_mrpc_bs32_adam_nowd_nomom_nonest_attn_nl6` | 1/1 | **0.8954** | - |
| 10 | `albert_glue_tiki_mrpc_bs32_adam_nowd_nomom_nonest_frtgt_a...` | 1/1 | **0.8938** | - |
| 11 | `albert_glue_tiki_mrpc_bs32_adam_nowd_nomom_nonest_frtgt_a...` | 1/1 | **0.8889** | - |
| 12 | `albert_glue_tiki_qnli_bs32_sgd_nowd_nomom_nonest_attn` | 0/1 | **0.8733** | learning_rate=0.0300, transfer_lr=0.1000, fast_lr=0.0100 |
| 13 | `albert_glue_tiki_stsb_bs16_adam_nowd_nomom_nonest_frtgt_a...` | 1/1 | **0.8675** | - |
| 14 | `albert_glue_tiki_stsb_bs16_adam_nowd_nomom_nonest_attn` | 1/2 | **0.8656** | learning_rate=0.00291 |
| 15 | `albert_glue_tiki_stsb_bs16_adam_nowd_nomom_nonest_frtgt_a...` | 1/1 | **0.8642** | - |

#### 4.1.5 TikiTaka SQuAD (2 studies)

| # | Study Name | Trials (Complete/Total) | Best Value | Key Parameters |
|---|-----------|------------------------|------------|----------------|
| 1 | `albert_squad_tiki_bs48_sgd_nowd_nomom_nonest_all_te1_tpe3...` | 0/1 | **N/A** | - |
| 2 | `albert_squad_tiki_bs48_adam_nowd_nomom_nonest_attn` | 0/1 | **N/A** | - |

> 두 study 모두 1 trial만 실행되었으며 완료된 trial이 없음 (early stage).

### 4.2 BERT

#### 4.2.1 TikiTaka SQuAD (47 studies)

BERT-base에서 SQuAD v1.1 TikiTaka 실험. 다양한 layer config (all/qkv), noise 설정, LR range 탐색.

**Top 10 Studies:**

| # | Study Name | Trials (Complete/Total) | Best Value | Key Parameters |
|---|-----------|------------------------|------------|----------------|
| 1 | `bert_squad_tiki_all_los_lr001_10_20trials` | 4/5 | **87.49** | learning_rate=0.0625 |
| 2 | `bert_squad_tiki_all_nlos_lr001_10_20trials` | 4/5 | **87.44** | learning_rate=0.0625 |
| 3 | `bert_squad_tiki_qkv_ntideal_los_lr001_10_20trials` | 4/5 | **86.70** | learning_rate=0.0625 |
| 4 | `bert_squad_tiki_qkv_ntideal_nlos_lr001_10_20trials` | 4/5 | **86.50** | learning_rate=0.0625 |
| 5 | `bert_squad_tiki_0225_1853` | 9/17 | **80.11** | learning_rate=0.3034 |
| 6 | `bert_squad_tiki_0226_1817` | 4/5 | **79.68** | learning_rate=0.6482 |
| 7 | `bert_squad_tiki_0226_1706` | 1/2 | **79.60** | learning_rate=0.6482 |
| 8 | `bert_squad_tiki_0225_1729` | 1/2 | **79.53** | learning_rate=0.00056 |
| 9 | `bert_squad_tiki_0226_0945` | 7/10 | **79.50** | learning_rate=1.1844 |
| 10 | `bert_squad_tiki_0226_1507` | 2/5 | **77.33** | learning_rate=0.6482 |

> Best F1: **87.49** (`bert_squad_tiki_all_los_lr001_10_20trials`)
>
> 전체 47 studies 중 유효한 결과가 있는 study: 29개

### 4.3 MobileBERT

#### 4.3.1 LoRA GLUE (42 studies)

MobileBERT에서 LoRA-LRTT GLUE 실험. QKV layer 대상, 다양한 clipping/scaling 실험 포함.

| Task | Best Value | Study | Trials |
|------|-----------|-------|--------|
| SST2 | **0.8647** | `mobilebert_sst2_lrtt_bs64_sgd_decay_nowd_nomom_nonest_c` | 60/60 |
| MRPC | **0.8525** | `mobilebert_mrpc_lrtt_bs32_adam_decay_nowd_nomom_nonest_` | 14/30 |
| COLA | **0.3548** | `mobilebert_cola_lrtt_bs32_adam_decay_nowd_nomom_nonest_` | 5/7 |
| RTE | **0.6318** | `rte_adam_lr6e3_v2` | 14/30 |
| MNLI | **0.6792** | `mobilebert_mnli_lrtt_bs64_sgd_decay_nowd_nomom_nonest_c` | 3/4 |
| SQUAD | **N/A** | `mobilebert_squad_lrtt_bs32_adam_qkv` | 0/3 |
| OTHER | **0.5378** | `all_layer_grid_efflr` | 14/15 |

**Top 10 Studies:**

| # | Study Name | Trials (Complete/Total) | Best Value | Key Parameters |
|---|-----------|------------------------|------------|----------------|
| 1 | `mobilebert_sst2_lrtt_bs64_sgd_decay_nowd_nomom_nonest_com...` | 60/60 | **0.8647** | learning_rate=0.9589, lora_alpha=0.00942, target_ab_lr=0.0244 |
| 2 | `mobilebert_mrpc_lrtt_bs32_adam_decay_nowd_nomom_nonest_co...` | 14/30 | **0.8525** | lora_alpha=0.00534, target_ab_lr=0.00054 |
| 3 | `mobilebert_sst2_lrtt_bs64_sgd_decay_nowd_nomom_nonest_noi...` | 6/7 | **0.8463** | learning_rate=0.00000, lora_alpha=0.0236 |
| 4 | `mobilebert_sst2_lrtt_bs64_sgd_decay_nowd_nomom_nonest_noi...` | 50/61 | **0.8452** | learning_rate=0.00000, lora_alpha=0.0284 |
| 5 | `mobilebert_mrpc_lrtt_bs32_adam_decay_nowd_nomom_nonest_co...` | 15/30 | **0.8395** | learning_rate=0.0143, lora_alpha=0.00413, target_ab_lr=0.00120 |
| 6 | `test_clip_analog_norm100_mrpc` | 3/3 | **0.8230** | learning_rate=0.2369, lora_alpha=0.0412, target_ab_lr=0.0526 |
| 7 | `test_clip_analog_norm1000_mrpc` | 3/3 | **0.8226** | learning_rate=0.5000, lora_alpha=0.0100, target_ab_lr=0.0300 |
| 8 | `test_clip_highalpha_mrpc` | 10/10 | **0.8169** | learning_rate=0.2704, lora_alpha=0.0146, target_ab_lr=0.0329 |
| 9 | `test_analogclip_norm100_v2_mrpc` | 5/5 | **0.8149** | learning_rate=0.5000, lora_alpha=0.0100, target_ab_lr=0.0300 |
| 10 | `test_epoch20_mrpc` | 3/3 | **0.8122** | learning_rate=0.5000, lora_alpha=0.0100, target_ab_lr=0.0300 |

#### 4.3.2 LoRA GLUE TrainScale (1 study)

| # | Study Name | Trials (Complete/Total) | Best Value | Key Parameters |
|---|-----------|------------------------|------------|----------------|
| 1 | `mobilebert_sst2_lrtt_bs64_sgd_decay_nowd_nomom_nonest_noi...` | 16/17 | **0.5252** | learning_rate=0.00000, lora_alpha=0.00407 |

> Output scaling 학습 가능 여부 테스트. SST2에서 best 0.5252 (16/17 trials).

#### 4.3.3 LoRA SQuAD (9 studies)

| # | Study Name | Trials (Complete/Total) | Best Value | Key Parameters |
|---|-----------|------------------------|------------|----------------|
| 1 | `mobilebert_squad_lrtt_bs64_sgd_decay_nowd_nomom_nonest_co...` | 24/25 | **67.18** | learning_rate=0.5045, lora_alpha=0.0321, target_ab_lr=0.00268 |
| 2 | `mobilebert_squad_lora_bs64_sgd_decay_nowd_nomom_nonest_no...` | 20/20 | **54.39** | learning_rate=0.2858, lora_alpha=0.0610, target_ab_lr=0.00909 |
| 3 | `mobilebert_squad_lrtt_bs64_sgd_decay_nowd_nomom_nonest_no...` | 14/59 | **47.97** | learning_rate=0.00000, lora_alpha=0.0164 |
| 4 | `mobilebert_squad_lora_bs64_sgd_decay_nowd_nomom_nonest_no...` | 1/2 | **45.09** | learning_rate=0.2858, lora_alpha=0.0610, target_ab_lr=0.00909 |
| 5 | `mobilebert_squad_lrtt_bs64_sgd_decay_nowd_nomom_nonest_co...` | 0/1 | **18.16** | learning_rate=0.5045, lora_alpha=0.0321, target_ab_lr=0.00268 |
| 6 | `mobilebert_squad_lrtt_bs64_sgd_decay_nowd_nomom_nonest_no...` | 3/4 | **15.57** | learning_rate=1.0000, lora_alpha=0.00000, target_ab_lr=2.0000 |
| 7 | `mobilebert_squad_lrtt_bs64_sgd_decay_nowd_nomom_nonest_no...` | 3/4 | **14.91** | learning_rate=1.0000, lora_alpha=0.00000, target_ab_lr=2.0000 |
| 8 | `mobilebert_squad_lora_bs64_sgd_decay_nowd_nomom_nonest_no...` | 0/1 | **6.68** | learning_rate=0.00000, lora_alpha=0.00000 |
| 9 | `mobilebert_squad_lrtt_bs64_sgd_decay_nowd_nomom_nonest_qkv` | 0/25 | **N/A** | - |

> Best F1: **67.18** (combos config, QKV layers)

#### 4.3.4 TikiTaka v1 GLUE (22 studies)

| Task | Best Value | Study | Trials |
|------|-----------|-------|--------|
| SST2 | **0.8498** | `sst2_alllayer_top5_grid` | 12/12 |
| MRPC | **0.8134** | `mobilebert_glue_tiki_mrpc_bs32_sgd_nowd_nomom_nonest_qk` | 18/30 |
| COLA | **0.0959** | `mobilebert_glue_tiki_cola_bs16_sgd_nowd_nomom_nonest_qk` | 25/30 |
| STSB | **-1.0000** | `mobilebert_glue_tiki_stsb_bs16_sgd_nowd_nomom_nonest_qk` | 30/30 |
| RTE | **0.5451** | `mobilebert_glue_tiki_rte_bs32_sgd_nowd_nomom_nonest_qkv` | 24/30 |
| QQP | **N/A** | `mobilebert_glue_tiki_qqp_bs64_sgd_nowd_nomom_nonest_all` | 0/4 |
| QNLI | **N/A** | `mobilebert_glue_tiki_qnli_bs64_sgd_nowd_nomom_nonest_al` | 0/1 |
| MNLI | **N/A** | `mobilebert_glue_tiki_mnli_bs64_sgd_nowd_nomom_nonest_al` | 0/1 |

**All Studies:**

| # | Study Name | Trials (Complete/Total) | Best Value | Key Parameters |
|---|-----------|------------------------|------------|----------------|
| 1 | `sst2_alllayer_top5_grid` | 12/12 | **0.8498** | learning_rate=0.00000, fast_lr=1.0000, transfer_every=0.00000 |
| 2 | `mobilebert_glue_tiki_sst2_bs64_sgd_nowd_nomom_nonest_all` | 36/36 | **0.8475** | learning_rate=0.00000, fast_lr=1.0000, transfer_every=0.00000 |
| 3 | `mobilebert_glue_tiki_mrpc_bs32_sgd_nowd_nomom_nonest_qkv_los` | 18/30 | **0.8134** | learning_rate=0.00175, fast_lr=0.00315, transfer_every=2.0000 |
| 4 | `mobilebert_glue_tiki_mrpc_bs32_sgd_nowd_nomom_nonest_qkv_...` | 2/4 | **0.8122** | learning_rate=0.00110, fast_lr=0.00611, transfer_every=0.00000 |
| 5 | `mobilebert_glue_tiki_mrpc_bs32_sgd_nowd_nomom_nonest_qkv` | 1/2 | **0.8122** | learning_rate=0.0727, fast_lr=8.6274, transfer_every=0.00000 |
| 6 | `mobilebert_glue_tiki_rte_bs32_sgd_nowd_nomom_nonest_qkv_los` | 24/30 | **0.5451** | learning_rate=0.00373, fast_lr=0.00840, transfer_every=4.0000 |
| 7 | `mobilebert_glue_tiki_rte_bs64_sgd_nowd_nomom_nonest_ffn_tpe` | 9/12 | **0.5379** | learning_rate=0.00561, fast_lr=0.0815, transfer_every=0.00000 |
| 8 | `mobilebert_glue_tiki_rte_bs32_sgd_nowd_nomom_nonest_ffn_l...` | 7/8 | **0.5307** | learning_rate=0.00175, fast_lr=0.00315, transfer_every=2.0000 |
| 9 | `mobilebert_glue_tiki_rte_bs64_sgd_nowd_nomom_nonest_all_l...` | 2/3 | **0.5271** | learning_rate=0.00561, fast_lr=0.0815, transfer_every=0.00000 |
| 10 | `mobilebert_glue_tiki_rte_bs64_sgd_nowd_nomom_nonest_all_tpe` | 6/8 | **0.5271** | learning_rate=0.00561, fast_lr=0.0815, transfer_every=0.00000 |
| 11 | `mobilebert_glue_tiki_sst2_bs32_sgd_nowd_nomom_nonest_qkv_los` | 1/2 | **0.5138** | learning_rate=0.00110, fast_lr=0.00611, transfer_every=0.00000 |
| 12 | `mobilebert_glue_tiki_cola_bs16_sgd_nowd_nomom_nonest_qkv_los` | 25/30 | **0.0959** | learning_rate=0.00350, fast_lr=0.0761, transfer_every=0.00000 |
| 13 | `mobilebert_glue_tiki_mrpc_tpe_10t_5ep_w5pct` | 0/1 | **0.0000** | learning_rate=0.0727, fast_lr=0.7979, transfer_every=0.00000 |
| 14 | `mobilebert_glue_tiki_stsb_bs16_sgd_nowd_nomom_nonest_qkv_los` | 30/30 | **-1.0000** | learning_rate=0.00561, fast_lr=0.0815, transfer_every=0.00000 |
| 15 | `mobilebert_glue_tiki_rte_uimfalse_te64_test` | 0/1 | **N/A** | - |
| 16 | `mobilebert_glue_tiki_qnli_bs64_sgd_nowd_nomom_nonest_all_...` | 0/1 | **N/A** | - |
| 17 | `mobilebert_glue_tiki_mnli_bs64_sgd_nowd_nomom_nonest_all_tpe` | 0/1 | **N/A** | - |
| 18 | `mobilebert_glue_tiki_qqp_bs64_sgd_nowd_nomom_nonest_all_tpe` | 0/4 | **N/A** | - |
| 19 | `mobilebert_glue_tiki_qqp_bs64_sgd_nowd_nomom_nonest_all_l...` | 0/4 | **N/A** | - |
| 20 | `mobilebert_glue_tiki_rte_bs64_sgd_nowd_nomom_nonest_ffn` | 0/1 | **N/A** | - |
| 21 | `mobilebert_glue_tiki_qnli_bs64_sgd_nowd_nomom_nonest_all_tpe` | 0/1 | **N/A** | - |
| 22 | `mobilebert_glue_tiki_mnli_bs64_sgd_nowd_nomom_nonest_all_...` | 0/1 | **N/A** | - |

#### 4.3.5 TikiTaka v2 GLUE (10 studies)

TikiTaka v2 preset 설정으로 RTE 반복 실험 (run1~5, seq1~5).

| # | Study Name | Trials (Complete/Total) | Best Value | Key Parameters |
|---|-----------|------------------------|------------|----------------|
| 1 | `tiki2_rte_preset_run5` | 1/1 | **N/A** | - |
| 2 | `tiki2_rte_preset_seq5` | 1/1 | **N/A** | - |
| 3 | `tiki2_rte_preset_run1` | 1/1 | **N/A** | - |
| 4 | `tiki2_rte_preset_seq3` | 1/1 | **N/A** | - |
| 5 | `tiki2_rte_preset_run3` | 1/1 | **N/A** | - |
| 6 | `tiki2_rte_preset_seq1` | 1/1 | **N/A** | - |
| 7 | `tiki2_rte_preset_seq4` | 1/1 | **N/A** | - |
| 8 | `tiki2_rte_preset_seq2` | 1/1 | **N/A** | - |
| 9 | `tiki2_rte_preset_run2` | 1/1 | **N/A** | - |
| 10 | `tiki2_rte_preset_run4` | 1/1 | **N/A** | - |

> 모든 study가 preset config로 1 trial씩 실행됨. `trial_values`가 비어 있어 중간값만 기록.

#### 4.3.6 TikiTaka v2 SQuAD (7 studies)

| # | Study Name | Trials (Complete/Total) | Best Value | Key Parameters |
|---|-----------|------------------------|------------|----------------|
| 1 | `mobilebert_squad_tiki2_bs64_sgd_nowd_nomom_nonest_qkv_uim...` | 11/12 | **73.50** | learning_rate=0.00000, transfer_lr=1.0000, fast_lr=0.00000 |
| 2 | `mobilebert_squad_tiki2_bs64_sgd_nowd_nomom_nonest_qkv_uim...` | 20/22 | **62.40** | learning_rate=0.00000, transfer_lr=1.0000, fast_lr=0.00000 |
| 3 | `mobilebert_squad_tiki2_bs64_sgd_nowd_nomom_nonest_qkv` | 3/4 | **20.52** | learning_rate=2.0000, fast_lr=0.00000, transfer_every=2.0000 |
| 4 | `mobilebert_squad_tiki2_bs64_sgd_nowd_nomom_nonest_qkv_no_...` | 0/1 | **10.28** | learning_rate=0.00000, transfer_lr=0.00000, fast_lr=1.0000 |
| 5 | `mobilebert_squad_tiki2_bs64_sgd_nowd_nomom_nonest_qkv_uim...` | 1/1 | **8.73** | learning_rate=1.0000, fast_lr=1.0000, transfer_every=1.0000 |
| 6 | `mobilebert_squad_tiki2_bs64_sgd_nowd_nomom_nonest_qkv_uim...` | 0/1 | **N/A** | - |
| 7 | `mobilebert_squad_tiki2_bs64_sgd_nowd_nomom_nonest_qkv_uimb` | 0/1 | **N/A** | - |

> Best F1: **73.50** (UIMB v4 config, 11/12 trials)

---
## 5. 태스크별 Best Results 비교

### GLUE Tasks

| Task | ALBERT LoRA | ALBERT TikiTaka | MobileBERT LoRA | MobileBERT TikiTaka v1 |
|------|------------|----------------|----------------|----------------------|
| SST2 | 0.9140 | 0.9174 | 0.8647 | 0.8498 |
| MRPC | 0.9151 | 0.9122 | 0.8525 | 0.8134 |
| STSB | 0.8851 | 0.8675 | N/A | -1.0000 |
| COLA | 0.4887 | 0.5139 | 0.3548 | 0.0959 |
| RTE | 0.7401 | 0.7329 | 0.6318 | 0.5451 |
| MNLI | N/A | N/A | 0.6792 | N/A |
| QQP | 0.7315 | N/A | N/A | N/A |
| QNLI | N/A | 0.8733 | N/A | N/A |

### SQuAD

| Model × Method | Best F1 | Study | Trials |
|---------------|---------|-------|--------|
| ALBERT LoRA | **85.00** | `albert_squad_lrtt_bs48_sgd_decay_nowd_nomom_n` | 13/16 |
| BERT TikiTaka | **87.49** | `bert_squad_tiki_all_los_lr001_10_20trials` | 4/5 |
| MobileBERT LoRA | **67.18** | `mobilebert_squad_lrtt_bs64_sgd_decay_nowd_nom` | 24/25 |
| MobileBERT TikiTaka v2 | **73.50** | `mobilebert_squad_tiki2_bs64_sgd_nowd_nomom_no` | 11/12 |

---
## 6. 하이퍼파라미터 패턴 분석

### 6.1 공통 패턴

- **Learning Rate:** 대부분 0.0001~0.001 범위에서 최적 (log scale 탐색)
- **Optimizer:** SGD (decay, no WD, no momentum, no Nesterov)가 LoRA-LRTT에서 주로 사용
- **Optimizer:** Adam이 TikiTaka GLUE에서 주로 사용
- **Batch Size:** 태스크 크기에 따라 16 (COLA/STSB) ~ 128 (QQP) 사용

### 6.2 LoRA-LRTT 패턴

- `warm_alpha` + `convnt_all` 조합이 가장 높은 성능 (ALBERT GLUE)
- `combos` config가 단일 config보다 일관적으로 우수
- QKV layer만 사용 시 all layer 대비 성능 하락 (MobileBERT)

### 6.3 TikiTaka 패턴

- `transfer_lr` 2~8 범위가 효과적
- `transfer_every` 20~160, 작은 태스크에서 낮은 값 선호
- `in_chop_prob` 0.02~0.035 최적
- `auto_granularity` 170~340 범위

### 6.4 실패 사례

- **STSB (Regression):** TikiTaka에서 음수 correlation 발생 → MSE loss explosion (MobileBERT v1: -1.0)
- **MobileBERT MRPC ablrwarm:** best_value=0.0 반복 → warm-up 설정 문제
- **Albert TikiTaka SQuAD:** 2 studies 모두 완료 trial 0개 → 학습 시작 실패
- **MobileBERT LoRA glue_trainscale:** 0.5252 → output scaling 학습이 도움 안 됨

---
## 7. TikiTaka v2 Sweep 분석 (tikitaka_sweep/)

**Date:** 2026-02-04~05 | **Model:** MobileBERT | **GPU:** NVIDIA H200

### Sweep 결과

| Task | Best Trial | Best Value | Metric | Digital Baseline | Gap |
|------|-----------|-----------|--------|-----------------|-----|
| RTE | 1 | 0.5199 | accuracy | 0.5632 | -4.3%p |
| MRPC | 0 | 0.6544 | F1 | 0.9004 | -24.6%p |
| COLA | 8 | 0.6932 | matthews | 0.0* | N/A |
| STSB | 0 | -0.0532 | spearman | 0.8771 | **FAILED** |

*COLA digital baseline 0.0은 측정 오류로 추정.

### Best 하이퍼파라미터

| Parameter | RTE | MRPC | COLA |
|-----------|-----|------|------|
| learning_rate | 0.000136 | 0.000131 | 0.000194 |
| transfer_lr | 2.378 | 7.36 | 2.637 |
| transfer_every | 20 | 160 | 74 |
| fast_lr | 0.487 | 0.86 | 0.457 |
| auto_granularity | 169.6 | 306.0 | 340.9 |
| in_chop_prob | 0.035 | 0.02 | 0.029 |

### STSB 실패 분석

- **근본 원인:** STSB는 regression 태스크 (0~5 연속값 예측)
- Analog output scale mismatch → MSE loss explosion
- Classification 태스크는 softmax 정규화로 안정, regression은 직접 예측으로 불안정
- **해결 방안:** output sigmoid × 5 scaling 또는 label 0~1 정규화

### Process Hang 분석

- COLA → SST2 전환 시 hang 발생 (n_jobs=2 deadlock)
- WandB session management 충돌
- **해결:** n_jobs=1로 재실행

### 미완료 태스크

| Task | 데이터 크기 |
|------|-----------|
| SST2 | 67K |
| QNLI | 105K |
| QQP | 364K |
| MNLI | 393K |
| SQuAD | 87K (10K subset) |

---
## 8. 스크립트 / 쉘 / 로그 인벤토리

### 8.1 Python 스크립트 (scripts/)

| 경로 | 용도 |
|------|------|
| `scripts/baseline/optuna_baseline_glue.py` | Frozen analog QKV baseline (GLUE) |
| `scripts/baseline/optuna_baseline_squad.py` | Frozen analog QKV baseline (SQuAD) |
| `scripts/albert/lora/` | ALBERT LoRA-LRTT Optuna 스크립트 |
| `scripts/albert/tiki/` | ALBERT TikiTaka Optuna 스크립트 |
| `scripts/bert/tiki/` | BERT TikiTaka Optuna 스크립트 |
| `scripts/mobilebert/lora/` | MobileBERT LoRA-LRTT Optuna 스크립트 |
| `scripts/mobilebert/tiki/` | MobileBERT TikiTaka Optuna 스크립트 |

### 8.2 Shell 스크립트 (shell_scripts/)

| 경로 | 용도 |
|------|------|
| `shell_scripts/baseline/` | Baseline 실행 스크립트 |
| `shell_scripts/albert/lora/glue/` | ALBERT LoRA GLUE 실행 |
| `shell_scripts/albert/lora/squad/` | ALBERT LoRA SQuAD 실행 |
| `shell_scripts/albert/tikitaka/glue/` | ALBERT TikiTaka GLUE 실행 |
| `shell_scripts/mobilebert/lora/glue/` | MobileBERT LoRA GLUE 실행 |
| `shell_scripts/mobilebert/tikitaka/glue/` | MobileBERT TikiTaka GLUE 실행 |

### 8.3 로그 (logs/)

| 경로 | 크기 | 내용 |
|------|------|------|
| `logs/baseline/` | 294M | Baseline 학습 로그 |
| `logs/albert/lora/glue/` | 1.8M | ALBERT LoRA GLUE 로그 |
| `logs/albert/tikitaka/glue/` | 8K | ALBERT TikiTaka GLUE 로그 |
| `logs/bert/tikitaka/squad/` | 64M | BERT TikiTaka SQuAD 로그 |
| `logs/mobilebert/lora/glue/` | 4K | MobileBERT LoRA GLUE 로그 |
| `logs/mobilebert/tikitaka/glue/` | 166M | MobileBERT TikiTaka GLUE 로그 |

### 8.4 Optuna DB 파일 분포

| 카테고리 | DB 수 | 유효 Studies |
|---------|------|-------------|
| `results/_misc/analoglora_alllayer/` | 1 | 1 |
| `results/albert/lora/glue/` | 44 | 42 |
| `results/albert/lora/glue_2stage/` | 21 | 17 |
| `results/albert/lora/squad/` | 5 | 4 |
| `results/albert/tikitaka/glue/` | 54 | 51 |
| `results/albert/tikitaka/squad/` | 2 | 0 |
| `results/bert/tikitaka/squad/` | 47 | 29 |
| `results/mobilebert/lora/glue/` | 42 | 34 |
| `results/mobilebert/lora/glue_trainscale/` | 1 | 1 |
| `results/mobilebert/lora/squad/` | 9 | 8 |
| `results/mobilebert/tikitaka/glue_v1/` | 22 | 14 |
| `results/mobilebert/tikitaka/glue_v2/` | 10 | 0 |
| `results/mobilebert/tikitaka/squad_v2/` | 7 | 5 |
| **Total** | **265** | **206** |

---

*이 문서는 `/data/sub_results/results/` 내 336개 Optuna DB 파일에서 자동 추출하여 생성되었습니다.*