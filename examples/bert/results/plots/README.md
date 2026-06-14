# BERT SQuAD LRTT — plots/ 구조

LRTT (Low-Rank TikiTaka) BERT SQuAD fine-tuning 실험들의 분석 + 논문 figure.

## 폴더 구성

```
plots/
├── paper/        # 논문 figure (fig5*, fig6*, figS*) — final/manuscript
├── paper_src/    # paper sources (tex + bib)
├── scripts/      # launch + analysis scripts (investigate_*, replicate_*, run_sweep_*, plot_*)
├── analysis/     # 실험별 데이터 + 분석 plot + README (번호 순서로 진행)
│   ├── 01_*  ... 02_*  ...  11_* (각 실험)
│   ├── _etc_misc/        # 기타 잡다한 분석
│   └── _global_outputs/  # cross-experiment 결과 (LoRA 비교, threshold plot, 종합 보고서)
└── EXPERIMENT_NOTES.md   # 실험 노트 (시계열)
```

## 분석 실험 (analysis/)

각 폴더 prefix `NN_` 는 진행 시간 순서. 폴더명 format:
`{NN}_{purpose}__{device}__{varied}__{seed}_{YYYYMMDD}`

| Folder | 핵심 질문 | 결과 요약 |
|---|---|---|
| 01_noise_asymmetry | A/B noise (6t1c) 비대칭이 학습/collapse에 미치는 영향 | both noise만 epoch 1 collapse |
| 02_noise_asymmetry_multitile | 위와 같으나 L0/L6/L11 multi-tile diag 추가 | both collapse 재현, L11 폭발 확인 |
| 03_autoscale_failure | AUTO_SCALE_MODE=separate가 collapse를 막는가 | 4/4 모두 Type 2 failure (F1 ~7%) |
| 04_lrfloor_variants | MIN_LR_RATE 영향 (seed/lr_floor 변형) | lrfloor001에서 결정적 collapse |
| 05_prepull_vs_current | LRTT controller prepull 버전 비교 | controller 차이 검증 |
| 06_bilinear_hypothesis_posthoc | 기존 데이터로 bilinear 가설 사후 검증 | trajectory analysis |
| 07_gcoh_fastlr_ablation | fast_lr 0.474/0.1/0.05/0.01 ablation + G coherence | dose-response 확인 |
| 08_gcoh_seedstats | minimal-diag seed 43/44/45 통계 | seed44/45 stuck pattern 발견 |
| 09_gcoh_fi_true | FORWARD_INJECT=True 단일 seed test | epoch 2 collapse (single seed 결론 무효) |
| 10_gcoh_fastlr_push | fast_lr 0.7/1.0으로 강제 cascade | dose-response cascade onset 입증 |
| 11_ideal_multiseed | constantstepideal에서도 같은 collapse 일어나는가 | 진행 중 |

## 논문 figure (paper/)

- fig5f: AB weight dynamics
- fig5g: C bit sweep
- fig5h: AB bit sweep
- fig5i: cosine similarity
- fig6a: target comparison (qkvo/ffn/all)
- fig6b: rank sweep
- fig6c: noise asymmetry summary
- fig6d: erank delta
- figS7-S11: supplementary

각 figure 4 file: `.py` (script) + `.png` (raster) + `.svg` (vector) + `_data.json` (data)
