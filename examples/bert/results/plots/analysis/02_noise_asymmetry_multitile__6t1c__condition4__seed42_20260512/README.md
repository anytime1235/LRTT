# Noise asymmetry — 4-condition multi-tile diag (Phase 2 replication)

**Date**: 2026-05-12  
**Launch**: scripts/replicate_4conditions_diag.py with MULTI_TILE_DIAG=True

## Question
01번 실험과 같은 조건이지만 L0/L6/L11 × {query, key, value, output} 모두 추적
→ collapse가 어느 layer에서 발생하는지 식별

## Setup
01번과 동일하지만 **MULTI_TILE_DIAG=True**, ERANK_RATE_LIMIT_STEPS=10

## Variables
01번과 동일 4-condition

## Result Summary
- 01번 결과 재현 (F1 분포 ±0.6%p 일치)
- **L11.attention.output.dense에서 collapse cascade 관측** (no_noise epoch 5에 ‖A·B‖ 22500까지 폭발)
- 다른 layer는 stable

## Files
- diag_{no_noise,a_only,b_only,both}.json — per-condition multi-tile diag
- diag_plot{1..5}{,_L11}.{png,svg} — L0 (first) 및 L11 (last) 비교
- cross_run_divergence_L11.png — 같은 seed 다른 outcome (chaotic divergence)
- f1_comparison.png/.svg — 4-condition F1 bar chart
- summary_20260512_095208.json — orchestrator summary
