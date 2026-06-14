# Bilinear unstable mode 가설 post-hoc 검증 (기존 데이터)

**Date**: 2026-05-12  
**Launch**: scripts/verify_bilinear_collapse.py (no new GPU run)

## Question
collapse가 bilinear update structure (ΔA = -η·G·B^T, ΔB = -η·A^T·G)의
unstable mode 동작인지를 **기존 데이터로** 사후 분석

## Method
- 04번 (no_noise_variants) 데이터 사용
- seed42 (stable) vs seed42_lrfloor001 (collapse) trajectory 비교
- ‖x‖, ‖d‖ 외부 신호는 일정한데 ‖XB‖, ‖DA‖만 폭발 → bilinear amplification signature

## Result Summary
- ‖XB‖, ‖DA‖ trajectory에서 bilinear amplification 패턴 확인
- 가설과 consistent (단, sufficient 증명 아님)

## Files
- trajectory_seed42.png — stable case trajectory
- trajectory_seed42_lrfloor001.png — collapse case
- comparison_amplification.png — amplification ratio comparison
