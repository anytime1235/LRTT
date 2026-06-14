# Prepull vs current LRTT controller 비교

**Date**: 2026-05-07  
**Launch**: scripts/investigate_prepull_vs_current.py

## Question
최근 LRTT controller (git pull 후) 변경이 collapse 양상에 영향?
→ prepull controller (이전 버전) 와 current controller에서 같은 결과인지

## Setup
- no_noise, default hyperparams (seed=42, 43)
- prepull 버전: src/aihwkit/simulator/tiles/lrtt_controller_prepull.py (local backup)
- current 버전: src/aihwkit/simulator/tiles/lrtt_controller.py (HEAD)

## Variables
- controller 버전 (prepull vs current)
- seed (42, 43)
4 combinations total

## Result Summary
- 두 버전에서 collapse 양상 비슷
- controller 변경이 collapse 직접 원인 아님 확인

## Files
- runlog_prepull_check_*.txt (4 runs)
- summary_20260507_211603.json
