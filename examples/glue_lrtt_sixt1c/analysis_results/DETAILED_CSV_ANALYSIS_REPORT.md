# MobileBERT LRTT 분석 결과 CSV/MD 파일 상세 분석 보고서

## 분석 일자
2026-02-03

## 분석 대상 파일

| 파일 | 위치 | 설명 |
|------|------|------|
| `mobilebert_weights.csv` | `analysis_results/` | MobileBERT 전체 레이어 가중치 통계 (1113개 레이어) |
| `bert_base_weights.csv` | `analysis_results/` | BERT-base 전체 레이어 가중치 통계 (201개 레이어) |
| `mobilebert_problem_layers.csv` | `analysis_results/` | MobileBERT 문제 레이어 목록 (30개) |
| `MOBILEBERT_LRTT_ANALYSIS_REPORT.md` | `analysis_results/` | 최종 분석 보고서 |

---

## 1. 문제 레이어 상세 분석 (mobilebert_problem_layers.csv)

### 문제 레이어 Top 10 (abs_max 기준)

| 순위 | 레이어 | Min | Max | Abs Max | |w|>1 % | |w|>10 % |
|------|--------|-----|-----|---------|--------|---------|
| 1 | layer.2.attention.self.key.bias | -15.30 | 21.93 | 21.93 | 82.8% | 8.59% |
| 2 | layer.4.attention.self.key.bias | -13.10 | 16.77 | 16.77 | 89.8% | 7.03% |
| 3 | layer.6.attention.self.key.bias | -13.71 | 15.42 | 15.42 | 71.9% | 3.91% |
| 4 | layer.8.attention.self.key.bias | -14.30 | 10.38 | 14.30 | 74.2% | 3.12% |
| 5 | layer.0.attention.self.key.bias | -9.77 | 13.91 | 13.91 | 73.4% | 1.56% |
| 6 | layer.3.attention.self.key.bias | -11.38 | 11.69 | 11.69 | 82.0% | 1.56% |
| 7 | layer.7.attention.self.key.bias | -10.76 | 11.22 | 11.22 | 64.1% | 1.56% |
| 8 | layer.1.attention.self.key.bias | -11.04 | 10.67 | 11.04 | 74.2% | 1.56% |
| 9 | layer.5.attention.self.key.bias | -10.28 | 8.87 | 10.28 | 75.8% | 1.56% |
| 10 | layer.23.intermediate.dense.bias | -7.54 | 0.51 | 7.54 | 41.2% | 0.00% |

### 핵심 발견

1. **attention.self.key.bias가 주요 문제** (23개 레이어, 전체 문제 레이어의 76.7%)
2. **Layer 0~8이 가장 심각** (abs_max > 10)
3. **Layer 2가 최악** (max=+21.93, min=-15.30)
4. **클리핑 비율 매우 높음**: 70~90%의 가중치가 |w|>1

---

## 2. 모델 비교 분석 (mobilebert_weights.csv vs bert_base_weights.csv)

### 전체 통계 비교

| 메트릭 | MobileBERT | BERT-base | 비율 |
|--------|-----------|-----------|------|
| 총 레이어 수 | 1113 | 201 | 5.54x |
| 총 파라미터 수 | 24,582,914 | 109,483,778 | - |
| 가중치 Min | -15.30 | -6.82 | - |
| 가중치 Max | 21.93 | 3.74 | - |
| |w|>1 레이어 수 | 368 | 48 | - |
| |w|>5 레이어 수 | 30 | 4 | - |
| |w|>10 레이어 수 | 9 | 0 | - |

### attention.self.key.bias 레이어별 비교

| Layer | MobileBERT Range | BERT-base Range | 배율 |
|-------|------------------|-----------------|------|
| layer.0 | [-9.77, +13.91] | [-0.0124, +0.0106] | 1029x |
| layer.1 | [-11.04, +10.67] | [-0.0280, +0.0221] | 433x |
| layer.2 | [-15.30, +21.93] | [-0.0160, +0.0271] | 864x |
| layer.3 | [-11.38, +11.69] | [-0.0219, +0.0215] | 532x |
| layer.4 | [-13.10, +16.77] | [-0.0175, +0.0221] | 753x |

**결론**: MobileBERT의 key.bias는 BERT-base 대비 **500~1000배** 크다!

---

## 3. LRTT 호환성 분석

### LRTT 제약 조건

| 제약 | 값 | 출처 |
|------|---|------|
| w_max | 1.0 | SoftBoundsDevice |
| w_min | -1.0 | SoftBoundsDevice |
| 허용 범위 | [-1.0, +1.0] | set_weights() 클리핑 |

### 클리핑 영향 분석

**MobileBERT layer.2.attention.self.key.bias 예시:**

```
원본 범위: [-15.30, +21.93]
클리핑 후: [-1.0, +1.0]

정보 손실 = 1 - (허용범위 / 원본범위)
          = 1 - (2 / 37.23)
          = 94.6% 손실!

클리핑 대상 비율: 82.8% (128개 중 106개)
```

### LRTT 호환성 판정

| 모델 | 최대 abs_max | 클리핑 손실 | 판정 |
|------|------------|-----------|------|
| BERT-base | 6.82 | ~81% | ⚠️ 제한적 호환 |
| MobileBERT | **21.93** | **~95%** | ❌ **비호환** |

---

## 4. 실패 메커니즘 요약

```
1. MobileBERT 로드
   └── key.bias 범위: [-15.30, +21.93]

2. convert_to_analog() 호출
   └── set_weights() 시 [-1, 1]로 클리핑

3. Forward Pass
   └── 클리핑된 가중치로 Attention 계산
   └── 스케일 불일치로 오차 증폭

4. 결과
   └── Logits: [-6,367,677, +4,225,070] (폭발)
   └── Loss: 200,000+ (발산)
   └── Accuracy: ~50% (랜덤)
```

---

## 5. 해결책 검증 결과

| 실험 | 방법 | 결과 |
|------|------|------|
| 1 | 가중치 전체 정규화 | 부분 성공 (LRTT 변환 후 실패) |
| 2 | w_max 확장 (±25) | 실패 |
| 3 | 문제 레이어 제외 | 실패 |
| 4 | BERT 가중치 x10 확대 | 성공 (문제 미재현) |
| 5 | key bias만 정규화 | 실패 |

---

## 6. 최종 권장 사항

1. **MobileBERT + LRTT**: ❌ **사용 불가**
2. **BERT-base + LRTT**: ✅ **권장**
3. **향후 개선 방향**:
   - Layer-wise weight scaling 구현
   - Architecture-aware conversion 추가
   - Dynamic w_max/w_min 지원

---

## 7. 생성된 시각화

- `comprehensive_comparison.png` - 모델 비교 종합 차트
- `key_bias_heatmap.png` - Key bias 레이어별 히트맵
- `weight_distribution_comparison.png` - 가중치 분포 비교

---

**분석 완료일:** 2026-02-03
**분석자:** Claude Code
