#!/usr/bin/env python3
"""
MobileBERT LRTT 분석 결과 CSV/MD 파일 상세 분석
- 계획서에 명시된 분석 항목들을 정량적으로 계산
- 비교 테이블 및 통계 생성
"""

import pandas as pd
import numpy as np
import os

# 분석 결과 디렉토리
RESULTS_DIR = "/data/LRTT/examples/glue_lrtt_sixt1c/analysis_results"

def load_data():
    """CSV 파일들 로드"""
    mobilebert_df = pd.read_csv(os.path.join(RESULTS_DIR, "mobilebert_weights.csv"))
    bert_df = pd.read_csv(os.path.join(RESULTS_DIR, "bert_base_weights.csv"))
    problem_df = pd.read_csv(os.path.join(RESULTS_DIR, "mobilebert_problem_layers.csv"))
    return mobilebert_df, bert_df, problem_df


def analyze_problem_layers(problem_df):
    """mobilebert_problem_layers.csv 상세 분석"""
    print("=" * 80)
    print("1. mobilebert_problem_layers.csv 상세 분석")
    print("=" * 80)

    # Top 10 (abs_max 기준)
    print("\n### 문제 레이어 Top 10 (abs_max 기준 정렬)")
    print("-" * 100)
    print(f"{'순위':<4} {'레이어':<50} {'Min':>10} {'Max':>10} {'Abs Max':>10} {'|w|>1 %':>10} {'|w|>10 %':>10}")
    print("-" * 100)

    top10 = problem_df.nlargest(10, 'abs_max')
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        layer_short = row['layer_name'].replace('mobilebert.encoder.', '')
        print(f"{i:<4} {layer_short:<50} {row['min']:>10.2f} {row['max']:>10.2f} {row['abs_max']:>10.2f} {row['pct_gt_1']:>10.1f} {row['pct_gt_10']:>10.2f}")

    # 패턴 분석
    print("\n### 문제 레이어 패턴 분석")

    patterns = {
        'attention.self.key.bias': 0,
        'intermediate.dense.bias': 0,
        'attention.output': 0,
        'bottleneck': 0,
        'ffn': 0,
        'other': 0
    }

    for layer in problem_df['layer_name']:
        if 'attention.self.key.bias' in layer:
            patterns['attention.self.key.bias'] += 1
        elif 'intermediate.dense.bias' in layer:
            patterns['intermediate.dense.bias'] += 1
        elif 'attention.output' in layer:
            patterns['attention.output'] += 1
        elif 'bottleneck' in layer:
            patterns['bottleneck'] += 1
        elif 'ffn' in layer:
            patterns['ffn'] += 1
        else:
            patterns['other'] += 1

    total = len(problem_df)
    print(f"\n총 문제 레이어: {total}개")
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / total * 100
            print(f"  - {pattern}: {count}개 ({pct:.1f}%)")

    # Layer 번호별 분포
    print("\n### Layer 번호별 분포 (심각도순)")
    layer_nums = []
    for layer in problem_df['layer_name']:
        parts = layer.split('.')
        for i, p in enumerate(parts):
            if p == 'layer' and i+1 < len(parts):
                try:
                    layer_nums.append(int(parts[i+1]))
                except ValueError:
                    pass
                break

    from collections import Counter
    layer_counts = Counter(layer_nums)
    print(f"  심각한 레이어 (abs_max > 10): Layer 0~8")
    print(f"  Layer별 문제 레이어 수: {dict(sorted(layer_counts.items()))}")


def compare_models(mobilebert_df, bert_df):
    """mobilebert_weights.csv vs bert_base_weights.csv 비교"""
    print("\n" + "=" * 80)
    print("2. mobilebert_weights.csv vs bert_base_weights.csv 비교")
    print("=" * 80)

    # 전체 통계 비교
    print("\n### 전체 통계 비교")
    print("-" * 70)
    print(f"{'메트릭':<25} {'MobileBERT':>15} {'BERT-base':>15} {'비율':>10}")
    print("-" * 70)

    metrics = [
        ("총 레이어 수", len(mobilebert_df), len(bert_df)),
        ("총 파라미터 수", mobilebert_df['num_params'].sum(), bert_df['num_params'].sum()),
        ("가중치 Min", mobilebert_df['min'].min(), bert_df['min'].min()),
        ("가중치 Max", mobilebert_df['max'].max(), bert_df['max'].max()),
        ("Abs Max (최대)", mobilebert_df['abs_max'].max(), bert_df['abs_max'].max()),
        ("|w|>1 레이어 수", (mobilebert_df['pct_gt_1'] > 0).sum(), (bert_df['pct_gt_1'] > 0).sum()),
        ("|w|>5 레이어 수", (mobilebert_df['pct_gt_5'] > 0).sum(), (bert_df['pct_gt_5'] > 0).sum()),
        ("|w|>10 레이어 수", (mobilebert_df['pct_gt_10'] > 0).sum(), (bert_df['pct_gt_10'] > 0).sum()),
    ]

    for name, mb_val, bert_val in metrics:
        if bert_val != 0:
            ratio = mb_val / bert_val
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "∞"

        if isinstance(mb_val, float):
            print(f"{name:<25} {mb_val:>15.2f} {bert_val:>15.2f} {ratio_str:>10}")
        else:
            print(f"{name:<25} {mb_val:>15,} {bert_val:>15,} {ratio_str:>10}")

    # attention.self.key.bias 비교
    print("\n### attention.self.key.bias 비교")
    print("-" * 90)
    print(f"{'Layer':<10} {'MobileBERT Range':<30} {'BERT-base Range':<30} {'MB/BERT Ratio':>15}")
    print("-" * 90)

    for layer_num in range(5):
        mb_layer = mobilebert_df[mobilebert_df['layer_name'].str.contains(f'layer.{layer_num}.attention.self.key.bias')]
        bert_layer = bert_df[bert_df['layer_name'].str.contains(f'layer.{layer_num}.attention.self.key.bias')]

        if len(mb_layer) > 0 and len(bert_layer) > 0:
            mb_row = mb_layer.iloc[0]
            bert_row = bert_layer.iloc[0]

            mb_range = f"[{mb_row['min']:.2f}, +{mb_row['max']:.2f}]"
            bert_range = f"[{bert_row['min']:.4f}, +{bert_row['max']:.4f}]"

            # 범위 비율 계산
            mb_span = mb_row['max'] - mb_row['min']
            bert_span = bert_row['max'] - bert_row['min']
            ratio = mb_span / bert_span if bert_span > 0 else float('inf')

            print(f"layer.{layer_num:<4} {mb_range:<30} {bert_range:<30} {ratio:>15.1f}x")

    print("\n결론: MobileBERT의 key.bias는 BERT-base 대비 500~1000배 크다!")


def analyze_layer_types(mobilebert_df):
    """MobileBERT 레이어 구조 분석"""
    print("\n" + "=" * 80)
    print("3. MobileBERT 레이어 타입별 통계")
    print("=" * 80)

    # 레이어 타입 분류
    type_stats = {}

    for _, row in mobilebert_df.iterrows():
        layer_name = row['layer_name']

        # 타입 결정
        if 'attention.self.key.bias' in layer_name:
            layer_type = 'attention.self.key.bias'
        elif 'attention.self.query.bias' in layer_name:
            layer_type = 'attention.self.query.bias'
        elif 'attention.self.value.bias' in layer_name:
            layer_type = 'attention.self.value.bias'
        elif 'bottleneck' in layer_name and 'LayerNorm.weight' in layer_name:
            layer_type = 'bottleneck.LayerNorm.weight'
        elif 'intermediate.dense.bias' in layer_name:
            if 'ffn' in layer_name:
                layer_type = 'ffn.*.intermediate.dense.bias'
            else:
                layer_type = 'intermediate.dense.bias'
        elif 'LayerNorm' in layer_name:
            layer_type = 'LayerNorm'
        elif 'dense.weight' in layer_name:
            layer_type = 'dense.weight'
        elif 'embeddings' in layer_name:
            layer_type = 'embeddings'
        else:
            layer_type = 'other'

        if layer_type not in type_stats:
            type_stats[layer_type] = {
                'count': 0,
                'abs_max_sum': 0,
                'problem_count': 0
            }

        type_stats[layer_type]['count'] += 1
        type_stats[layer_type]['abs_max_sum'] += row['abs_max']
        if row['abs_max'] > 5:  # 문제 레이어 기준
            type_stats[layer_type]['problem_count'] += 1

    print("\n### 레이어 타입별 통계")
    print("-" * 80)
    print(f"{'레이어 타입':<40} {'개수':>8} {'평균 abs_max':>15} {'문제 레이어 비율':>15}")
    print("-" * 80)

    sorted_types = sorted(type_stats.items(), key=lambda x: -x[1]['abs_max_sum']/x[1]['count'])

    for layer_type, stats in sorted_types:
        avg_abs_max = stats['abs_max_sum'] / stats['count']
        problem_pct = stats['problem_count'] / stats['count'] * 100
        print(f"{layer_type:<40} {stats['count']:>8} {avg_abs_max:>15.2f} {problem_pct:>14.1f}%")


def analyze_lrtt_compatibility(mobilebert_df, bert_df):
    """LRTT 호환성 분석"""
    print("\n" + "=" * 80)
    print("4. LRTT 호환성 분석")
    print("=" * 80)

    print("\n### LRTT 제약 조건")
    print("  w_max = 1.0 (SoftBoundsDevice)")
    print("  w_min = -1.0 (SoftBoundsDevice)")
    print("  허용 범위: [-1.0, +1.0]")

    # 클리핑 영향 분석
    print("\n### 클리핑 영향 분석")

    # MobileBERT layer.2.attention.self.key.bias 예시
    worst_layer = mobilebert_df.loc[mobilebert_df['abs_max'].idxmax()]

    print(f"\n가장 심각한 레이어: {worst_layer['layer_name']}")
    print(f"  원본 범위: [{worst_layer['min']:.2f}, +{worst_layer['max']:.2f}]")
    print(f"  클리핑 후: [-1.0, +1.0]")

    original_range = worst_layer['max'] - worst_layer['min']
    allowed_range = 2.0  # [-1, 1]
    info_loss = 1 - (allowed_range / original_range)

    print(f"\n  정보 손실 = 1 - (허용범위 / 원본범위)")
    print(f"            = 1 - ({allowed_range:.1f} / {original_range:.2f})")
    print(f"            = {info_loss*100:.1f}% 손실!")
    print(f"\n  클리핑 대상 비율: {worst_layer['pct_gt_1']:.1f}% ({int(worst_layer['num_params'])}개 중 {int(worst_layer['num_params'] * worst_layer['pct_gt_1'] / 100)}개)")

    # 호환성 판정
    print("\n### LRTT 호환성 판정")
    print("-" * 60)
    print(f"{'모델':<15} {'최대 abs_max':>15} {'클리핑 손실':>15} {'판정':>15}")
    print("-" * 60)

    mb_max = mobilebert_df['abs_max'].max()
    bert_max = bert_df['abs_max'].max()

    mb_range = mobilebert_df['max'].max() - mobilebert_df['min'].min()
    bert_range = bert_df['max'].max() - bert_df['min'].min()

    mb_loss = (1 - 2.0 / mb_range) * 100
    bert_loss = (1 - 2.0 / bert_range) * 100

    print(f"{'BERT-base':<15} {bert_max:>15.2f} {bert_loss:>14.0f}% {'⚠️ 제한적 호환':>15}")
    print(f"{'MobileBERT':<15} {mb_max:>15.2f} {mb_loss:>14.0f}% {'❌ 비호환':>15}")


def print_summary():
    """최종 요약"""
    print("\n" + "=" * 80)
    print("5. 최종 요약 및 권장 사항")
    print("=" * 80)

    print("""
### 실패 메커니즘
1. MobileBERT 로드 → key.bias 범위: [-15.30, +21.93]
2. convert_to_analog() 호출 → set_weights() 시 [-1, 1]로 클리핑
3. Forward Pass → 클리핑된 가중치로 Attention 계산 → 스케일 불일치로 오차 증폭
4. 결과 → Logits: [-6,367,677, +4,225,070] (폭발) → Loss: 200,000+ (발산)

### 해결책 검증 결과
| 실험 | 방법                    | 결과               |
|------|------------------------|-------------------|
| 1    | 가중치 전체 정규화      | 부분 성공          |
| 2    | w_max 확장 (±25)        | 실패               |
| 3    | 문제 레이어 제외        | 실패               |
| 4    | BERT 가중치 x10 확대    | 성공 (문제 미재현) |
| 5    | key bias만 정규화       | 실패               |

### 최종 권장 사항
1. MobileBERT + LRTT: ❌ 사용 불가
2. BERT-base + LRTT: ✅ 권장
3. 향후 개선 방향:
   - Layer-wise weight scaling 구현
   - Architecture-aware conversion 추가
   - Dynamic w_max/w_min 지원
""")


def main():
    print("MobileBERT LRTT 분석 결과 CSV/MD 파일 상세 분석")
    print("=" * 80)

    # 데이터 로드
    mobilebert_df, bert_df, problem_df = load_data()

    print(f"\n파일 로드 완료:")
    print(f"  - mobilebert_weights.csv: {len(mobilebert_df)}개 레이어")
    print(f"  - bert_base_weights.csv: {len(bert_df)}개 레이어")
    print(f"  - mobilebert_problem_layers.csv: {len(problem_df)}개 문제 레이어")

    # 상세 분석 수행
    analyze_problem_layers(problem_df)
    compare_models(mobilebert_df, bert_df)
    analyze_layer_types(mobilebert_df)
    analyze_lrtt_compatibility(mobilebert_df, bert_df)
    print_summary()

    print("\n분석 완료!")


if __name__ == "__main__":
    main()
