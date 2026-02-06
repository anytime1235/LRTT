#!/usr/bin/env python
# coding=utf-8
"""Phase 1: 가중치 범위 상세 분석 - MobileBERT vs BERT-base 비교"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import os

# Results directory
RESULTS_DIR = "/data/LRTT/examples/glue_lrtt_sixt1c/analysis_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def analyze_model_weights(model_name: str) -> pd.DataFrame:
    """모델의 모든 Linear 레이어 가중치 통계 분석"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")

    model_config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config, use_safetensors=True
    )

    results = []

    for name, param in model.named_parameters():
        w = param.data.cpu().numpy().flatten()

        # 가중치 통계 계산
        stats = {
            'layer_name': name,
            'shape': str(param.shape),
            'num_params': len(w),
            'min': np.min(w),
            'max': np.max(w),
            'mean': np.mean(w),
            'std': np.std(w),
            'abs_mean': np.mean(np.abs(w)),
            'abs_max': np.max(np.abs(w)),
            'pct_gt_1': 100 * np.sum(np.abs(w) > 1) / len(w),  # |w| > 1인 비율
            'pct_gt_5': 100 * np.sum(np.abs(w) > 5) / len(w),  # |w| > 5인 비율
            'pct_gt_10': 100 * np.sum(np.abs(w) > 10) / len(w),  # |w| > 10인 비율
        }
        results.append(stats)

    df = pd.DataFrame(results)
    return df


def compare_models():
    """MobileBERT vs BERT-base 비교 분석"""

    # MobileBERT 분석
    mobilebert_df = analyze_model_weights("google/mobilebert-uncased")
    mobilebert_df['model'] = 'MobileBERT'

    # BERT-base 분석
    bert_df = analyze_model_weights("bert-base-uncased")
    bert_df['model'] = 'BERT-base'

    # 결과 저장
    mobilebert_df.to_csv(f"{RESULTS_DIR}/mobilebert_weights.csv", index=False)
    bert_df.to_csv(f"{RESULTS_DIR}/bert_base_weights.csv", index=False)

    return mobilebert_df, bert_df


def print_summary(mobilebert_df: pd.DataFrame, bert_df: pd.DataFrame):
    """요약 통계 출력"""

    print("\n" + "="*80)
    print("PHASE 1: 가중치 범위 분석 요약")
    print("="*80)

    # 전체 통계
    print("\n[1] 전체 가중치 범위 비교")
    print("-"*60)
    print(f"{'Metric':<20} {'MobileBERT':>15} {'BERT-base':>15}")
    print("-"*60)
    print(f"{'Min':<20} {mobilebert_df['min'].min():>15.4f} {bert_df['min'].min():>15.4f}")
    print(f"{'Max':<20} {mobilebert_df['max'].max():>15.4f} {bert_df['max'].max():>15.4f}")
    print(f"{'Abs Max':<20} {mobilebert_df['abs_max'].max():>15.4f} {bert_df['abs_max'].max():>15.4f}")

    # |w| > 1인 레이어 개수
    mb_gt1 = (mobilebert_df['abs_max'] > 1).sum()
    bert_gt1 = (bert_df['abs_max'] > 1).sum()
    print(f"{'Layers |w|>1':<20} {mb_gt1:>15} {bert_gt1:>15}")

    # |w| > 5인 레이어 개수
    mb_gt5 = (mobilebert_df['abs_max'] > 5).sum()
    bert_gt5 = (bert_df['abs_max'] > 5).sum()
    print(f"{'Layers |w|>5':<20} {mb_gt5:>15} {bert_gt5:>15}")

    # |w| > 10인 레이어 개수
    mb_gt10 = (mobilebert_df['abs_max'] > 10).sum()
    bert_gt10 = (bert_df['abs_max'] > 10).sum()
    print(f"{'Layers |w|>10':<20} {mb_gt10:>15} {bert_gt10:>15}")

    # 문제 레이어 식별 (MobileBERT)
    print("\n[2] MobileBERT 문제 레이어 (|w| > 5)")
    print("-"*80)
    problem_layers = mobilebert_df[mobilebert_df['abs_max'] > 5].sort_values('abs_max', ascending=False)

    for _, row in problem_layers.iterrows():
        print(f"{row['layer_name']}")
        print(f"    Range: [{row['min']:.4f}, {row['max']:.4f}], |Max|: {row['abs_max']:.4f}")
        print(f"    Shape: {row['shape']}, Params: {row['num_params']:,}")
        print()

    # 레이어 타입별 분류
    print("\n[3] MobileBERT 문제 레이어 패턴 분석")
    print("-"*60)

    # 패턴별 그룹화
    patterns = {
        'attention.self.query': [],
        'attention.self.key': [],
        'attention.self.value': [],
        'attention.output': [],
        'intermediate': [],
        'output.dense': [],
        'bottleneck': [],
        'LayerNorm': [],
        'embedding': [],
        'other': []
    }

    for _, row in problem_layers.iterrows():
        name = row['layer_name']
        matched = False
        for pattern in patterns.keys():
            if pattern in name and pattern != 'other':
                patterns[pattern].append((name, row['abs_max']))
                matched = True
                break
        if not matched:
            patterns['other'].append((name, row['abs_max']))

    for pattern, layers in patterns.items():
        if layers:
            print(f"\n{pattern} ({len(layers)} layers):")
            for name, abs_max in sorted(layers, key=lambda x: -x[1])[:5]:  # Top 5
                print(f"    {abs_max:.4f}: {name}")

    return problem_layers


def plot_weight_distribution(mobilebert_df: pd.DataFrame, bert_df: pd.DataFrame):
    """가중치 범위 히스토그램 시각화"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Abs Max 분포 비교
    ax1 = axes[0, 0]
    ax1.hist(mobilebert_df['abs_max'], bins=50, alpha=0.5, label='MobileBERT', color='red')
    ax1.hist(bert_df['abs_max'], bins=50, alpha=0.5, label='BERT-base', color='blue')
    ax1.axvline(x=1.0, color='green', linestyle='--', label='LRTT w_max=1.0')
    ax1.set_xlabel('Max Absolute Weight')
    ax1.set_ylabel('Number of Layers')
    ax1.set_title('Layer-wise Max |Weight| Distribution')
    ax1.legend()
    ax1.set_xlim(0, 25)

    # 2. Log scale로 Abs Max
    ax2 = axes[0, 1]
    ax2.hist(np.log10(mobilebert_df['abs_max'] + 1e-10), bins=50, alpha=0.5, label='MobileBERT', color='red')
    ax2.hist(np.log10(bert_df['abs_max'] + 1e-10), bins=50, alpha=0.5, label='BERT-base', color='blue')
    ax2.axvline(x=0, color='green', linestyle='--', label='w_max=1.0')
    ax2.set_xlabel('Log10(Max |Weight|)')
    ax2.set_ylabel('Number of Layers')
    ax2.set_title('Log Scale Max |Weight| Distribution')
    ax2.legend()

    # 3. MobileBERT 레이어별 abs_max (상위 30개)
    ax3 = axes[1, 0]
    top30 = mobilebert_df.nlargest(30, 'abs_max')
    y_pos = np.arange(len(top30))
    ax3.barh(y_pos, top30['abs_max'], color='red', alpha=0.7)
    ax3.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='LRTT w_max=1.0')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([name.split('.')[-2] + '.' + name.split('.')[-1]
                         for name in top30['layer_name']], fontsize=6)
    ax3.set_xlabel('Max |Weight|')
    ax3.set_title('MobileBERT Top 30 Layers by |Weight|')
    ax3.legend()
    ax3.invert_yaxis()

    # 4. BERT-base 레이어별 abs_max (상위 30개)
    ax4 = axes[1, 1]
    top30_bert = bert_df.nlargest(30, 'abs_max')
    y_pos = np.arange(len(top30_bert))
    ax4.barh(y_pos, top30_bert['abs_max'], color='blue', alpha=0.7)
    ax4.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='LRTT w_max=1.0')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([name.split('.')[-2] + '.' + name.split('.')[-1]
                         for name in top30_bert['layer_name']], fontsize=6)
    ax4.set_xlabel('Max |Weight|')
    ax4.set_title('BERT-base Top 30 Layers by |Weight|')
    ax4.legend()
    ax4.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/weight_distribution_comparison.png", dpi=150)
    print(f"\nPlot saved to: {RESULTS_DIR}/weight_distribution_comparison.png")
    plt.close()


def calculate_clipping_impact(mobilebert_df: pd.DataFrame):
    """LRTT w_max=1 클리핑 시 정보 손실 추정"""

    print("\n" + "="*80)
    print("[4] LRTT 클리핑 시 예상 정보 손실")
    print("="*80)
    print("\nLRTT w_max=1.0 적용 시 클리핑될 가중치 비율:")
    print("-"*60)

    # 각 레이어의 |w| > 1 비율
    problem_layers = mobilebert_df[mobilebert_df['pct_gt_1'] > 0].sort_values('pct_gt_1', ascending=False)

    total_params = 0
    clipped_params = 0

    for _, row in problem_layers.head(20).iterrows():
        name_short = row['layer_name'].replace('mobilebert.', '').replace('encoder.', '')
        print(f"{name_short[:50]:<50} {row['pct_gt_1']:>6.2f}%")

    for _, row in mobilebert_df.iterrows():
        total_params += row['num_params']
        clipped_params += row['num_params'] * row['pct_gt_1'] / 100

    print(f"\n{'Total Parameters':<50} {total_params:,}")
    print(f"{'Estimated Clipped Parameters':<50} {int(clipped_params):,}")
    print(f"{'Overall Clipping Rate':<50} {100*clipped_params/total_params:.4f}%")


def main():
    print("="*80)
    print("PHASE 1: MobileBERT LRTT 실패 원인 - 가중치 범위 분석")
    print("="*80)

    # 1. 모델 비교 분석
    mobilebert_df, bert_df = compare_models()

    # 2. 요약 통계
    problem_layers = print_summary(mobilebert_df, bert_df)

    # 3. 히스토그램 시각화
    plot_weight_distribution(mobilebert_df, bert_df)

    # 4. 클리핑 영향 분석
    calculate_clipping_impact(mobilebert_df)

    # 5. 문제 레이어 목록 저장
    problem_layers.to_csv(f"{RESULTS_DIR}/mobilebert_problem_layers.csv", index=False)
    print(f"\nProblem layers saved to: {RESULTS_DIR}/mobilebert_problem_layers.csv")

    print("\n" + "="*80)
    print("Phase 1 완료!")
    print("="*80)


if __name__ == "__main__":
    main()
