import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 결과 디렉토리
results_dir = "/root/results/sweep_lrtt_loraalpha_200ep/"

# 모든 실험 결과 수집
experiments = []

for exp_dir in sorted(os.listdir(results_dir)):
    exp_path = os.path.join(results_dir, exp_dir)
    if not os.path.isdir(exp_path):
        continue

    trial_params_path = os.path.join(exp_path, "trial_params.json")
    summary_path = os.path.join(exp_path, "summary.csv")

    # trial_params.json 읽기
    if os.path.exists(trial_params_path):
        with open(trial_params_path, 'r') as f:
            params = json.load(f)
    else:
        # trial_params.json이 없으면 실험이 완료되지 않은 것
        params = {
            "trial": None,
            "best_accuracy": None,
            "stopped_epoch": None
        }
        # 디렉토리 이름에서 파라미터 추출
        parts = exp_dir.split('_')
        for part in parts:
            if part.startswith('te'):
                params['transfer_every'] = int(part[2:])
            elif part.startswith('la'):
                params['lora_alpha'] = float(part[2:])
            elif part.startswith('t'):
                params['trial'] = int(part[1:])

    # summary.csv 읽기
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        params['final_epoch'] = len(df) - 1
        params['final_train_loss'] = df['train_loss'].iloc[-1]
        params['final_eval_loss'] = df['eval_loss'].iloc[-1]
        params['final_top1'] = df['eval_top1'].iloc[-1]
        params['final_top5'] = df['eval_top5'].iloc[-1]
        if params.get('best_accuracy') is None:
            params['best_accuracy'] = df['eval_top1'].max()
        params['training_history'] = df

    params['exp_name'] = exp_dir
    experiments.append(params)

# 결과 출력
print("="*80)
print("Sweep Results Summary: LR_TT vs LoRA Alpha (200 epochs)")
print("="*80)
print()

# 완료된 실험들 (trial_params.json 존재)
completed_exps = [e for e in experiments if e.get('stopped_epoch') is not None]
if completed_exps:
    print("Completed Experiments:")
    print("-"*80)
    for exp in completed_exps:
        print(f"\nExperiment: {exp['exp_name']}")
        print(f"  Trial: {exp['trial']}")
        print(f"  Transfer Every: {exp['transfer_every']} epochs")
        print(f"  LoRA Alpha: {exp['lora_alpha']}")
        print(f"  Learning Rate: {exp['lr']:.6f}")
        print(f"  Transfer LR: {exp['transfer_lr']:.6f}")
        print(f"  Best Accuracy: {exp['best_accuracy']:.2f}%")
        print(f"  Stopped Epoch: {exp['stopped_epoch']}")
        if 'final_top1' in exp:
            print(f"  Final Accuracy: {exp['final_top1']:.2f}%")

# 진행 중인 실험들
ongoing_exps = [e for e in experiments if e.get('stopped_epoch') is None]
if ongoing_exps:
    print("\n" + "="*80)
    print("Ongoing Experiments:")
    print("-"*80)
    for exp in ongoing_exps:
        print(f"\nExperiment: {exp['exp_name']}")
        print(f"  Transfer Every: {exp.get('transfer_every', 'N/A')} epochs")
        print(f"  LoRA Alpha: {exp.get('lora_alpha', 'N/A')}")
        if 'final_epoch' in exp:
            print(f"  Current Epoch: {exp['final_epoch']}")
            print(f"  Current Top-1: {exp['final_top1']:.2f}%")
            print(f"  Current Top-5: {exp['final_top5']:.2f}%")
            print(f"  Best so far: {exp['best_accuracy']:.2f}%")

# 통계 분석
if completed_exps:
    print("\n" + "="*80)
    print("Statistical Summary (LoRA Alpha = 0.01):")
    print("-"*80)
    la001_exps = [e for e in completed_exps if e['lora_alpha'] == 0.01]
    if la001_exps:
        best_accs = [e['best_accuracy'] for e in la001_exps]
        print(f"  Number of trials: {len(la001_exps)}")
        print(f"  Best accuracy - Mean: {np.mean(best_accs):.2f}%")
        print(f"  Best accuracy - Std: {np.std(best_accs):.2f}%")
        print(f"  Best accuracy - Min: {np.min(best_accs):.2f}%")
        print(f"  Best accuracy - Max: {np.max(best_accs):.2f}%")

        # 각 trial별 하이퍼파라미터와 정확도
        print("\n  Individual trials:")
        for i, exp in enumerate(sorted(la001_exps, key=lambda x: x['trial'])):
            print(f"    Trial {exp['trial']}: "
                  f"LR={exp['lr']:.4f}, "
                  f"Transfer_LR={exp['transfer_lr']:.4f}, "
                  f"Best={exp['best_accuracy']:.2f}%")

# 시각화
print("\n" + "="*80)
print("Generating visualizations...")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# LoRA Alpha 0.01 실험들의 학습 곡선
ax = axes[0, 0]
for exp in [e for e in experiments if e.get('lora_alpha') == 0.01 and 'training_history' in e]:
    df = exp['training_history']
    ax.plot(df['epoch'], df['eval_top1'],
            label=f"Trial {exp['trial']}, Best={exp['best_accuracy']:.2f}%",
            alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Top-1 Accuracy (%)')
ax.set_title('Training Curves (LoRA Alpha = 0.01)')
ax.legend()
ax.grid(True, alpha=0.3)

# Loss 곡선
ax = axes[0, 1]
for exp in [e for e in experiments if e.get('lora_alpha') == 0.01 and 'training_history' in e]:
    df = exp['training_history']
    ax.plot(df['epoch'], df['eval_loss'],
            label=f"Trial {exp['trial']}",
            alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Evaluation Loss')
ax.set_title('Evaluation Loss Curves (LoRA Alpha = 0.01)')
ax.legend()
ax.grid(True, alpha=0.3)

# Best accuracy 비교
ax = axes[1, 0]
if completed_exps:
    la001_trials = sorted([e for e in completed_exps if e['lora_alpha'] == 0.01],
                          key=lambda x: x['trial'])
    trial_nums = [e['trial'] for e in la001_trials]
    best_accs = [e['best_accuracy'] for e in la001_trials]
    colors = ['green' if acc > 64 else 'orange' if acc > 62 else 'red' for acc in best_accs]
    bars = ax.bar(trial_nums, best_accs, color=colors, alpha=0.7)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Best Top-1 Accuracy (%)')
    ax.set_title('Best Accuracy by Trial (LoRA Alpha = 0.01)')
    ax.set_xticks(trial_nums)
    ax.grid(True, alpha=0.3, axis='y')
    # 값 표시
    for i, (trial, acc) in enumerate(zip(trial_nums, best_accs)):
        ax.text(trial, acc + 0.3, f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)

# 하이퍼파라미터 vs 성능
ax = axes[1, 1]
if la001_exps:
    lrs = [e['lr'] for e in la001_exps]
    transfer_lrs = [e['transfer_lr'] for e in la001_exps]
    best_accs = [e['best_accuracy'] for e in la001_exps]

    # LR vs Best Accuracy
    scatter = ax.scatter(lrs, best_accs, c=transfer_lrs, cmap='viridis',
                        s=100, alpha=0.7, edgecolors='black')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Best Top-1 Accuracy (%)')
    ax.set_title('LR vs Accuracy (color: Transfer LR)')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Transfer LR')

    # Trial 번호 표시
    for exp in la001_exps:
        ax.annotate(f"T{exp['trial']}",
                   (exp['lr'], exp['best_accuracy']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8)

plt.tight_layout()
save_path = '/root/sweep_lrtt_loraalpha_analysis.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {save_path}")

# CSV로 결과 저장
if completed_exps:
    results_df = pd.DataFrame([{
        'exp_name': e['exp_name'],
        'trial': e['trial'],
        'transfer_every': e['transfer_every'],
        'lora_alpha': e['lora_alpha'],
        'lr': e['lr'],
        'transfer_lr': e['transfer_lr'],
        'best_accuracy': e['best_accuracy'],
        'stopped_epoch': e['stopped_epoch']
    } for e in completed_exps])

    csv_path = '/root/sweep_lrtt_loraalpha_summary.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to: {csv_path}")

print("\nAnalysis complete!")
