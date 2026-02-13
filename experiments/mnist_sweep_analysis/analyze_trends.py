import json
import numpy as np

with open('/data/results/mnist/lrtt_sweep_best_configs.json') as f:
    data = json.load(f)

ranks = [1, 4, 8, 16, 32, 64]
tes = [1, 10, 50, 100, 500, 1000]

# ── Extract matrices ──
hybrid = data['hybrid_sweep']['by_rank_te']
decay = data['decay_sweep']['by_rank_te']

print("=" * 80)
print("HYBRID (Hard Reset) - Best Accuracy by (Rank, TE)")
print("=" * 80)
print(f"{'Rank':>6}", end="")
for te in tes:
    print(f"  TE={te:>4}", end="")
print()
print("-" * 80)

h_mat = {}
for r in ranks:
    h_mat[r] = {}
    print(f"{r:>6}", end="")
    for te in tes:
        entry = hybrid[str(r)][str(te)]
        acc = entry['best_acc'] if isinstance(entry, dict) else entry
        note = entry.get('note', '') if isinstance(entry, dict) else ''
        h_mat[r][te] = acc
        marker = "*" if note else " "
        print(f"  {acc:>6.2f}{marker}", end="")
    print()

print("\n* = rerun with different hyperparameters (see notes)")

print("\n" + "=" * 80)
print("DECAY - Best Accuracy by (Rank, TE)")
print("=" * 80)
print(f"{'Rank':>6}", end="")
for te in tes:
    print(f"  TE={te:>4}", end="")
print()
print("-" * 80)

d_mat = {}
for r in ranks:
    d_mat[r] = {}
    print(f"{r:>6}", end="")
    for te in tes:
        entry = decay[str(r)][str(te)]
        acc = entry['best_acc'] if isinstance(entry, dict) else entry
        d_mat[r][te] = acc
        print(f"  {acc:>6.2f} ", end="")
    print()

# ── Trend Analysis ──
print("\n" + "=" * 80)
print("TREND ANALYSIS: TE 방향 (각 rank에서 TE 증가에 따른 변화)")
print("=" * 80)

for label, mat in [("HYBRID", h_mat), ("DECAY", d_mat)]:
    print(f"\n--- {label} ---")
    for r in ranks:
        accs = [mat[r][te] for te in tes]
        best_te = tes[np.argmax(accs)]
        worst_te = tes[np.argmin(accs)]
        spread = max(accs) - min(accs)

        # Find non-monotonic dips: a TE value worse than BOTH its neighbors
        dips = []
        for j in range(1, len(tes) - 1):
            if accs[j] < accs[j-1] and accs[j] < accs[j+1]:
                drop = min(accs[j-1], accs[j+1]) - accs[j]
                dips.append((tes[j], accs[j], drop))

        print(f"  Rank {r:>2}: best={max(accs):.2f}(TE={best_te}), worst={min(accs):.2f}(TE={worst_te}), spread={spread:.2f}")
        if dips:
            for te_dip, acc_dip, drop in dips:
                severity = "⚠️  SUSPICIOUS" if drop > 2.0 else "   minor"
                print(f"         {severity} dip at TE={te_dip}: {acc_dip:.2f} (neighbors 양쪽보다 {drop:.2f}% 낮음)")

# ── Rank 방향 분석 ──
print("\n" + "=" * 80)
print("TREND ANALYSIS: Rank 방향 (각 TE에서 rank 증가에 따른 변화)")
print("=" * 80)

for label, mat in [("HYBRID", h_mat), ("DECAY", d_mat)]:
    print(f"\n--- {label} ---")
    for te in tes:
        accs = [mat[r][te] for r in ranks]
        best_r = ranks[np.argmax(accs)]
        worst_r = ranks[np.argmin(accs)]
        spread = max(accs) - min(accs)

        dips = []
        for j in range(1, len(ranks) - 1):
            if accs[j] < accs[j-1] and accs[j] < accs[j+1]:
                drop = min(accs[j-1], accs[j+1]) - accs[j]
                dips.append((ranks[j], accs[j], drop))

        print(f"  TE={te:>4}: best={max(accs):.2f}(r={best_r}), worst={min(accs):.2f}(r={worst_r}), spread={spread:.2f}")
        if dips:
            for r_dip, acc_dip, drop in dips:
                severity = "⚠️  SUSPICIOUS" if drop > 2.0 else "   minor"
                print(f"         {severity} dip at r={r_dip}: {acc_dip:.2f} (neighbors 양쪽보다 {drop:.2f}% 낮음)")

# ── Decay vs Hybrid gap ──
print("\n" + "=" * 80)
print("DECAY - HYBRID 차이 (양수 = decay가 더 좋음)")
print("=" * 80)
print(f"{'Rank':>6}", end="")
for te in tes:
    print(f"  TE={te:>4}", end="")
print()
print("-" * 80)
for r in ranks:
    print(f"{r:>6}", end="")
    for te in tes:
        diff = d_mat[r][te] - h_mat[r][te]
        marker = "!!" if abs(diff) > 4 else "  "
        print(f"  {diff:>+5.2f}{marker}", end="")
    print()

# ── Suspect under-optimized combinations ──
print("\n" + "=" * 80)
print("⚠️  최적화 부족 의심 조합 (주변 대비 비정상적으로 낮은 값)")
print("=" * 80)

for label, mat in [("HYBRID", h_mat), ("DECAY", d_mat)]:
    print(f"\n--- {label} ---")
    suspects = []
    for i, r in enumerate(ranks):
        for j, te in enumerate(tes):
            acc = mat[r][te]
            neighbors = []
            if i > 0: neighbors.append(mat[ranks[i-1]][te])
            if i < len(ranks)-1: neighbors.append(mat[ranks[i+1]][te])
            if j > 0: neighbors.append(mat[r][tes[j-1]])
            if j < len(tes)-1: neighbors.append(mat[r][tes[j+1]])

            avg_neighbor = np.mean(neighbors)
            min_neighbor = min(neighbors)
            gap = avg_neighbor - acc

            if gap > 2.0:  # 이웃 평균보다 2% 이상 낮으면
                suspects.append((r, te, acc, avg_neighbor, gap))

    if suspects:
        suspects.sort(key=lambda x: -x[4])
        for r, te, acc, avg_n, gap in suspects:
            # Check if it has a note (rerun)
            entry = (hybrid if label == "HYBRID" else decay)[str(r)][str(te)]
            note = ""
            if isinstance(entry, dict) and 'note' in entry:
                note = f" [rerun: {entry['note']}]"
            print(f"  r={r:>2}, TE={te:>4}: {acc:.2f}% (이웃 평균 {avg_n:.2f}%, gap={gap:.2f}%){note}")
    else:
        print("  의심 조합 없음")

# ── Anomalies from the JSON ──
print("\n" + "=" * 80)
print("JSON에 기록된 anomalies (훈련 실패)")
print("=" * 80)
for a in data['anomalies']:
    print(f"  [{a['sweep']:>6}] {a['name']}: {a['acc']}%")

# ── Summary of rerun status ──
print("\n" + "=" * 80)
print("Rerun v3 결과 요약 (hybrid 실패 조합 재실행)")
print("=" * 80)
for c in data['rerun_v3_results']['configs']:
    print(f"  r={c['rank']:>2}, TE={c['te']:>4}: best={c['best']:.2f}, avg={c['avg']:.2f} → {c['status']}")
