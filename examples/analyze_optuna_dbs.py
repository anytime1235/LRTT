#!/usr/bin/env python3
"""
Analyze three Optuna SQLite databases:
  - Extract study names, trial counts, search space definitions
  - Identify top trials by accuracy
  - Compute parameter statistics across top-N trials
"""

import sqlite3
import json
import statistics
from collections import defaultdict, Counter

DBS = [
    (
        "LRTT",
        "/data/LRTT_vit/examples/results/optuna_vitsptlsa_lrtt/optuna_vitsptlsa_lrtt_main.db",
    ),
    (
        "TTV1",
        "/data/LRTT_vit/examples/results/optuna_vitsptlsa_ttv1/optuna_vitsptlsa_ttv1_main.db",
    ),
    (
        "TTV2",
        "/data/LRTT_vit/examples/results/optuna_vitsptlsa_ttv2/optuna_vitsptlsa_ttv2_main.db",
    ),
]

SEPARATOR = "=" * 100
SUBSEP = "-" * 80


def classify_distribution(dist_json_str):
    """Parse the distribution JSON and return a human-readable description."""
    d = json.loads(dist_json_str)
    name = d.get("name", "unknown")

    if name == "CategoricalDistribution":
        choices = d.get("attributes", {}).get("choices", [])
        return "Categorical", f"choices={choices}"
    elif name == "FloatDistribution":
        attrs = d.get("attributes", {})
        low = attrs.get("low")
        high = attrs.get("high")
        log = attrs.get("log", False)
        step = attrs.get("step")
        extra = ""
        if log:
            extra += ", log=True"
        if step is not None:
            extra += f", step={step}"
        return "Float", f"[{low}, {high}]{extra}"
    elif name == "IntDistribution":
        attrs = d.get("attributes", {})
        low = attrs.get("low")
        high = attrs.get("high")
        log = attrs.get("log", False)
        step = attrs.get("step", 1)
        extra = ""
        if log:
            extra += ", log=True"
        if step != 1:
            extra += f", step={step}"
        return "Int", f"[{low}, {high}]{extra}"
    elif name == "UniformDistribution":
        attrs = d.get("attributes", {})
        low = attrs.get("low")
        high = attrs.get("high")
        return "Float (Uniform)", f"[{low}, {high}]"
    elif name == "LogUniformDistribution":
        attrs = d.get("attributes", {})
        low = attrs.get("low")
        high = attrs.get("high")
        return "Float (LogUniform)", f"[{low}, {high}]"
    elif name == "IntUniformDistribution":
        attrs = d.get("attributes", {})
        low = attrs.get("low")
        high = attrs.get("high")
        return "Int (Uniform)", f"[{low}, {high}]"
    elif name == "DiscreteUniformDistribution":
        attrs = d.get("attributes", {})
        low = attrs.get("low")
        high = attrs.get("high")
        q = attrs.get("q")
        return "Float (Discrete)", f"[{low}, {high}], q={q}"
    else:
        return name, str(d.get("attributes", {}))


def decode_param_value(param_name, param_value_float, dist_json_str):
    """
    Optuna stores categorical params as integer indices in param_value.
    Decode them back to the actual categorical value.
    For numeric params, return the float/int directly.
    """
    d = json.loads(dist_json_str)
    name = d.get("name", "unknown")

    if name == "CategoricalDistribution":
        choices = d.get("attributes", {}).get("choices", [])
        idx = int(param_value_float)
        if 0 <= idx < len(choices):
            return choices[idx]
        return param_value_float
    elif "Int" in name:
        return int(param_value_float)
    else:
        return param_value_float


def compute_stats(values):
    """Compute min, max, mean, median for a list of numeric values."""
    numeric = [v for v in values if isinstance(v, (int, float))]
    if not numeric:
        # All categorical -- show frequency instead
        counts = Counter(values)
        return {"mode": counts.most_common(1)[0][0], "distribution": dict(counts)}
    return {
        "min": min(numeric),
        "max": max(numeric),
        "mean": statistics.mean(numeric),
        "median": statistics.median(numeric),
    }


def analyze_db(label, db_path):
    print(f"\n{SEPARATOR}")
    print(f"  DATABASE: {label}")
    print(f"  Path: {db_path}")
    print(SEPARATOR)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()

    # -- (a) Study names --
    cur.execute("SELECT study_id, study_name FROM studies")
    studies = cur.fetchall()
    print(f"\n[A] Studies in database:")
    for sid, sname in studies:
        cur.execute(
            "SELECT direction FROM study_directions WHERE study_id=? ORDER BY objective",
            (sid,),
        )
        directions = [r[0] for r in cur.fetchall()]
        print(f"     study_id={sid}  name='{sname}'  directions={directions}")

    for study_id, study_name in studies:
        print(f"\n{SUBSEP}")
        print(f"  Study: '{study_name}' (id={study_id})")
        print(SUBSEP)

        # -- (b) Count COMPLETE trials --
        cur.execute(
            "SELECT state, COUNT(*) FROM trials WHERE study_id=? GROUP BY state",
            (study_id,),
        )
        state_counts = cur.fetchall()
        total_complete = 0
        print(f"\n[B] Trial counts by state:")
        for state, cnt in state_counts:
            print(f"     {state}: {cnt}")
            if state == "COMPLETE":
                total_complete = cnt
        print(f"     --> COMPLETE trials: {total_complete}")

        # -- (c) Search space: all parameter names + distributions --
        cur.execute(
            """
            SELECT DISTINCT tp.param_name, tp.distribution_json
            FROM trial_params tp
            JOIN trials t ON tp.trial_id = t.trial_id
            WHERE t.study_id = ?
            GROUP BY tp.param_name
            """,
            (study_id,),
        )
        param_defs = cur.fetchall()

        print(f"\n[C] Search space ({len(param_defs)} parameters):")
        print(f"     {'Parameter':<35} {'Type':<22} {'Range / Choices'}")
        print(f"     {'-'*35} {'-'*22} {'-'*50}")

        # Store distribution info for later decoding
        param_dist_map = {}
        for pname, djson in param_defs:
            param_dist_map[pname] = djson
            dtype, desc = classify_distribution(djson)
            print(f"     {pname:<35} {dtype:<22} {desc}")

        # -- (d) Top 10 trials by accuracy (maximize) --
        cur.execute(
            """
            SELECT t.trial_id, t.number, tv.value
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id AND tv.objective = 0
            WHERE t.study_id = ? AND t.state = 'COMPLETE'
            ORDER BY tv.value DESC
            LIMIT 10
            """,
            (study_id,),
        )
        top10_rows = cur.fetchall()

        print(f"\n[D] Top 10 trials by accuracy (descending):")
        print(f"     {'Rank':<6} {'Trial#':<10} {'TrialID':<10} {'Accuracy'}")
        print(f"     {'-'*6} {'-'*10} {'-'*10} {'-'*15}")
        top10_ids = []
        for rank, (tid, tnum, val) in enumerate(top10_rows, 1):
            print(f"     {rank:<6} {tnum:<10} {tid:<10} {val}")
            top10_ids.append(tid)

        # Collect parameters for top-N trials
        def get_params_for_trials(trial_ids):
            if not trial_ids:
                return {}
            placeholders = ",".join("?" * len(trial_ids))
            cur.execute(
                f"""
                SELECT tp.trial_id, tp.param_name, tp.param_value, tp.distribution_json
                FROM trial_params tp
                WHERE tp.trial_id IN ({placeholders})
                """,
                trial_ids,
            )
            rows = cur.fetchall()
            params = defaultdict(dict)
            for tid, pname, pval, djson in rows:
                decoded = decode_param_value(pname, pval, djson)
                params[tid][pname] = decoded
            return params

        top10_params = get_params_for_trials(top10_ids)
        top5_ids = top10_ids[:5]
        top5_params = {tid: top10_params[tid] for tid in top5_ids if tid in top10_params}

        all_param_names = sorted(param_dist_map.keys())

        # -- (e) Stats across top 10 --
        print(f"\n[E] Parameter statistics across TOP 10 trials:")
        print(f"     {'Parameter':<35} {'min':<16} {'max':<16} {'mean':<16} {'median':<16}")
        print(f"     {'-'*35} {'-'*16} {'-'*16} {'-'*16} {'-'*16}")
        for pname in all_param_names:
            values = [top10_params[tid].get(pname) for tid in top10_ids if tid in top10_params]
            values = [v for v in values if v is not None]
            if not values:
                continue
            stats = compute_stats(values)

            if "distribution" in stats:
                print(f"     {pname:<35} [Categorical]  mode={stats['mode']}  freq={stats['distribution']}")
            else:
                print(
                    f"     {pname:<35} {stats['min']:<16.6g} {stats['max']:<16.6g} "
                    f"{stats['mean']:<16.6g} {stats['median']:<16.6g}"
                )

        # -- (f) Stats across top 5 --
        print(f"\n[F] Parameter statistics across TOP 5 trials:")
        print(f"     {'Parameter':<35} {'min':<16} {'max':<16} {'mean':<16} {'median':<16}")
        print(f"     {'-'*35} {'-'*16} {'-'*16} {'-'*16} {'-'*16}")
        for pname in all_param_names:
            values = [top5_params[tid].get(pname) for tid in top5_ids if tid in top5_params]
            values = [v for v in values if v is not None]
            if not values:
                continue
            stats = compute_stats(values)

            if "distribution" in stats:
                print(f"     {pname:<35} [Categorical]  mode={stats['mode']}  freq={stats['distribution']}")
            else:
                print(
                    f"     {pname:<35} {stats['min']:<16.6g} {stats['max']:<16.6g} "
                    f"{stats['mean']:<16.6g} {stats['median']:<16.6g}"
                )

        # -- (g) Actual parameter values for top 5 trials --
        print(f"\n[G] Actual parameter values for TOP 5 trials:")
        header = f"     {'Parameter':<35}"
        for rank, tid in enumerate(top5_ids, 1):
            tnum = [r[1] for r in top10_rows if r[0] == tid][0]
            tval = [r[2] for r in top10_rows if r[0] == tid][0]
            header += f" | #{tnum} (acc={tval:.4f})"
        print(header)
        print(f"     {'-'*35}" + (" |" + "-" * 24) * len(top5_ids))

        for pname in all_param_names:
            row = f"     {pname:<35}"
            for tid in top5_ids:
                val = top5_params.get(tid, {}).get(pname, "N/A")
                if isinstance(val, float):
                    row += f" | {val:<22.6g}"
                else:
                    row += f" | {str(val):<22}"
            print(row)

    conn.close()


def main():
    print("=" * 100)
    print("  OPTUNA DATABASE ANALYSIS REPORT")
    print("  Databases: LRTT, TTV1, TTV2")
    print("=" * 100)

    for label, path in DBS:
        try:
            analyze_db(label, path)
        except Exception as e:
            print(f"\n*** ERROR analyzing {label}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 100}")
    print("  END OF REPORT")
    print("=" * 100)


if __name__ == "__main__":
    main()
