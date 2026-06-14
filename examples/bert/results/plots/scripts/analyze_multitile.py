#!/usr/bin/env python3
"""Analyze a multi-tile diagnostic JSON to find which LRTT tile(s) diverge.

Loads diag_no_noise_multitile.json (output of investigate_no_noise_collapse.py),
plots ‖A‖, ‖B‖, ‖A·B‖, ‖C‖ for ALL tracked LRTT tiles (e.g. 48 = 12 layers × qkvo).
Highlights any tile whose ‖weight‖ changes substantially in epoch 5.

Supports both schemas:
  - Legacy: top-level "multi_tiles" dict (pre-unification runs)
  - Modern: top-level "tiles" dict (post-unification runs; use DIAG_TILES="all"
            in the fine_*_lrtt.py script to populate all layers)

Usage:
  python analyze_multitile.py [diag_json_path]
"""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

DEFAULT = Path(__file__).parent / "diag_no_noise_multitile" / "diag_no_noise_multitile.json"

_C_NORM_KEYS = ("norm_C_eff", "norm_C_raw")


def _get(rec, key_or_aliases, default=0):
    """Read first present key from a tuple of aliases (or a single key)."""
    if isinstance(key_or_aliases, str):
        key_or_aliases = (key_or_aliases,)
    for k in key_or_aliases:
        v = rec.get(k)
        if v is not None:
            return v
    return default


def load(path):
    with open(path) as f:
        return json.load(f)


def main(json_path):
    d = load(json_path)
    out_dir = Path(json_path).parent

    # Legacy "multi_tiles" first, else fall back to modern "tiles" (DIAG_TILES="all")
    multi = d.get("multi_tiles") or d.get("tiles", {})
    # Drop back-compat aliases ("first"/"last" would duplicate first_tile/last_tile if "tiles" is used).
    if not d.get("multi_tiles"):
        multi = {k: v for k, v in multi.items() if k not in ("first", "last") or len(multi) <= 2}
    if not multi:
        print(f"WARNING: no multi-tile data in {json_path}")
        print("Legacy runs: ensure MULTI_TILE_DIAG was True.")
        print('Modern runs: set DIAG_TILES = "all" in the fine_*_lrtt.py script to populate all tiles.')
        return

    print(f"Found {len(multi)} multi-tracked tiles:")
    tile_keys = sorted(multi.keys())
    for k in tile_keys:
        print(f"  {k}: {multi[k]['name']}")

    n_tiles = len(tile_keys)
    cmap = cm.get_cmap("turbo", n_tiles)
    colors = {k: cmap(i) for i, k in enumerate(tile_keys)}

    # Fig 1: ‖A‖, ‖B‖, ‖A·B‖, ‖C‖ trajectories — all tiles overlay
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    metrics = [(("norm_A",), "‖A‖"),
               (("norm_B",), "‖B‖"),
               (("norm_AB",), "‖A·B‖"),
               (_C_NORM_KEYS, "‖C‖")]
    for (keys, label), ax in zip(metrics, axes.flat):
        for k in tile_keys:
            steps = [r["step"] for r in multi[k]["steps"]]
            vals = [_get(r, keys) for r in multi[k]["steps"]]
            ax.plot(steps, vals, color=colors[k], linewidth=0.7, alpha=0.7,
                    label=multi[k]["name"].replace("bert.encoder.", "").replace(".analog_module", ""))
        ax.set_ylabel(label)
        ax.set_xlabel("Step")
        ax.set_title(f"{label} per tile ({n_tiles} LRTT tiles)")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    # legend on first axis only, smaller
    axes.flat[0].legend(loc="upper left", fontsize=5, ncol=2, framealpha=0.7)
    fig.suptitle(f"Multi-tile trajectories — best_F1={d.get('best_f1', 0):.2f}, "
                 f"epochs={d.get('config', {}).get('n_epochs', 5)}",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    out1 = out_dir / "multitile_plot1_norms.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out1}")

    # Fig 2: Per-tile change in ‖A‖, ‖B‖, ‖A·B‖, ‖C_raw‖ during epoch 5
    n_epochs = d.get("config", {}).get("n_epochs", 5)
    total_steps = d.get("total_steps", max(r["step"] for r in multi[tile_keys[0]]["steps"]))
    ep5_start = (n_epochs - 1) * (total_steps // n_epochs)

    print(f"\n=== Per-tile change during epoch {n_epochs} (steps {ep5_start} → {total_steps}) ===")
    print(f"{'tile':<48} {'∆‖A‖':>9} {'∆‖B‖':>9} {'∆‖A·B‖':>9} {'∆‖C‖':>11}")

    deltas = {}
    for k in tile_keys:
        steps = multi[k]["steps"]
        # Find closest entries to ep5_start and end
        s_idx = next((i for i, r in enumerate(steps) if r["step"] >= ep5_start), 0)
        e_idx = len(steps) - 1
        s_rec = steps[s_idx]
        e_rec = steps[e_idx]
        d_a = _get(e_rec, "norm_A") - _get(s_rec, "norm_A")
        d_b = _get(e_rec, "norm_B") - _get(s_rec, "norm_B")
        d_ab = _get(e_rec, "norm_AB") - _get(s_rec, "norm_AB")
        d_c = _get(e_rec, _C_NORM_KEYS) - _get(s_rec, _C_NORM_KEYS)
        deltas[k] = (d_a, d_b, d_ab, d_c)
        short_name = multi[k]["name"].replace("bert.encoder.", "").replace(".analog_module", "")
        print(f"{short_name:<48} {d_a:>+9.4f} {d_b:>+9.4f} {d_ab:>+9.4f} {d_c:>+11.4f}")

    # Sort by largest |∆‖A·B‖| change in epoch 5 — these are the suspects
    print(f"\n=== Top-5 tiles by |∆‖A·B‖| during epoch 5 ===")
    by_ab = sorted(deltas.items(), key=lambda kv: abs(kv[1][2]), reverse=True)
    for k, (da, db, dab, dc) in by_ab[:5]:
        short_name = multi[k]["name"].replace("bert.encoder.", "").replace(".analog_module", "")
        print(f"  {short_name:<46} ∆‖A·B‖={dab:+.4f}  ∆‖C‖={dc:+.4f}")

    print(f"\n=== Top-5 tiles by |∆‖C‖| during epoch 5 ===")
    by_c = sorted(deltas.items(), key=lambda kv: abs(kv[1][3]), reverse=True)
    for k, (da, db, dab, dc) in by_c[:5]:
        short_name = multi[k]["name"].replace("bert.encoder.", "").replace(".analog_module", "")
        print(f"  {short_name:<46} ∆‖C‖={dc:+.4f}  ∆‖A·B‖={dab:+.4f}")

    # Fig 3: Highlight outliers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for k in tile_keys:
        steps = [r["step"] for r in multi[k]["steps"]]
        vals_ab = [_get(r, "norm_AB") for r in multi[k]["steps"]]
        vals_c = [_get(r, _C_NORM_KEYS) for r in multi[k]["steps"]]
        # is this tile in top-3 by ∆‖A·B‖?
        is_outlier = k in [kk for kk, _ in by_ab[:3]]
        lw, alpha = (1.5, 1.0) if is_outlier else (0.5, 0.4)
        label = multi[k]["name"].replace("bert.encoder.", "").replace(".analog_module", "") if is_outlier else None
        axes[0].plot(steps, vals_ab, color=colors[k], linewidth=lw, alpha=alpha, label=label)
        axes[1].plot(steps, vals_c, color=colors[k], linewidth=lw, alpha=alpha, label=label)
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("‖A·B‖")
    axes[0].set_title("‖A·B‖ — top-3 movers highlighted")
    axes[0].axvline(ep5_start, color="gray", linestyle="--", linewidth=0.8, label=f"epoch {n_epochs} start")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    axes[1].set_xlabel("Step"); axes[1].set_ylabel("‖C‖")
    axes[1].set_title("‖C‖ — top-3 movers highlighted")
    axes[1].axvline(ep5_start, color="gray", linestyle="--", linewidth=0.8, label=f"epoch {n_epochs} start")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    fig.suptitle("Outlier identification — which tile(s) move most during epoch 5", fontsize=12)
    fig.tight_layout()
    out2 = out_dir / "multitile_plot2_outliers.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out2}")

    # Print epoch_history
    eh = d.get("epoch_history", [])
    print(f"\n=== Epoch history ===")
    for e in eh:
        print(f"  ep {e.get('epoch')}: f1={e.get('f1', 0):.2f}, em={e.get('em', 0):.2f}, "
              f"train_loss={e.get('train_loss', 0):.4f}, lr={e.get('lr', 0):.4g}")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    main(p)
