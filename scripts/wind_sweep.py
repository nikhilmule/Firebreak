"""Wind-speed experiment: every controller on the same maps at several wind speeds.

    python scripts/wind_sweep.py --winds 0 2 4 6 8 10 12 14 --seeds 20 --workers 8 --out results/wind
    python scripts/wind_sweep.py ... --controllers greedy_c_stable gnn_policy --policy runs/gnn/best.pt

Writes episodes.csv (one row per episode), summary.csv (mean and sd per wind x controller)
and wind_sweep.png.
"""
from __future__ import annotations

import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np

from common import job

DEFAULT = ["none", "nearest", "greedy_c", "ga", "pso", "greedy_c_stable", "ga_stable", "pso_stable", "greedy_break"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winds", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 14])
    ap.add_argument("--controllers", nargs="+", default=DEFAULT)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--drones", type=int, default=6)
    ap.add_argument("--fires", type=int, default=18)
    ap.add_argument("--policy", default=None, help="checkpoint for gnn_policy")
    ap.add_argument("--stick", type=float, default=0.0, help="gnn_policy: bonus for keeping its current sector")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/wind")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    jobs = [(c, 1000 + s, a.drones, a.fires, w, a.policy, a.stick)
            for w in a.winds for s in range(a.seeds) for c in a.controllers]
    print(f"{len(jobs)} episodes on {a.workers} workers")
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2), 1):
            rows.append(r)
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    keys = ["wind", "controller", "seed", "homes_lost", "homes", "value_saved_pct", "area_ha",
            "retargets_per_min", "ms_per_decision", "t"]
    with open(os.path.join(a.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["wind"], r["controller"], r["seed"])))

    summary = []
    for wnd in a.winds:
        for c in a.controllers:
            sel = [r for r in rows if r["wind"] == wnd and r["controller"] == c]
            h = np.array([r["homes_lost"] for r in sel])
            summary.append(dict(wind=wnd, controller=c, homes_lost_mean=h.mean(), homes_lost_sd=h.std(),
                                value_saved_mean=np.mean([r["value_saved_pct"] for r in sel]),
                                area_ha_mean=np.mean([r["area_ha"] for r in sel]),
                                retargets_per_min=np.mean([r["retargets_per_min"] for r in sel]),
                                ms_per_decision=np.mean([r["ms_per_decision"] for r in sel]), n=len(sel)))
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0].keys()))
        w.writeheader()
        for s in summary:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in a.controllers:
            ss = [s for s in summary if s["controller"] == c]
            x = [s["wind"] for s in ss]
            m = np.array([s["homes_lost_mean"] for s in ss])
            sd = np.array([s["homes_lost_sd"] for s in ss]) / np.sqrt(a.seeds)
            ax.plot(x, m, marker="o", label=c, color="grey" if c == "none" else None,
                    linestyle="--" if c == "none" else "-")
            ax.fill_between(x, m - sd, m + sd, alpha=0.15)
        ax.set_xlabel("wind speed (m/s)")
        ax.set_ylabel(f"homes lost (mean of {a.seeds} maps, ±1 s.e.)")
        ax.set_title(f"{a.drones} drones vs {a.fires} spot fires")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "wind_sweep.png"), dpi=150)
    except ImportError:
        print("matplotlib not installed: skipped the plot")

    print("\nhomes lost (mean) by wind:")
    print("controller".ljust(18) + "".join(f"{w:>7g}" for w in a.winds))
    for c in a.controllers:
        print(c.ljust(18) + "".join(f"{s['homes_lost_mean']:7.1f}" for s in summary if s["controller"] == c))


if __name__ == "__main__":
    main()
