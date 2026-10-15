"""Compare controllers on the same maps for one scenario.

    python scripts/evaluate.py --seeds 30 --wind 7
    python scripts/evaluate.py --controllers greedy_c_stable gnn_policy --policy runs/gnn/best.pt
"""
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool

import numpy as np

from common import job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+",
                    default=["none", "nearest", "greedy_c", "ga", "pso", "greedy_c_stable", "ga_stable", "pso_stable", "greedy_break"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=2000)
    ap.add_argument("--drones", type=int, default=6)
    ap.add_argument("--fires", type=int, default=18)
    ap.add_argument("--wind", type=float, default=7.0)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--stick", type=float, default=0.0, help="gnn_policy: bonus for keeping its current sector")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--world", choices=["ours", "simfire"], default="ours",
                    help="simfire = same drones and scoring, fire spread from MITRE SimFire's Rothermel model (Python 3.9 env)")
    ap.add_argument("--tmax", type=int, default=3600, help="--world simfire: seconds per episode (its fires are slower)")
    a = ap.parse_args()
    jobs = [(c, a.first_seed + s, a.drones, a.fires, a.wind, a.policy, a.stick, a.world, a.tmax) for c in a.controllers for s in range(a.seeds)]
    with Pool(a.workers) as pool:
        rows = pool.map(job, jobs)
    print(f"{a.drones} drones vs {a.fires} fires, wind {a.wind} m/s, {a.seeds} maps, fire model: {a.world}\n")
    print(f"{'controller':18}{'homes lost':>16}{'value saved':>13}{'ha burnt':>10}{'retarget/min':>14}{'breaks':>8}{'ms/decision':>13}")
    for c in a.controllers:
        sel = [r for r in rows if r["controller"] == c]
        h = np.array([r["homes_lost"] for r in sel])
        print(f"{c:18}{h.mean():9.1f} ± {h.std():4.1f}{np.mean([r['value_saved_pct'] for r in sel]):12.1f}%"
              f"{np.mean([r['area_ha'] for r in sel]):10.0f}{np.mean([r['retargets_per_min'] for r in sel]):14.1f}"
              f"{np.mean([r.get('breaks', 0) for r in sel]):8.1f}"
              f"{np.mean([r['ms_per_decision'] for r in sel]):13.2f}")

    # paired comparison: same maps, so compare map by map against the first controller
    ref = a.controllers[0]
    by = {(r["controller"], r["seed"]): r["homes_lost"] for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    print(f"\npaired vs {ref} (negative = fewer homes lost than {ref}):")
    for c in a.controllers[1:]:
        d = np.array([by[(c, s)] - by[(ref, s)] for s in seeds])
        se = d.std(ddof=1) / np.sqrt(len(d))
        wins, losses = int((d < 0).sum()), int((d > 0).sum())
        verdict = "clearly better" if d.mean() + 2 * se < 0 else "clearly worse" if d.mean() - 2 * se > 0 else "no clear difference"
        print(f"  {c:18} {d.mean():+6.1f} homes  (95% CI {d.mean() - 2 * se:+.1f} to {d.mean() + 2 * se:+.1f}), "
              f"better on {wins}, worse on {losses} of {len(d)} maps -> {verdict}")


if __name__ == "__main__":
    main()
