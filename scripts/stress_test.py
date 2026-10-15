"""Stress test: does the recommended controller stay on top when conditions get worse?

Starts from the baseline (6 drones, 18 fires, wind 8 m/s) and changes ONE thing at a time:
fleet size, number of fires, wind, sensor noise, missed fires, drone failures, sensing range,
and a sudden wind shift. Every controller runs on the same maps in every condition.

    python stress_test.py --seeds 30 --out ../results/stress
    python stress_test.py --seeds 30 --controllers greedy_break gnn_policy --policy ../runs/gnn4/best.pt

Writes episodes.csv, summary.csv and summary.md (one table per stress axis, with a map-by-map
comparison against the reference controller, the first one listed).
"""
from __future__ import annotations

import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np

from common import get_controller
from firebreak import World, Params

BASE = dict(drones=6, fires=18, wind=8.0)
AXES = [
    ("Fleet size", "drones", [2, 4, 6, 9, 12], "drones"),
    ("Number of fires", "fires", [6, 12, 18, 30, 40], "fires"),
    ("Wind speed", "wind", [2.0, 5.0, 8.0, 11.0, 14.0], "m/s"),
    ("Sensor noise", "obs_noise", [0.0, 0.25, 0.5, 1.0], "sd of error"),
    ("Missed fires", "miss_rate", [0.0, 0.15, 0.3, 0.5], "share unseen"),
    ("Drone failures", "fail_per_hour", [0.0, 2.0, 4.0, 8.0], "per drone-hour"),
    ("Sensing range", "sense_range", [0.0, 2000.0, 1000.0, 600.0], "m (0 = whole map)"),
    ("Wind shift at 10 min", "wind_shift_deg", [0.0, 45.0, 90.0, 150.0], "degrees"),
]
DEFAULT = ["greedy_break", "ga_break", "pso_break", "nearest_break", "greedy_c_stable", "none"]


def condition_key(knob, val):
    c = dict(BASE)
    params = {}
    if knob in ("drones", "fires", "wind"):
        c[knob] = val
    elif val:
        params[knob] = val          # 0 = setting off = the baseline itself
    return c, params


def run(args):
    name, seed, cond, params, policy = args
    ctl = get_controller(name, policy, cond["drones"])
    w = World(seed, 0 if name == "none" else cond["drones"], cond["fires"], cond["wind"], Params(**params))
    while not w.done:
        w.step(ctl)
    s = w.summary()
    s.update(controller=name, seed=seed, **cond, **{k: v for k, v in params.items()})
    s["cond"] = repr((sorted(cond.items()), sorted(params.items())))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", default=DEFAULT)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--first-seed", type=int, default=5000)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="../results/stress")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    conds = {}
    for title, knob, vals, unit in AXES:
        for v in vals:
            c, p = condition_key(knob, v)
            conds[repr((sorted(c.items()), sorted(p.items())))] = (c, p)
    jobs = [(n, a.first_seed + s, c, p, a.policy) for (c, p) in conds.values() for n in a.controllers for s in range(a.seeds)]
    print(f"{len(conds)} conditions x {len(a.controllers)} controllers x {a.seeds} maps = {len(jobs)} runs on {a.workers} workers")
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs, chunksize=2), 1):
            rows.append(r)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    keys = ["cond", "controller", "seed", "drones", "fires", "wind", "obs_noise", "miss_rate", "fail_per_hour",
            "sense_range", "wind_shift_deg", "homes_lost", "homes", "value_saved_pct", "area_ha", "breaks", "failures"]
    with open(os.path.join(a.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ref = a.controllers[0]
    by = {}
    for r in rows:
        by.setdefault((r["cond"], r["controller"]), {})[r["seed"]] = r["homes_lost"]
    summary, md = [], [f"# Stress test\n\nBaseline: {BASE['drones']} drones, {BASE['fires']} fires, wind {BASE['wind']} m/s. "
                       f"{a.seeds} maps per condition. Paired difference = homes lost minus {ref}, same maps; "
                       f"negative is better. 95% range = mean ± 2 standard errors.\n"]
    for title, knob, vals, unit in AXES:
        md.append(f"\n## {title} ({unit})\n")
        md.append("| " + knob + " | " + " | ".join(a.controllers) + " |")
        md.append("|" + "---|" * (len(a.controllers) + 1))
        for v in vals:
            c, p = condition_key(knob, v)
            ck = repr((sorted(c.items()), sorted(p.items())))
            refm = by[(ck, ref)]
            cells = []
            for n in a.controllers:
                h = by[(ck, n)]
                arr = np.array([h[s] for s in sorted(h)])
                d = np.array([h[s] - refm[s] for s in sorted(h)])
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
                verdict = "ref" if n == ref else ("better" if d.mean() + 2 * se < 0 else "worse" if d.mean() - 2 * se > 0 else "tie")
                summary.append(dict(axis=title, knob=knob, value=v, controller=n, homes_lost=round(arr.mean(), 2),
                                    sd=round(arr.std(), 2), diff_vs_ref=round(d.mean(), 2), ci_lo=round(d.mean() - 2 * se, 2),
                                    ci_hi=round(d.mean() + 2 * se, 2), verdict=verdict, n=len(arr)))
                cells.append(f"{arr.mean():.1f}" + ("" if n == ref else f" ({d.mean():+.1f}, {verdict})"))
            md.append(f"| {v:g} | " + " | ".join(cells) + " |")
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(md) + "\n")
    beaten = [s for s in summary if s["verdict"] == "better"]
    print("\n".join(md))
    print(f"\n{ref} was clearly beaten in {len(beaten)} of {len(conds)} conditions"
          + (": " + ", ".join(f"{s['controller']} at {s['knob']}={s['value']:g}" for s in beaten) if beaten else "."))


if __name__ == "__main__":
    main()
