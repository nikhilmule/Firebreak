"""Record replays for the 3D viewer: every controller on the SAME map, one JSON file.

    python record.py --controllers greedy_break gnn_policy --policy ../runs/gnn3/best.pt \
        --seed 4242 --wind 8 --out ../replays/showcase.json

Open the Firebreak 3D Replay page and use "Load replay" to pick the file.
Tip: run evaluate.py first and pick a seed where the controllers differ, so the replay shows it.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import get_controller
from firebreak import World
from firebreak.world import MODES, WET, N, CELL


def record(name, policy, seed, drones, fires, wind, every):
    ctl = get_controller(name, policy, drones)
    w = World(seed, 0 if name == "none" else drones, fires, wind)
    prev = np.zeros(N * N, np.int8)
    frames = []

    def snap():
        st = w.state.ravel().astype(np.int8).copy()
        # mark long-lasting firebreak wetting as state 4 so the viewer can draw it differently
        st[(st == WET) & (w.timer.ravel() > w.p.wet_time)] = 4
        diff = np.flatnonzero(st != prev)
        ch = np.empty(2 * len(diff), np.int64)
        ch[0::2], ch[1::2] = diff, st[diff]
        prev[:] = st
        dr = []
        for d in w.drones:
            ax = ay = -1
            if d.mode == "assigned" and d.aim >= 0:
                ax, ay = int((d.aim % N + 0.5) * CELL), int((d.aim // N + 0.5) * CELL)
            dr += [int(d.x), int(d.y), MODES.index(d.mode), d.water, int(d.task == "break"), ax, ay]
        s = w.summary()
        frames.append(dict(t=w.t, wd=round(w.wind_dir, 3), ch=ch.tolist(), d=dr,
                           s=[s["homes_lost"], round(s["value_saved_pct"], 1), round(s["area_ha"])]))

    snap()
    while not w.done:
        w.step(ctl)
        if w.t % every == 0 or w.done:
            snap()
    s = w.summary()
    return w, dict(controller=name, frames=frames, events=[list(e[:3]) + [int(e[3] == "break")] for e in w.events],
                   summary={k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", default=["greedy_c_stable", "greedy_break"])
    ap.add_argument("--policy", default=None)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--drones", type=int, default=6)
    ap.add_argument("--fires", type=int, default=18)
    ap.add_argument("--wind", type=float, default=8.0)
    ap.add_argument("--every", type=int, default=3, help="seconds of fire time per frame")
    ap.add_argument("--out", default="../replays/replay.json")
    a = ap.parse_args()

    runs, world = [], None
    for c in a.controllers:
        w, run = record(c, a.policy, a.seed, a.drones, a.fires, a.wind, a.every)
        world = world or w
        runs.append(run)
        print(f"{c:18} homes lost {run['summary']['homes_lost']:4d}  value saved {run['summary']['value_saved_pct']:5.1f}%  "
              f"firebreaks {run['summary']['breaks']:3d}  frames {len(run['frames'])}")
    meta = dict(version=1, N=N, cell=CELL, seed=a.seed, wind=a.wind, drones=a.drones, fires=a.fires,
                every=a.every, fuel="".join(str(int(v)) for v in world.fuel.ravel()),
                lake=world.lake, town=world.town, base=list(world.base), homes=world.homes,
                tank=world.p.tank)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(meta=meta, runs=runs), f, separators=(",", ":"))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
