"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from firebreak import CONTROLLERS, run_episode   # noqa: E402

_POLICY = {}


def get_controller(name, policy_path=None, drones=6, stick=0.0):
    if name == "none":
        return CONTROLLERS["nearest"]            # irrelevant: run with 0 drones
    if name.startswith("gnn_policy"):
        if ":" in name:                       # gnn_policy:2 = stickiness bonus of 2
            stick = float(name.split(":", 1)[1])
        key = (policy_path, drones, stick)
        if key not in _POLICY:
            import torch
            torch.set_num_threads(1)          # many workers: one thread each, or they fight for cores
            from firebreak.policy import GNNPolicy
            from firebreak.env import PolicyController
            ck = torch.load(policy_path, map_location="cpu")
            cfg = ck["config"]
            pol = GNNPolicy(hidden=cfg["hidden"], rounds=cfg["rounds"])
            pol.load_state_dict(ck["state"])
            pol.eval()
            _POLICY[key] = PolicyController(pol, drones, comm_range=cfg["comm_range"], stick=stick,
                                             mode=cfg.get("mode", "full"))
        return _POLICY[key]
    return CONTROLLERS[name]


def job(args):
    """One episode: (controller name, seed, drones, fires, wind, policy_path[, stick, world, tmax]) -> result row."""
    name, seed, drones, fires, wind, policy_path = args[:6]
    stick = args[6] if len(args) > 6 else 0.0
    ctl = get_controller(name, policy_path, drones, stick)
    timing = {"ms": 0.0, "calls": 0}

    def timed(*a):
        t = time.perf_counter()
        out = ctl(*a)
        timing["ms"] += (time.perf_counter() - t) * 1000
        timing["calls"] += 1
        return out

    world = args[7] if len(args) > 7 else "ours"
    if world == "simfire":
        from firebreak import Params
        from firebreak.simfire_world import SimFireWorld
        w = SimFireWorld(seed, 0 if name == "none" else drones, fires, wind, Params(tmax=args[8]))
        while not w.done:
            w.step(timed)
        res = w.summary()
    else:
        res = run_episode(timed, seed=seed, drones=0 if name == "none" else drones, fires=fires, wind=wind)
    res.update(controller=name, seed=seed, wind=wind, drones=drones, fires=fires,
               ms_per_decision=timing["ms"] / max(1, timing["calls"]))
    return res
