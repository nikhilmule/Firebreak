"""Fast end-to-end check: python tests/test_smoke.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from firebreak import World, CONTROLLERS, run_episode, objective   # noqa: E402
from firebreak.env import FirebreakEnv                              # noqa: E402


def test_world_runs_and_scores():
    r = run_episode(CONTROLLERS["greedy_c"], seed=3)
    assert 0 <= r["homes_lost"] <= r["homes"]
    assert 0 <= r["value_saved_pct"] <= 100


def test_fleet_beats_no_fleet():
    none = np.mean([run_episode(CONTROLLERS["nearest"], seed=s, drones=0)["homes_lost"] for s in range(3)])
    greedy = np.mean([run_episode(CONTROLLERS["greedy_c"], seed=s)["homes_lost"] for s in range(3)])
    assert greedy < none


def test_objective_diminishing_returns():
    P = np.array([[0.5, 0.2], [0.5, 0.2]])
    v = np.array([1.0, 1.0])
    assert abs(objective([0, 0], P, v) - 0.75) < 1e-9     # two drones on one sector: 1 - 0.5 * 0.5
    assert abs(objective([0, 1], P, v) - 0.70) < 1e-9


def test_env_and_policy():
    from firebreak.policy import GNNPolicy
    env = FirebreakEnv(drones=4, fires=8)
    obs = env.reset(1)
    pol = GNNPolicy(hidden=32)
    for _ in range(20):
        obs, r, done, info = env.step(pol.act_numpy(obs))
        assert r <= 0
        if done:
            break


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
