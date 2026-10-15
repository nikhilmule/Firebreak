"""Multi-agent RL environment around World.

Every decision (3 s of fire time) each drone that can act picks one of 2K+1 actions:
    0..K-1   attack candidate sector k (douse the flames)
    K..2K-1  lay a firebreak ahead of candidate sector k (wet strip across the wind, downwind)
    2K       hold
The team shares one reward:

    r = -(value lost this step) - area_weight * (cells burnt this step)

Observation (numpy, padded to `max_drones` slots):
    drone    (M, DRONE_F)  own state
    cand     (M, K, CAND_F) features of each drone's candidate sectors
    cand_mask(M, K)        which candidates exist
    break_mask(M, K)       which candidates have room for a firebreak downwind
    act_mask (M,)          drone can act this step
    adj      (M, M)        communication links (within comm range, incl. self)
    glob     (GLOB_F,)     shared scenario features
"""
from __future__ import annotations

import math
import numpy as np

from .world import World, Params, N, CELL, MODES, BURN

DRONE_F, CAND_F, GLOB_F = 12, 13, 7
MAP_M = N * CELL


def build_obs(world: World, avail, tg, P, max_drones: int, k: int = 8, comm_range: float = 1500.0):
    M = max_drones
    drone = np.zeros((M, DRONE_F), np.float32)
    cand = np.zeros((M, k, CAND_F), np.float32)
    cand_mask = np.zeros((M, k), bool)
    break_mask = np.zeros((M, k), bool)
    act_mask = np.zeros(M, bool)
    cand_idx = np.full((M, k), -1, np.int64)          # candidate -> index into tg
    p = world.p
    avail_ids = {d.id: i for i, d in enumerate(avail)}
    vscale = max(1.0, float(tg.value.max())) if len(tg) else 1.0
    if len(tg):
        ax = (tg.aim % N + 0.5) * CELL
        ay = (tg.aim // N + 0.5) * CELL
        # what the rest of the fleet could do on each sector (shared over the mesh)
        crowd_p = P.sum(0)                                       # sum of success chances, all drones
        working = np.array([sum(1 for d in world.drones if d.target == b) for b in tg.bin], np.float32)
        gain = P * tg.value[None, :]
        gscale = max(1e-6, float(gain.max())) if gain.size else 1.0
        has_break = np.array([world.break_cell(int(a_)) >= 0 for a_ in tg.aim])

    for d in world.drones[:M]:
        s = d.id
        mode = np.zeros(6, np.float32)
        mode[MODES.index(d.mode)] = 1
        drone[s, :4] = [d.x / MAP_M, d.y / MAP_M, d.water / p.tank, d.batt / p.endurance]
        drone[s, 4:10] = mode
        drone[s, 11] = float(d.task == "break")
        i = avail_ids.get(d.id)
        if i is None or not len(tg):
            continue
        score = P[i] * tg.value
        order = [j for j in np.argsort(-score)[:k] if P[i, j] > 0]
        cur = np.flatnonzero(tg.bin == d.target)
        if len(cur) and P[i, cur[0]] > 0 and cur[0] not in order:     # keep the current job visible
            order = (order[:k - 1] + [int(cur[0])])
        if not order:
            continue
        act_mask[s] = True
        drone[s, 10] = 1
        for c, j in enumerate(order):
            dx, dy = ax[j] - d.x, ay[j] - d.y
            mine = float(tg.bin[j] == d.target)
            cand[s, c] = [tg.value[j] / vscale, P[i, j], tg.n[j] / 64, math.hypot(dx, dy) / MAP_M,
                          dx / MAP_M, dy / MAP_M, mine, 1.0,
                          crowd_p[j] - P[i, j], (working[j] - mine) / 3, gain[i, j] / gscale,
                          float(has_break[j]), math.log10(world.threat.flat[tg.aim[j]] + 1e-3) / 3]
            cand_mask[s, c] = True
            break_mask[s, c] = bool(has_break[j])
            cand_idx[s, c] = j

    xy = np.array([[d.x, d.y] for d in world.drones[:M]] + [[1e9, 1e9]] * (M - len(world.drones[:M])))
    dist = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    adj = (dist <= comm_range).astype(np.float32)
    exists = np.zeros(M, bool)
    exists[:len(world.drones)] = True
    adj *= exists[:, None] & exists[None, :]
    np.fill_diagonal(adj, exists.astype(np.float32))

    burning = int((world.state == BURN).sum())
    glob = np.array([world.t / p.tmax, world.wind / 14, math.cos(world.wind_dir), math.sin(world.wind_dir),
                     len(tg) / 40, world.value_lost / world.total_value, burning / 1000], np.float32)
    obs = dict(drone=drone, cand=cand, cand_mask=cand_mask, break_mask=break_mask, act_mask=act_mask, adj=adj, glob=glob,
               exists=exists)
    return obs, cand_idx


def restrict_to_sectors(obs, cand_idx, avail, a_g, k):
    """'break' mode: greedy has already picked each drone's sector (a_g). Leave the policy only
    the choice for that sector: attack it, lay a firebreak ahead of it, or hold.
    Drones whose sector is not among their candidates, or has no room for a break, just attack."""
    obs = dict(obs)
    cm = np.zeros_like(obs["cand_mask"])
    bm = np.zeros_like(obs["break_mask"])
    am = np.zeros_like(obs["act_mask"])
    slot = np.full(len(obs["act_mask"]), -1)
    for i, d in enumerate(avail):
        j = a_g[i]
        if j < 0:
            continue
        hit = np.flatnonzero(cand_idx[d.id] == j)
        if not len(hit):
            continue
        c = hit[0]
        slot[d.id] = c
        if obs["break_mask"][d.id, c]:
            cm[d.id, c] = bm[d.id, c] = am[d.id] = True
    obs.update(cand_mask=cm, break_mask=bm, act_mask=am)
    return obs, slot


def actions_to_assignment(world, avail, actions, cand_idx, k):
    """Policy actions -> (sector index per drone, task per drone: 0 attack / 1 firebreak)."""
    a = np.full(len(avail), -1)
    tasks = np.zeros(len(avail), int)
    for i, d in enumerate(avail):
        act = int(actions[d.id])
        if act < 2 * k:
            c, task = act % k, act // k
            if cand_idx[d.id, c] >= 0:
                a[i], tasks[i] = cand_idx[d.id, c], task
    return a, tasks


def assignment_to_actions(avail, a, cand_idx, max_drones, k, tasks=None):
    """Map a controller's assignment (and tasks) onto action indices (used for imitation)."""
    acts = np.full(max_drones, 2 * k)
    for i, d in enumerate(avail):
        if a[i] >= 0:
            hit = np.flatnonzero(cand_idx[d.id] == a[i])
            if len(hit):
                acts[d.id] = hit[0] + (k if tasks is not None and tasks[i] == 1 else 0)
    return acts


class FirebreakEnv:
    def __init__(self, drones=6, fires=18, wind=7.0, max_drones=None, k=8, comm_range=1500.0,
                 area_weight=0.002, params: Params | None = None, mode="break"):
        """wind: a number, or a (low, high) tuple sampled uniformly each episode.
        mode: "break" = greedy picks sectors, the policy picks attack / firebreak / hold for each;
              "full"  = the policy picks sector and task itself."""
        self.mode = mode
        self.drones, self.fires, self.wind = drones, fires, wind
        self.M = max_drones or drones
        self.k, self.comm_range, self.area_weight = k, comm_range, area_weight
        self.params = params
        self.world: World | None = None

    def reset(self, seed: int):
        rng = np.random.default_rng(seed + 99991)
        wind = rng.uniform(*self.wind) if isinstance(self.wind, (tuple, list)) else self.wind
        self.world = World(seed, self.drones, self.fires, wind, self.params)
        return self._advance_to_decision(0.0)[0]

    def _decision(self):
        w = self.world
        self._tg = w.build_targets()
        self._avail = w.assignable()
        self._P = w.p_matrix(self._avail, self._tg)
        obs, self._cand_idx = build_obs(w, self._avail, self._tg, self._P, self.M, self.k, self.comm_range)
        if self.mode == "break" and len(self._avail) and len(self._tg):
            from .controllers import _greedy_c
            self._a_g = _greedy_c(self._avail, self._tg, self._P, 0.5)
            obs, self._slot = restrict_to_sectors(obs, self._cand_idx, self._avail, self._a_g, self.k)
        self._obs = obs
        return obs

    def _advance_to_decision(self, reward):
        """Tick until some drone can act (or the episode ends), summing reward."""
        w = self.world
        while True:
            obs = self._decision()
            if obs["act_mask"].any() or w.done:
                return obs, reward
            if self.mode == "break" and len(self._avail) and len(self._tg):
                w.apply_assignment(self._avail, self._tg, self._P, self._a_g)   # nobody to ask: attack
            reward += self._run_ticks()

    def _run_ticks(self):
        w = self.world
        v0, b0 = w.value_lost, w.burnt_cells
        for _ in range(w.p.replan):
            w.tick()
            if w.done:
                break
        return -(w.value_lost - v0) - self.area_weight * (w.burnt_cells - b0)

    def expert_actions(self, controller):
        out = controller(self.world, self._avail, self._tg, self._P, self.world.ctl_rng)
        a, tasks = out if isinstance(out, tuple) else (out, None)
        if self.mode == "break":
            acts = np.full(self.M, 2 * self.k)
            for i, d in enumerate(self._avail):
                c = self._slot[d.id] if len(self._tg) else -1
                if c >= 0 and self._obs["act_mask"][d.id]:
                    acts[d.id] = c + (self.k if tasks is not None and tasks[i] == 1 else 0) if a[i] >= 0 else 2 * self.k
            return acts
        acts = assignment_to_actions(self._avail, a, self._cand_idx, self.M, self.k, tasks)
        # a firebreak with no room downwind is carried out as an attack: label it that way
        k = self.k
        for s_ in range(self.M):
            if k <= acts[s_] < 2 * k and not self._obs["break_mask"][s_, acts[s_] - k]:
                acts[s_] -= k
        return acts

    def _break_mode_assignment(self, actions):
        a = np.array(self._a_g)
        tasks = np.zeros(len(self._avail), int)
        for i, d in enumerate(self._avail):
            if not self._obs["act_mask"][d.id]:
                continue
            act = int(actions[d.id])
            if act == 2 * self.k:
                a[i] = -1                                      # hold
            elif act >= self.k:
                tasks[i] = 1
        return a, tasks

    def step(self, actions):
        w = self.world
        if self.mode == "break" and len(self._avail) and len(self._tg):
            a, tasks = self._break_mode_assignment(actions)
        else:
            a, tasks = actions_to_assignment(w, self._avail, actions, self._cand_idx, self.k)
        w.apply_assignment(self._avail, self._tg, self._P, a, tasks)
        r = self._run_ticks()
        if w.done:
            return self._decision(), r, True, w.summary()
        obs, r = self._advance_to_decision(r)
        return obs, r, w.done, w.summary()


class PolicyController:
    """Wrap a trained policy so it runs like any controller (evaluate.py, wind_sweep.py)."""

    def __init__(self, policy, max_drones, k=8, comm_range=1500.0, greedy=True, stick=0.0, mode="full"):
        """stick: logit bonus for the sector a drone is already working (0 = off).
        mode: must match the mode the policy was trained in ("break" or "full")."""
        self.policy, self.M, self.k, self.comm_range, self.greedy = policy, max_drones, k, comm_range, greedy
        self.stick, self.mode = stick, mode
        self.__name__ = "gnn_policy"

    def __call__(self, world, avail, tg, P, rng):
        import torch
        obs, cand_idx = build_obs(world, avail, tg, P, self.M, self.k, self.comm_range)
        if self.mode == "break":
            from .controllers import _greedy_c
            a_g = _greedy_c(avail, tg, P, 0.5)
            obs, _ = restrict_to_sectors(obs, cand_idx, avail, a_g, self.k)
            with torch.no_grad():
                dist, _ = self.policy.dist_and_value(self.policy.to_tensors([obs]))
                acts = (dist.probs.argmax(-1) if self.greedy else dist.sample())[0].numpy()
            a, tasks = np.array(a_g), np.zeros(len(avail), int)
            for i, d in enumerate(avail):
                if obs["act_mask"][d.id]:
                    if acts[d.id] == 2 * self.k:
                        a[i] = -1
                    elif acts[d.id] >= self.k:
                        tasks[i] = 1
            return a, tasks
        with torch.no_grad():
            dist, _ = self.policy.dist_and_value(self.policy.to_tensors([obs]))
            logits = dist.logits[0].clone()
            if self.stick:
                mine = torch.as_tensor(obs["cand"][:, :, 6] > 0.5)          # candidate = current job
                logits[:, :2 * self.k] += self.stick * torch.cat([mine, mine], -1).float()
            acts = (logits.argmax(-1) if self.greedy
                    else torch.distributions.Categorical(logits=logits).sample()).numpy()
        return actions_to_assignment(world, avail, acts, cand_idx, self.k)
