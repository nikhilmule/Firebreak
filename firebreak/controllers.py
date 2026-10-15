"""Assignment controllers.

Every controller has the signature

    controller(world, avail, targets, P, rng) -> np.ndarray of length len(avail)

where entry i is the index of the sector (into `targets`) for drone avail[i], or -1 to hold.
P[i, j] is the chance drone i puts out sector j (World.p_matrix).

The smart controllers maximise objective C:
    J = sum_j v_j * (1 - prod_i (1 - p_ij))
"""
from __future__ import annotations

import numpy as np
from .world import CELL, N


def objective(a, P, v):
    """Objective C for one assignment vector a (len n, -1 = hold)."""
    return objective_batch(np.asarray(a)[None, :], P, v)[0]


def objective_batch(G, P, v):
    """Objective C for a population of assignments G (pop x n). Vectorised."""
    pop, n = G.shape
    valid = G >= 0
    Gc = np.where(valid, G, 0)
    p = np.where(valid, P[np.arange(n)[None, :], Gc], 0.0)
    logmiss = np.zeros((pop, P.shape[1]))
    np.add.at(logmiss, (np.repeat(np.arange(pop), n), Gc.ravel()), np.log1p(-p).ravel())
    return ((1 - np.exp(logmiss)) * v[None, :]).sum(1)


def _switch_count(G, current, tg_bins):
    """How many drones would change to a different sector than they are working now."""
    cur_idx = np.array([np.flatnonzero(tg_bins == c)[0] if c >= 0 and (tg_bins == c).any() else -2
                        for c in current])
    has = cur_idx >= 0
    return ((G != cur_idx[None, :]) & has[None, :]).sum(1)


def candidates(P, v, k=10):
    """Each drone's k most promising sectors by v_j * p_ij."""
    score = P * v[None, :]
    out = []
    for row in score:
        idx = np.argsort(-row)[:k]
        out.append(idx[row[idx] > 0])
    return out


# ------------------------------------------------------------------ baselines
def nearest(world, avail, tg, P, rng):
    ax = (tg.aim % N + 0.5) * CELL
    ay = (tg.aim // N + 0.5) * CELL
    a = np.full(len(avail), -1)
    for i, d in enumerate(avail):
        dist = np.hypot(ax - d.x, ay - d.y)
        dist[P[i] <= 0] = np.inf
        if np.isfinite(dist).any():
            a[i] = int(dist.argmin())
    return a


def make_greedy_c(switch_cost=0.0):
    def g(world, avail, tg, P, rng):
        return _greedy_c(avail, tg, P, switch_cost)
    g.__name__ = "greedy_c" if not switch_cost else "greedy_c_stable"
    return g


def _greedy_c(avail, tg, P, switch_cost):
    """Repeatedly give one drone the sector with the largest marginal gain in J.
    J is monotone submodular, so this greedy is within (1 - 1/e) of the optimum."""
    n, T = P.shape
    a = np.full(n, -1)
    miss = np.ones(T)
    free = np.ones(n, bool)
    cost = np.zeros((n, T))
    if switch_cost:
        for i, d in enumerate(avail):
            if d.target >= 0 and (tg.bin == d.target).any():
                cost[i, tg.bin != d.target] = switch_cost * tg.value.mean()
    for _ in range(n):
        gain = (tg.value * miss)[None, :] * P - cost
        gain[~free] = -1
        i, j = np.unravel_index(int(gain.argmax()), gain.shape)
        if gain[i, j] <= 1e-9:
            break
        a[i], free[i] = j, False
        miss[j] *= 1 - P[i, j]
    return a


# ------------------------------------------------------- genetic algorithm
def make_ga(pop=48, gens=30, mut=None, elite=2, switch_cost=0.0):
    """switch_cost > 0 subtracts switch_cost * mean(v) per drone that changes sector (plan stability)."""
    def ga(world, avail, tg, P, rng):
        n = len(avail)
        cand = candidates(P, tg.value)
        sizes = np.array([max(1, len(c)) for c in cand])
        table = np.full((n, sizes.max()), -1)
        for i, c in enumerate(cand):
            table[i, :len(c)] = c if len(c) else -1
        current = [d.target for d in avail]
        penalty = switch_cost * tg.value.mean()

        def genes(u):  # u in [0,1) -> candidate index -> sector index
            k = np.minimum((u * sizes[None, :]).astype(int), sizes[None, :] - 1)
            return table[np.arange(n)[None, :], k]

        def fit(G):
            f = objective_batch(G, P, tg.value)
            if penalty:
                f = f - penalty * _switch_count(G, current, tg.bin)
            return f

        G = genes(rng.random((pop, n)))
        F = fit(G)
        m = mut if mut is not None else 1.0 / n
        for _ in range(gens):
            order = np.argsort(-F)
            G, F = G[order], F[order]
            # tournament selection (size 3)
            def pick(k):
                idx = rng.integers(0, pop, (k, 3))
                return idx[np.arange(k), F[idx].argmax(1)]
            k = pop - elite
            p1, p2 = G[pick(k)], G[pick(k)]
            child = np.where(rng.random((k, n)) < 0.5, p1, p2)       # uniform crossover
            mutate = rng.random((k, n)) < m
            child = np.where(mutate, genes(rng.random((k, n))), child)
            G = np.vstack([G[:elite], child])
            F = np.concatenate([F[:elite], fit(child)])
        return G[int(F.argmax())]
    ga.__name__ = "ga" if not switch_cost else "ga_stable"
    return ga


# ------------------------------------------------------ particle swarm
def make_pso(particles=32, iters=40, switch_cost=0.0):
    """Random-key PSO: each drone's coordinate in [0, K) is floored to pick one of its candidates."""
    def pso(world, avail, tg, P, rng):
        n = len(avail)
        cand = candidates(P, tg.value)
        sizes = np.array([max(1, len(c)) for c in cand], dtype=float)
        table = np.full((n, int(sizes.max())), -1)
        for i, c in enumerate(cand):
            table[i, :len(c)] = c if len(c) else -1
        current = [d.target for d in avail]
        penalty = switch_cost * tg.value.mean()

        def decode(X):
            k = np.clip(X.astype(int), 0, sizes.astype(int)[None, :] - 1)
            return table[np.arange(n)[None, :], k]

        def fit(X):
            G = decode(X)
            f = objective_batch(G, P, tg.value)
            if penalty:
                f = f - penalty * _switch_count(G, current, tg.bin)
            return f

        X = rng.random((particles, n)) * sizes[None, :]
        V = (rng.random((particles, n)) - 0.5) * sizes[None, :] * 0.5
        F = fit(X)
        PB, PF = X.copy(), F.copy()
        g = int(F.argmax())
        GB, GF = X[g].copy(), F[g]
        for it in range(iters):
            w = 0.9 - 0.5 * it / iters
            r1, r2 = rng.random((2, particles, n))
            V = w * V + 1.5 * r1 * (PB - X) + 1.5 * r2 * (GB[None, :] - X)
            X = np.clip(X + V, 0, sizes[None, :] - 1e-6)
            F = fit(X)
            better = F > PF
            PB[better], PF[better] = X[better], F[better]
            g = int(PF.argmax())
            if PF[g] > GF:
                GB, GF = PB[g].copy(), PF[g]
        return decode(GB[None, :])[0]
    pso.__name__ = "pso" if not switch_cost else "pso_stable"
    return pso


def make_greedy_break(switch_cost=0.5, p_max=1.01, reach=20):
    """Hand-made firebreak rule on top of greedy. For each drone greedy assigns, if its water is
    unlikely to put the sector out (p < p_max; the default 1.01 means always) and the sector is
    close enough to threaten something valuable (threat above e^(-reach/14)), lay a firebreak
    ahead of it instead of attacking the flames."""
    def gb(world, avail, tg, P, rng):
        a = _greedy_c(avail, tg, P, switch_cost)
        tasks = np.zeros(len(avail), int)
        for i, j in enumerate(a):
            if j < 0:
                continue
            if P[i, j] < p_max and world.threat.flat[tg.aim[j]] > np.exp(-reach / 14):
                tasks[i] = 1
        return a, tasks
    gb.__name__ = "greedy_break"
    return gb


def with_breaks(base, name, reach=20):
    """Wrap any sector-assignment controller with the same firebreak rule greedy_break uses:
    a drone whose sector threatens something valuable lays a firebreak ahead of it."""
    def wrapped(world, avail, tg, P, rng):
        out = base(world, avail, tg, P, rng)
        a = out[0] if isinstance(out, tuple) else out
        tasks = np.array([1 if j >= 0 and world.threat.flat[tg.aim[j]] > np.exp(-reach / 14) else 0 for j in a])
        return a, tasks
    wrapped.__name__ = name
    return wrapped


greedy_c = make_greedy_c()
greedy_break = make_greedy_break()
greedy_c_stable = make_greedy_c(switch_cost=0.5)
ga = make_ga()
pso = make_pso()
ga_stable = make_ga(switch_cost=0.5)
pso_stable = make_pso(switch_cost=0.5)

CONTROLLERS = {
    "nearest": nearest,
    "greedy_c": greedy_c,
    "ga": ga,
    "pso": pso,
    "greedy_c_stable": greedy_c_stable,
    "greedy_break": greedy_break,
    "ga_break": with_breaks(make_ga(switch_cost=0.5), "ga_break"),
    "pso_break": with_breaks(make_pso(switch_cost=0.5), "pso_break"),
    "nearest_break": with_breaks(nearest, "nearest_break"),
    "ga_stable": ga_stable,
    "pso_stable": pso_stable,
}
