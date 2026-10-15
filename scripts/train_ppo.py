"""Train the GNN fleet policy: imitation (BC + DAgger) first, then anchored PPO.

Actions per drone: attack one of K candidate sectors, lay a firebreak ahead of one, or hold.
The default expert is greedy_break (greedy assignment + a hand-made firebreak rule).

Recommended run (CPU is fine):

    python train_ppo.py --mode break --bc-episodes 20 --dagger-rounds 2 --iters 200 \
        --wind-min 2 --wind-max 12 --out ../runs/gnn4

Stage 1  Behaviour cloning: copy the expert (greedy_break) on states it visits.
Stage 2  DAgger: let the policy fly, ask the expert what it would have done in the states the
         policy actually reaches, add those labels, retrain. Fixes the "never saw this mistake" gap.
Stage 3  PPO fine-tuning with per-drone clipped ratios (MAPPO style) and an imitation anchor that
         fades out over --anchor-iters, so PPO improves on the expert instead of forgetting it.

Checkpoints: bc.pt (after imitation), best.pt (best held-out score, imitation included), last.pt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from firebreak import CONTROLLERS, run_episode                     # noqa: E402
from firebreak.env import FirebreakEnv, PolicyController           # noqa: E402
from firebreak.policy import GNNPolicy                             # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drones", type=int, default=6)
    ap.add_argument("--fires", type=int, default=18)
    ap.add_argument("--wind-min", type=float, default=7.0)
    ap.add_argument("--wind-max", type=float, default=7.0)
    # imitation
    ap.add_argument("--expert", default="greedy_break")
    ap.add_argument("--bc-episodes", type=int, default=40, help="0 = skip imitation")
    ap.add_argument("--bc-epochs", type=int, default=15)
    ap.add_argument("--dagger-rounds", type=int, default=4)
    ap.add_argument("--dagger-episodes", type=int, default=10)
    # PPO
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--episodes-per-iter", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--anchor", type=float, default=0.5, help="weight of the imitation loss at iter 1")
    ap.add_argument("--anchor-iters", type=int, default=100, help="iters over which the anchor fades")
    ap.add_argument("--anchor-min", type=float, default=0.1, help="anchor never fades below this")
    ap.add_argument("--smooth", type=float, default=0.1, help="label smoothing for imitation")
    ap.add_argument("--reward-scale", type=float, default=0.1)
    # model / scenario
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=2, help="message-passing rounds")
    ap.add_argument("--comm-range", type=float, default=1500.0, help="metres")
    ap.add_argument("--area-weight", type=float, default=0.002, help="penalty per burnt cell next to lost value")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="start from this checkpoint (skips imitation)")
    ap.add_argument("--mode", default="break", choices=["break", "full"],
                    help="break: greedy picks sectors, the policy picks attack/firebreak/hold; full: policy picks both")
    ap.add_argument("--out", default="runs/gnn")
    return ap.parse_args()


def make_env(a):
    wind = a.wind_min if a.wind_min == a.wind_max else (a.wind_min, a.wind_max)
    return FirebreakEnv(drones=a.drones, fires=a.fires, wind=wind, comm_range=a.comm_range,
                        area_weight=a.area_weight, mode=a.mode)


def save(policy, a, path):
    torch.save({"state": policy.state_dict(), "config": vars(a)}, path)


# ---------------------------------------------------------------- imitation
def collect(env, expert, n, seed_base, policy=None):
    """States + expert labels. policy=None: the expert flies; else the policy flies (DAgger)."""
    obs_buf, lab_buf = [], []
    for e in range(n):
        obs, done = env.reset(seed_base + e), False
        while not done:
            lab = env.expert_actions(expert)
            obs_buf.append(obs)
            lab_buf.append(lab)
            act = lab if policy is None else policy.act_numpy(obs, greedy=False)
            obs, _, done, _ = env.step(act)
    return obs_buf, lab_buf


SMOOTH = 0.1   # label smoothing: keep some probability on the other valid actions so PPO can explore


def smoothed_nll(dist, labels, m, eps=None):
    eps = SMOOTH if eps is None else eps
    logp = dist.logits.log_softmax(-1)                        # (B,M,A)
    valid = (dist.logits > -1e8).float()
    nll = -logp.gather(-1, labels[..., None]).squeeze(-1)
    uni = -(logp * valid).sum(-1) / valid.sum(-1).clamp(min=1)
    return (((1 - eps) * nll + eps * uni) * m).sum() / m.sum().clamp(min=1)


def imitation_loss(policy, t, labels):
    dist, _ = policy.dist_and_value(t)
    return smoothed_nll(dist, labels, t["act_mask"].float())


def fit_bc(policy, opt, obs, labels, a, tag):
    labels = torch.as_tensor(np.stack(labels))
    for ep in range(a.bc_epochs):
        perm = torch.randperm(len(obs))
        tot = 0.0
        for s in range(0, len(perm), a.minibatch):
            idx = perm[s:s + a.minibatch]
            loss = imitation_loss(policy, GNNPolicy.to_tensors([obs[i] for i in idx]), labels[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
    print(f"[{tag}] {len(obs)} labelled decisions, final loss {tot / len(obs):.3f}", flush=True)


# --------------------------------------------------------------------- PPO
def rollout(policy, env, expert, n_episodes, seed_base, reward_scale):
    buf = dict(obs=[], act=[], logp=[], val=[], rew=[], done=[], lab=[])
    stats = []
    for e in range(n_episodes):
        obs, done = env.reset(seed_base + e), False
        while not done:
            buf["lab"].append(torch.as_tensor(env.expert_actions(expert)))
            t = GNNPolicy.to_tensors([obs])
            with torch.no_grad():
                dist, v = policy.dist_and_value(t)
                act = dist.sample()
                logp = dist.log_prob(act)                    # per drone (1, M)
            nobs, r, done, info = env.step(act[0].numpy())
            buf["obs"].append(obs)
            buf["act"].append(act[0])
            buf["logp"].append(logp[0])
            buf["val"].append(v[0])
            buf["rew"].append(r * reward_scale)
            buf["done"].append(done)
            obs = nobs
        stats.append(info)
    return buf, stats


def gae(rew, val, done, gamma, lam):
    T = len(rew)
    adv = torch.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        nv = 0.0 if done[t] or t + 1 >= T else float(val[t + 1])
        delta = rew[t] + gamma * nv - float(val[t])
        last = delta + gamma * lam * (0.0 if done[t] else last)
        adv[t] = last
    return adv, adv + torch.stack(val)


def ppo_update(policy, opt, buf, a, anchor):
    adv, ret = gae(buf["rew"], buf["val"], buf["done"], a.gamma, a.lam)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    acts, old, labs = torch.stack(buf["act"]), torch.stack(buf["logp"]), torch.stack(buf["lab"])
    n = len(buf["obs"])
    out = []
    for _ in range(a.ppo_epochs):
        perm = torch.randperm(n)
        for s in range(0, n, a.minibatch):
            idx = perm[s:s + a.minibatch]
            t = GNNPolicy.to_tensors([buf["obs"][i] for i in idx])
            dist, v = policy.dist_and_value(t)
            m = t["act_mask"].float()
            k = m.sum().clamp(min=1)
            ratio = torch.exp(dist.log_prob(acts[idx]) - old[idx])        # one ratio per drone
            A = adv[idx][:, None]
            pg = -(torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A) * m).sum() / k
            vloss = 0.5 * (v - ret[idx]).pow(2).mean()
            ent = (dist.entropy() * m).sum() / k
            imit = smoothed_nll(dist, labs[idx], m)
            loss = pg + vloss - a.entropy * ent + anchor * imit
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            out = [pg.item(), vloss.item(), ent.item(), imit.item()]
    return out


def evaluate(policy, a, seeds, winds):
    policy.eval()
    ctl = PolicyController(policy, a.drones, comm_range=a.comm_range, greedy=True, mode=a.mode)
    res = [run_episode(ctl, seed=s, drones=a.drones, fires=a.fires, wind=w) for s, w in zip(seeds, winds)]
    policy.train()
    return float(np.mean([r["homes_lost"] for r in res])), float(np.mean([r["value_saved_pct"] for r in res]))


def main():
    global SMOOTH
    a = parse()
    SMOOTH = a.smooth
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=2)
    env = make_env(a)
    expert = CONTROLLERS[a.expert]
    policy = GNNPolicy(hidden=a.hidden, rounds=a.rounds)
    opt = torch.optim.Adam(policy.parameters(), lr=a.lr)

    rng = np.random.default_rng(12345)
    eval_seeds = [900000 + i for i in range(a.eval_seeds)]
    eval_winds = (rng.uniform(a.wind_min, a.wind_max, a.eval_seeds) if a.wind_min != a.wind_max
                  else [a.wind_min] * a.eval_seeds)
    base = [run_episode(expert, seed=s, drones=a.drones, fires=a.fires, wind=w) for s, w in zip(eval_seeds, eval_winds)]
    base_homes = float(np.mean([r["homes_lost"] for r in base]))
    print(f"[eval] {a.expert} on {a.eval_seeds} held-out maps: {base_homes:.1f} homes lost", flush=True)

    best = float("inf")
    if a.init:
        policy.load_state_dict(torch.load(a.init, map_location="cpu")["state"])
    elif a.bc_episodes:
        bc_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        obs, lab = collect(env, expert, a.bc_episodes, 500000)
        fit_bc(policy, bc_opt, obs, lab, a, "bc")
        h, _ = evaluate(policy, a, eval_seeds, eval_winds)
        print(f"[eval] after BC: {h:.1f} homes lost vs {a.expert} {base_homes:.1f}", flush=True)
        for r in range(a.dagger_rounds):
            o2, l2 = collect(env, expert, a.dagger_episodes, 600000 + 1000 * r, policy=policy)
            obs += o2
            lab += l2
            fit_bc(policy, bc_opt, obs, lab, a, f"dagger {r + 1}")
            h, _ = evaluate(policy, a, eval_seeds, eval_winds)
            print(f"[eval] after DAgger round {r + 1}: {h:.1f} homes lost vs {a.expert} {base_homes:.1f}", flush=True)
        save(policy, a, os.path.join(a.out, "bc.pt"))
        best = h
        save(policy, a, os.path.join(a.out, "best.pt"))

    log = open(os.path.join(a.out, "log.csv"), "w")
    log.write("iter,episodes,train_homes_lost,eval_homes_lost,eval_value_saved,expert_homes_lost,anchor,pg,vloss,entropy,imitation,secs\n")
    for it in range(1, a.iters + 1):
        t0 = time.time()
        anchor = max(a.anchor_min, a.anchor * max(0.0, 1 - (it - 1) / max(1, a.anchor_iters)))
        buf, stats = rollout(policy, env, expert, a.episodes_per_iter, it * 1000, a.reward_scale)
        pg, vl, ent, imit = ppo_update(policy, opt, buf, a, anchor)
        train_h = float(np.mean([s["homes_lost"] for s in stats]))
        eh = ev = float("nan")
        if it % a.eval_every == 0 or it == a.iters:
            eh, ev = evaluate(policy, a, eval_seeds, eval_winds)
            if eh < best:
                best = eh
                save(policy, a, os.path.join(a.out, "best.pt"))
        save(policy, a, os.path.join(a.out, "last.pt"))
        secs = time.time() - t0
        log.write(f"{it},{it * a.episodes_per_iter},{train_h:.2f},{eh:.2f},{ev:.2f},{base_homes:.2f},{anchor:.3f},"
                  f"{pg:.4f},{vl:.4f},{ent:.4f},{imit:.4f},{secs:.1f}\n")
        log.flush()
        msg = (f"[ppo] iter {it} train {train_h:.1f} | anchor {anchor:.2f} imit {imit:.2f} "
               f"ent {ent:.2f} v {vl:.3f} | {secs:.0f}s")
        if not np.isnan(eh):
            msg += f" | eval {eh:.1f} vs {a.expert} {base_homes:.1f} (best {best:.1f})"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
