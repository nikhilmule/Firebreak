"""Graph neural network policy for the drone fleet (decentralised actors, centralised critic).

Actor, run per drone:
  1. encode own state + shared scenario features
  2. R rounds of message passing over the communication graph (mean of neighbours)
  3. score each candidate sector from [own embedding, sector features]; plus a "hold" score
Critic: pools all drone embeddings + scenario features -> state value (used in training only).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .env import DRONE_F, CAND_F, GLOB_F


def mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h), nn.ReLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class GNNPolicy(nn.Module):
    def __init__(self, hidden=128, rounds=2, k=8):
        super().__init__()
        self.k = k
        self.enc = mlp(DRONE_F + GLOB_F, hidden, hidden)
        self.msg = nn.ModuleList([mlp(2 * hidden, hidden, hidden) for _ in range(rounds)])
        self.cand_enc = mlp(CAND_F, hidden, hidden)
        self.score = mlp(2 * hidden, hidden, 2)        # [attack, firebreak] per candidate
        self.hold = mlp(hidden, hidden, 1)
        self.value = mlp(2 * hidden + GLOB_F, hidden, 1)

    def forward(self, drone, cand, cand_mask, break_mask, act_mask, adj, glob, exists):
        """All inputs batched: drone (B,M,DF) cand (B,M,K,CF) cand_mask/break_mask (B,M,K) act_mask (B,M)
        adj (B,M,M) glob (B,GF) exists (B,M). Returns logits (B,M,2K+1) and value (B,).
        Action layout: K attack, K firebreak, 1 hold."""
        B, M, _ = drone.shape
        g = glob[:, None, :].expand(B, M, glob.shape[-1])
        h = torch.relu(self.enc(torch.cat([drone, g], -1)))
        deg = adj.sum(-1, keepdim=True).clamp(min=1)
        for layer in self.msg:
            m = adj @ h / deg                                  # mean over linked drones
            h = h + torch.relu(layer(torch.cat([h, m], -1)))
        c = torch.relu(self.cand_enc(cand))                    # (B,M,K,H)
        hk = h[:, :, None, :].expand(-1, -1, cand.shape[2], -1)
        s = self.score(torch.cat([hk, c], -1))                 # (B,M,K,2)
        att = s[..., 0].masked_fill(~cand_mask, -1e9)
        brk = s[..., 1].masked_fill(~break_mask, -1e9)
        logits = torch.cat([att, brk, self.hold(h)], -1)       # hold = last index

        ex = exists.float()[..., None]
        mean = (h * ex).sum(1) / ex.sum(1).clamp(min=1)
        mx = (h + (ex - 1) * 1e9).max(1).values
        v = self.value(torch.cat([mean, mx, glob], -1)).squeeze(-1)
        return logits, v

    @staticmethod
    def to_tensors(obs_list, device="cpu"):
        keys = ["drone", "cand", "cand_mask", "break_mask", "act_mask", "adj", "glob", "exists"]
        return {k: torch.as_tensor(np.stack([o[k] for o in obs_list]), device=device) for k in keys}

    def dist_and_value(self, t):
        logits, v = self(t["drone"], t["cand"], t["cand_mask"], t["break_mask"], t["act_mask"], t["adj"],
                         t["glob"], t["exists"])
        return torch.distributions.Categorical(logits=logits), v

    @torch.no_grad()
    def act_numpy(self, obs, greedy=False):
        t = self.to_tensors([obs])
        dist, _ = self.dist_and_value(t)
        a = dist.probs.argmax(-1) if greedy else dist.sample()
        return a[0].numpy()
