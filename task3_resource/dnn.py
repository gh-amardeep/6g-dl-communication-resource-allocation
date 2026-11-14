"""Unsupervised learning-to-optimise DNN for RB assignment + power allocation.

Input : channel-gain matrix H (U x N) as 10log10(a) (standardised with fixed constants) and the budget Pmax [dBm].
Output: assignment logits (N x U, softmax over users per RB) and power logits (N, softmax) -> p_n = Pmax * softmax_n.
        => every RB gets exactly one user (hard argmax), p_n >= 0 and sum p_n = Pmax (never above the budget).
Loss  : L = -SumRate[Mbps] + lambda * sum_u max(0, 1 - R_u[Mbps])^2 (straight-through hard assignment, so the loss is
        evaluated on the same discrete assignment that is used at test time).  No MILP labels are used.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .system import P, Params, dbm_to_w, draw_channels

DB_MEAN, DB_STD = 30.0, 13.0          # fixed input scaling of 10log10(a) (empirical mean 31 dB, std 12.4 dB)
PMIN_DBM, PMAX_DBM = 15.0, 40.0


class AllocNet(nn.Module):
    def __init__(self, U=P.U, N=P.N, hidden=256):
        super().__init__()
        self.U, self.N = U, N
        self.body = nn.Sequential(nn.Linear(U * N + 1, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, N * U + N))

    def forward(self, a_db, pmax_dbm):
        """a_db: (B,U,N) = 10log10(a); pmax_dbm: (B,). Returns assign logits (B,U,N), power fractions (B,N)."""
        z = torch.cat([((a_db - DB_MEAN) / DB_STD).flatten(1),
                       ((pmax_dbm - 27.5) / 12.5).unsqueeze(1)], dim=1)
        o = self.body(z)
        al = o[:, :self.N * self.U].view(-1, self.N, self.U).transpose(1, 2)     # (B,U,N)
        pf = torch.softmax(o[:, self.N * self.U:], dim=1)                        # (B,N), sums to 1
        return al, pf


def hard_assign(logits):
    """one-hot over users per RB with softmax gradient (straight-through)."""
    soft = torch.softmax(logits, dim=1)
    hard = torch.zeros_like(soft).scatter_(1, soft.argmax(1, keepdim=True), 1.0)
    return hard + soft - soft.detach()


def rates_torch(x, power, a, p: Params = P):
    """x (B,U,N), power (B,N) [W], a (B,U,N) -> user rates [Mbps] (B,U)."""
    return (x * p.B * torch.log2(1 + power.unsqueeze(1) * a) / 1e6).sum(dim=2)


def loss_fn(x, power, a, lam, p: Params = P):
    R = rates_torch(x, power, a, p)
    qos = torch.relu(p.r_min / 1e6 - R) ** 2
    return (-R.sum(1) + lam * qos.sum(1)).mean(), R


def _batch(n, rng, p=P):
    a = draw_channels(n, rng, p)
    pdbm = rng.uniform(PMIN_DBM, PMAX_DBM, size=n)
    return (torch.tensor(a, dtype=torch.float64), torch.tensor(pdbm, dtype=torch.float64))


def train(steps=6000, batch=256, lr=2e-3, lam=20.0, seed=20241114, hidden=256, log_every=100, save_to=None):
    """Train on freshly generated channel realisations (no labels). Returns (model, history list)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = AllocNet(hidden=hidden).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist, t0 = [], time.time()
    for it in range(1, steps + 1):
        a, pdbm = _batch(batch, rng)
        logits, pf = net(10 * torch.log10(a), pdbm)
        pw = torch.pow(10.0, (pdbm - 30) / 10).unsqueeze(1) * pf
        loss, R = loss_fn(hard_assign(logits), pw, a, lam)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % log_every == 0 or it == 1:
            hist.append({"step": it, "loss": loss.item(), "sum_rate_mbps": R.sum(1).mean().item(),
                         "qos_satisfied": (R >= 1.0 - 1e-9).double().mean().item(), "seconds": time.time() - t0})
    if save_to:
        save(net, save_to, {"steps": steps, "batch": batch, "lr": lr, "lam": lam, "seed": seed, "hidden": hidden})
    return net, hist


@torch.no_grad()
def predict(net, a: np.ndarray, pmax_w: float):
    """Single drop a (U,N) -> (assign (U,N) one-hot, power (N,) W)."""
    pdbm = 10 * np.log10(pmax_w) + 30
    at = torch.tensor(a[None], dtype=torch.float64)
    logits, pf = net(10 * torch.log10(at), torch.tensor([pdbm], dtype=torch.float64))
    x = np.zeros_like(a)
    x[logits[0].argmax(0).numpy(), np.arange(a.shape[1])] = 1
    return x, pmax_w * pf[0].numpy(), {}


def save(net, path, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "meta": meta}, path)
    Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load(path):
    ck = torch.load(path, map_location="cpu")
    net = AllocNet(hidden=ck["meta"].get("hidden", 256)).double()
    net.load_state_dict(ck["state"]); net.eval()
    return net
