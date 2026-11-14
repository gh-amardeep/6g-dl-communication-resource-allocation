"""Train / evaluate / plot helpers shared by the three Task-2 notebooks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from task1_conventional.channel import complex_gaussian, ebn0_to_esn0_db, noise_variance
from task1_conventional.modulation import get_modem
from . import baselines as bl
from .autoencoder import (M, Autoencoder, awgn_channel, load_model, rayleigh_channel,
                          save_model, train, bits_to_int, int_to_bits)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "autoencoder"
RES = ROOT / "results" / "exercise1_task2"
(FIG, TAB) = RES / "figures", RES / "tables"
for d in (MODELS, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

SEED = 20241113
# Training settings (Eb/N0 range in dB is sampled uniformly per step). Documented in README.
CFG = {
    "ae74_awgn":     dict(n_real=7, rx_extra=0, channel="awgn",     rate=4 / 7, ebn0=(0.0, 8.0),  steps=6000, batch=1024, lr=3e-3),
    "ae74_rayleigh": dict(n_real=7, rx_extra=1, channel="rayleigh", rate=4 / 7, ebn0=(0.0, 20.0), steps=8000, batch=1024, lr=3e-3),
    "ae24_awgn":     dict(n_real=2, rx_extra=0, channel="awgn",     rate=4.0, power=0.5, ebn0=(4.0, 12.0), steps=6000, batch=1024, lr=3e-3),
}
# "rate" = information bits per transmitted symbol, so Es/N0 = Eb/N0 * rate:
#   (7,4): 4/7 bit per real symbol (Es = 1 per real symbol)
#   (2,4): 4 bits per complex symbol (Es = 1 per complex symbol, 0.5 per real dimension)


def train_model(name: str, seed: int = SEED, steps: int | None = None, save: bool = True):
    import torch
    from .autoencoder import train as _train
    c = dict(CFG[name]); 
    if steps is not None:
        c["steps"] = steps
    torch.manual_seed(seed)
    model = Autoencoder(c["n_real"], hidden=16, rx_extra=c["rx_extra"], power=c.get("power", 1.0))
    fac = awgn_channel if c["channel"] == "awgn" else rayleigh_channel
    hist = _train(model, fac, c["ebn0"], c["rate"], steps=c["steps"], batch=c["batch"], lr=c["lr"], seed=seed)
    model.eval()
    if save:
        save_model(model, MODELS / f"{name}.pt", meta={"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()}, "seed": seed})
        pd.DataFrame(hist, columns=["step", "loss"]).to_csv(TAB / f"{name}_training_loss.csv", index=False)
        json.dump({"name": name, "seed": seed, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()}},
                  open(MODELS / f"{name}_config.json", "w"), indent=2)
    return model, hist


def eval_74(ae, channel, ebn0_grid, n_blocks, seed=SEED):
    """BER/BLER of all four methods on identical information bits (and identical noise/fading
    for the coded systems) at each Eb/N0."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in ebn0_grid:
        res, n = bl.simulate_point(e, n_blocks, rng, channel=channel, ae=ae)
        for k, (be, ble) in res.items():
            rows.append(dict(channel=channel, method=k, ebn0_db=e, n_blocks=n, n_bits=4 * n,
                             bit_errors=be, ber=be / (4 * n), block_errors=ble, bler=ble / n))
    df = pd.DataFrame(rows)
    th = bl.q_bpsk_awgn if channel == "awgn" else bl.bpsk_rayleigh
    df["uncoded_theory_ber"] = np.where(df.method == "uncoded_bpsk", th(df.ebn0_db), np.nan)
    return df


# --------------------------------------------------------------------------- (2,4)
def eval_24(ae, ebn0_grid, n_blocks, seed=SEED):
    """AE (2,4) vs Gray 16-QAM (Task 1 modem) on one complex symbol per message, AWGN.
    Es = 1 per complex symbol, Es/N0 = 4 Eb/N0. Both use the same bits and noise."""
    import torch
    rng = np.random.default_rng(seed)
    qam = get_modem("16QAM")
    cb = ae.codebook().detach().numpy(); cbc = cb[:, 0] + 1j * cb[:, 1]
    rows = []
    for e in ebn0_grid:
        esn0 = float(ebn0_to_esn0_db(e, bps=4))
        n0 = noise_variance(esn0)
        be = {"autoencoder": 0, "qam16": 0}; se = {"autoencoder": 0, "qam16": 0}
        done = 0
        while done < n_blocks:
            n = min(500_000, n_blocks - done)
            msg = rng.integers(0, M, n); bits = int_to_bits(msg)
            nz = complex_gaussian(n, n0, rng)
            # AE
            r = cbc[msg] + nz
            rin = torch.from_numpy(np.stack([r.real, r.imag], 1).astype(np.float32))
            bh = int_to_bits(ae.predict(rin).numpy())
            # 16-QAM, Gray, hard (nearest-point) decision = ML on AWGN
            s = qam.modulate(bits.reshape(-1))
            yb = qam.demodulate_hard(s + nz)
            bq = np.asarray(yb).reshape(n, 4)
            for k, b in (("autoencoder", bh), ("qam16", bq)):
                err = b != bits
                be[k] += int(err.sum()); se[k] += int(err.any(1).sum())
            done += n
        for k in be:
            rows.append(dict(method=k, ebn0_db=e, esn0_db=esn0, n_blocks=n_blocks, bit_errors=be[k], ber=be[k] / (4 * n_blocks), symbol_errors=se[k], ser=se[k] / n_blocks))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plots
import matplotlib
matplotlib.use("Agg") if not hasattr(__import__("sys"), "ps1") and "ipykernel" not in __import__("sys").modules else None
import matplotlib.pyplot as plt

LABELS = {"uncoded_bpsk": "Uncoded BPSK", "hamming_hard": "Hamming(7,4) hard decision",
          "hamming_ml": "Hamming(7,4) ML (soft)", "autoencoder": "Autoencoder", "qam16": "Gray 16-QAM"}
STYLE = {"uncoded_bpsk": ("k", "o", "-"), "hamming_hard": ("C1", "s", "--"), "hamming_ml": ("C2", "^", "-"),
         "autoencoder": ("C3", "D", "-"), "qam16": ("C0", "o", "--")}


def plot_curves(df, metric, title, path=None, xcol="ebn0_db", ylabel="BER", methods=None, ax=None, suffix=""):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for k in (methods or df.method.unique()):
        d = df[(df.method == k) & (df[metric] > 0)]
        c, mk, ls = STYLE[k]
        ax.semilogy(d[xcol], d[metric], color=c, marker=mk, ls=ls, ms=4, label=LABELS[k] + suffix)
    ax.set_xlabel("Eb/N0 [dB] (energy per information bit)"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    if own:
        ax.figure.tight_layout()
        if path:
            ax.figure.savefig(path, dpi=150)
    return ax


def plot_constellations(ae, path=None):
    cb = ae.codebook().detach().numpy(); c = cb[:, 0] + 1j * cb[:, 1]
    q = np.asarray(get_modem("16QAM").constellation()[0]).ravel()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
    for a, pts, t, col in ((axs[0], c, "Learned (2,4) autoencoder", "C3"), (axs[1], q, "Conventional normalised 16-QAM", "C0")):
        a.scatter(pts.real, pts.imag, c=col, s=40)
        for i, z in enumerate(pts):
            a.annotate(str(i), (z.real, z.imag), textcoords="offset points", xytext=(4, 4), fontsize=7)
        a.set_title(f"{t}\naverage power = {np.mean(abs(pts)**2):.3f}, min distance = {min(abs(a_-b_) for i, a_ in enumerate(pts) for b_ in pts[i+1:]):.3f}", fontsize=9)
        a.set_xlabel("In-phase"); a.set_ylabel("Quadrature"); a.grid(alpha=0.3); a.set_aspect("equal")
        a.set_xlim(-1.6, 1.6); a.set_ylim(-1.6, 1.6)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
