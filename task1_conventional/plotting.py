"""Matplotlib figures for Task 1 (all saved to results/figures/{mode}_*.png)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .communication_system import MCS_TABLE
from .modulation import Modem

COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
LABEL = {m.index: f"MCS{m.index} {m.modulation} {m.rate}" for m in MCS_TABLE}
CH_NAME = {"awgn": "AWGN", "rayleigh": "Rayleigh block fading"}
MODE_NOTE = {"quick": "QUICK MODE - small sample, not final results", "full": ""}


def _finish(fig, ax, mode, title, xlabel, ylabel, path, legend=True, note=None):
    for a in np.atleast_1d(ax).ravel():
        a.set_xlabel(xlabel)
        a.grid(True, which="both", alpha=0.3)
    np.atleast_1d(ax).ravel()[0].set_ylabel(ylabel)
    fig.suptitle(title + (f"\n[{MODE_NOTE[mode]}]" if MODE_NOTE.get(mode) else ""), fontsize=11)
    if legend:
        np.atleast_1d(ax).ravel()[-1].legend(fontsize=8)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color="gray")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def plot_uncoded(df, mode="quick", path=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, chan in zip(ax, ["awgn", "rayleigh"]):
        d = df[df.channel == chan]
        for mod, mk in (("BPSK", "o"), ("QPSK", "s")):
            s = d[d.modulation == mod]
            if len(s):
                a.semilogy(s.ebn0_db, s.ber_sim.replace(0, np.nan), mk, ms=5, label=f"{mod} simulated")
        s = d[d.modulation == "BPSK"]
        a.semilogy(s.ebn0_db, s.ber_theory, "k-", label="theory")
        a.set_title(CH_NAME[chan] + (" (per-symbol fading, ZF)" if chan == "rayleigh" else ""), fontsize=10)
        a.set_ylim(1e-7, 1)
    return _finish(fig, ax, mode, "Task 1(a): uncoded BER vs theory", "Eb/N0 [dB] (uncoded: = Es/N0 for BPSK)",
                   "BER", path)


def _lines(ax, d, col, ylog=True, label=None, ls="-", marker="o", ci=False, floor=None):
    for m in sorted(d.mcs.unique()):
        s = d[d.mcs == m].sort_values("snr_db")
        y = s[col].astype(float).values
        y = np.where(y <= 0, np.nan, y) if ylog else y
        (ax.semilogy if ylog else ax.plot)(s.snr_db, y, ls, marker=marker, ms=3, color=COLORS[m - 1],
                                           label=(LABEL[m] if label is None else label))


def plot_ber_bler(sweep, mode="quick", prefix=None):
    """Return dict name -> figure."""
    figs = {}
    for metric, name, ylab in (("coded_ber", "coded_ber", "Coded BER (info bits, after Viterbi)"),
                               ("bler", "bler", "Sentence BLER")):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        for a, chan in zip(ax, ["awgn", "rayleigh"]):
            d = sweep[sweep.channel == chan]
            _lines(a, d, metric)
            a.set_title(CH_NAME[chan], fontsize=10)
            a.set_ylim(1e-5 if metric == "coded_ber" else 1e-4, 1.05)
            if metric == "bler":
                a.axhline(0.1, color="k", ls=":", lw=1)
                a.text(a.get_xlim()[0], 0.11, " 10% BLER target", fontsize=7)
        figs[name] = _finish(fig, ax, mode, f"Task 1(b): {ylab} vs SNR (Es/N0)", "SNR = Es/N0 [dB]", ylab,
                             None if prefix is None else f"{prefix}_{name}_vs_snr.png",
                             note="zero-error points are not drawn on the log axis")
    # AWGN vs Rayleigh per MCS
    for metric, ylab in (("coded_ber", "Coded BER"), ("bler", "Sentence BLER")):
        fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
        for m, a in zip(MCS_TABLE, ax.ravel()):
            for chan, ls, mk in (("awgn", "-", "o"), ("rayleigh", "--", "s")):
                d = sweep[(sweep.channel == chan) & (sweep.mcs == m.index)].sort_values("snr_db")
                y = d[metric].astype(float).values
                a.semilogy(d.snr_db, np.where(y <= 0, np.nan, y), ls, marker=mk, ms=3, label=CH_NAME[chan],
                           color=COLORS[m.index - 1] if chan == "awgn" else "k")
            a.set_title(LABEL[m.index], fontsize=9)
            a.set_ylim(1e-5 if metric == "coded_ber" else 1e-4, 1.05)
            a.legend(fontsize=7)
        _finish(fig, ax, mode, f"Task 1(b): AWGN vs Rayleigh - {ylab}", "SNR = Es/N0 [dB]", ylab,
                None if prefix is None else f"{prefix}_{metric}_awgn_vs_rayleigh.png", legend=False)
        figs[f"{metric}_awgn_vs_rayleigh"] = fig
    return figs


def plot_text_quality(sweep, mode="quick", prefix=None):
    figs = {}
    for col, ylab in (("bleu", "Sentence BLEU (0-1)"), ("word_acc", "Word accuracy (H/N)"),
                      ("cos", "Embedding cosine similarity")):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
        for a, chan in zip(ax, ["awgn", "rayleigh"]):
            _lines(a, sweep[sweep.channel == chan], col, ylog=False)
            a.set_title(CH_NAME[chan], fontsize=10)
            a.set_ylim(-0.02, 1.02)
        figs[col] = _finish(fig, ax, mode, f"Task 1(c): {ylab} vs SNR", "SNR = Es/N0 [dB]", ylab,
                            None if prefix is None else f"{prefix}_{col}_vs_snr.png")
    return figs


def plot_cliff(sweep, mcs=4, mode="quick", path=None, chan="awgn"):
    """BER / BLER / similarity of one MCS on one axis system: the cliff effect."""
    d = sweep[(sweep.channel == chan) & (sweep.mcs == mcs)].sort_values("snr_db")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].semilogy(d.snr_db, np.where(d.coded_ber > 0, d.coded_ber, np.nan), "o-", label="coded BER")
    ax[0].semilogy(d.snr_db, np.where(d.bler > 0, d.bler, np.nan), "s-", label="sentence BLER")
    ax[0].semilogy(d.snr_db, np.where(d.raw_ber > 0, d.raw_ber, np.nan), "^--", label="raw channel BER (pre-decoder)")
    ax[0].set_ylim(1e-5, 1.05)
    ax[1].plot(d.snr_db, d.bleu, "o-", label="BLEU")
    ax[1].plot(d.snr_db, d.word_acc, "s-", label="word accuracy")
    ax[1].plot(d.snr_db, d.cos, "^-", label="cosine similarity")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    _finish(fig, ax, mode, f"Cliff effect: {LABEL[mcs]} over {CH_NAME[chan]}", "SNR = Es/N0 [dB]", "error rate",
            path, legend=False)
    ax[1].set_ylabel("quality (1 = perfect)")
    fig.savefig(path, dpi=150) if path else None
    return fig


def plot_amc(curve_by_chan, mode="quick", prefix=None):
    figs = {}
    spec = [("mcs", "Selected MCS index", "amc_mcs_vs_snr"),
            ("efficiency", "Nominal spectral efficiency [bit/channel use]", "amc_efficiency_vs_snr"),
            ("bler", "BLER of selected MCS", "amc_bler_vs_snr")]
    for col, ylab, name in spec:
        fig, ax = plt.subplots(1, len(curve_by_chan), figsize=(6 * len(curve_by_chan), 4.2), squeeze=False)
        ax = ax[0]
        for a, (chan, cv) in zip(ax, curve_by_chan.items()):
            if col == "bler":
                a.semilogy(cv.snr_db, np.where(cv.bler > 0, cv.bler, np.nan), "o-", ms=3)
                a.axhline(0.1, color="k", ls=":")
                a.set_ylim(1e-4, 1.05)
            else:
                a.step(cv.snr_db, cv[col], where="post", color="C0")
                if col == "efficiency":
                    a.step(cv.snr_db, cv.goodput, where="post", color="C1", label="goodput = eta(1-BLER)")
                    a.legend(fontsize=8)
                if col == "mcs":
                    a.set_yticks([m.index for m in MCS_TABLE])
            out = cv[cv.outage]
            if len(out):
                a.plot(out.snr_db, [a.get_ylim()[0]] * len(out), "rv", ms=4, label="outage (fallback to MCS1)")
            a.set_title(CH_NAME[chan], fontsize=10)
        figs[name] = _finish(fig, ax, mode, f"Task 1(d): AMC - {ylab}", "SNR = Es/N0 [dB]", ylab,
                             None if prefix is None else f"{prefix}_{name}.png", legend=False)
    return figs


def plot_constellations(path=None):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    for a, n in zip(ax, ["QPSK", "16QAM", "64QAM"]):
        mo = Modem(n)
        pts, tab = mo.constellation()
        a.scatter(pts.real, pts.imag, s=14)
        if n != "64QAM":
            for p, b in zip(pts, tab):
                a.annotate("".join(map(str, b)), (p.real, p.imag), textcoords="offset points", xytext=(0, 5),
                           ha="center", fontsize=7)
        a.set_title(f"{n} (Gray, Es=1)")
        a.set_aspect("equal")
        a.set_xlabel("In-phase")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Quadrature")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def make_all_figures(mode="quick", root="."):
    root = Path(root)
    tab, fig_dir = root / "results/tables", root / "results/figures"
    prefix = str(fig_dir / mode)
    fig_dir.mkdir(parents=True, exist_ok=True)
    unc = pd.read_csv(tab / f"{mode}_uncoded_ber.csv")
    plot_uncoded(unc, mode, f"{prefix}_uncoded_ber.png")
    sweep = pd.read_csv(tab / f"{mode}_mcs_sweep.csv")
    plot_ber_bler(sweep, mode, prefix)
    plot_text_quality(sweep, mode, prefix)
    plot_cliff(sweep, 4, mode, f"{prefix}_cliff_mcs4_awgn.png", "awgn")
    plot_cliff(sweep, 4, mode, f"{prefix}_cliff_mcs4_rayleigh.png", "rayleigh")
    curves = {c: pd.read_csv(tab / f"{mode}_amc_curve_{c}.csv") for c in ("awgn", "rayleigh")
              if (tab / f"{mode}_amc_curve_{c}.csv").exists()}
    if curves:
        plot_amc(curves, mode, prefix)
    plot_constellations(f"{prefix}_constellations.png")
    plt.close("all")
