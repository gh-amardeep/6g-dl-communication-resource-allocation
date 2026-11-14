"""Sweep driver for Task 3: all six methods on identical channel realisations, power 15-40 dBm."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import allocation as al, dnn
from .amc_eval import AMCTable, load_amc, mean_sentence_bits, sentences_per_second
from .system import P, POWER_DBM_GRID, dbm_to_w, draw_channels, user_rates

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "exercise1_task3"
MODELS = ROOT / "models" / "resource_allocation"
METHODS = ["1 Round-robin + equal power", "2 Strongest user + water-filling", "3 MILP + equal power",
           "4 MILP + KKT water-filling", "5 DNN", "6 Hybrid DNN + KKT water-filling"]
SEED = 20241114


def jain(r):
    s = (r ** 2).sum()
    return float(r.sum() ** 2 / (len(r) * s)) if s > 0 else 0.0


def _timed(f, *a, **k):
    t = time.perf_counter()
    out = f(*a, **k)
    return out, time.perf_counter() - t


def evaluate_drop(a, pmax, net, table, bits, p=P):
    """Run all six methods on one channel drop -> list of dict (one per method) + user-rate arrays."""
    res = {}
    res[0], t1 = _timed(al.round_robin, a, pmax)
    res[1], t2 = _timed(al.strongest_user_wf, a, pmax)
    res[2], t3 = _timed(al.milp_equal_power, a, pmax)
    (p4, i4), t4w = _timed(al.kkt_waterfilling, res[2][0], a, pmax)
    res[3] = (res[2][0], p4, {"qos_feasible": i4["qos_feasible"], "milp_qos_feasible": res[2][2]["qos_feasible"]})
    t4 = t3 + t4w
    res[4], t5 = _timed(dnn.predict, net, a, pmax)
    (p6, i6), t6w = _timed(al.kkt_waterfilling, res[4][0], a, pmax)
    res[5] = (res[4][0], p6, {"qos_feasible": i6["qos_feasible"]})
    t6 = t5 + t6w
    times = [t1, t2, t3, t4, t5, t6]
    rows, rates = [], []
    for m, (x, pw, info) in res.items():
        R = user_rates(x, pw, a, p)
        spc, _, _, _ = sentences_per_second(x, pw, a, table, bits, p)
        rows.append({"method": METHODS[m], "sum_rate_mbps": R.sum() / 1e6, "qos_frac": float((R >= p.r_min * (1 - 1e-9)).mean()),
                     "all_qos": bool((R >= p.r_min * (1 - 1e-9)).all()), "jain": jain(R), "time_ms": times[m] * 1e3,
                     "sentences_per_s": spc, "power_used_w": float(pw.sum()), "budget_w": pmax,
                     "valid": bool((x.sum(0) <= 1).all() and (pw >= 0).all() and pw.sum() <= pmax * (1 + 1e-9)),
                     "qos_feasible_flag": info.get("qos_feasible", np.nan)})
        rates.append(R / 1e6)
    return rows, rates


def run_sweep(n_drops=300, seed=SEED, power_dbm=POWER_DBM_GRID, net=None, root=ROOT, verbose=True):
    net = net or dnn.load(MODELS / "dnn_alloc.pt")
    amc = load_amc(root)
    table, bits = AMCTable(amc), mean_sentence_bits(root)
    A = draw_channels(n_drops, np.random.default_rng(seed))            # identical drops for every power level and method
    per_drop, user_rates_all = [], {}
    for pdbm in power_dbm:
        pmax = dbm_to_w(pdbm)
        store = [[] for _ in METHODS]
        for d, a in enumerate(A):
            rows, rates = evaluate_drop(a, pmax, net, table, bits)
            for m, r in enumerate(rows):
                per_drop.append({"power_dbm": pdbm, "drop": d, **r})
                store[m].append(rates[m])
        user_rates_all[pdbm] = np.array(store)                        # (method, drops, users) [Mbps]
        if verbose:
            print("power", pdbm, "done", flush=True)
    df = pd.DataFrame(per_drop)
    return df, user_rates_all, bits


def summarise(df: pd.DataFrame, user_rates_all: dict) -> pd.DataFrame:
    out = []
    for (pdbm, method), g in df.groupby(["power_dbm", "method"], sort=False):
        m = METHODS.index(method)
        pooled = user_rates_all[pdbm][m].ravel()
        out.append({"power_dbm": pdbm, "method": method, "sum_rate_mbps": g.sum_rate_mbps.mean(),
                    "sum_rate_sem": g.sum_rate_mbps.std(ddof=1) / np.sqrt(len(g)),
                    "qos_pct": 100 * g.qos_frac.mean(), "all_users_qos_pct": 100 * g.all_qos.mean(),
                    "jain": g.jain.mean(), "p5_user_rate_mbps": float(np.percentile(pooled, 5)),
                    "time_ms": g.time_ms.median(), "sentences_per_s": g.sentences_per_s.mean(),
                    "sentences_per_s_sem": g.sentences_per_s.std(ddof=1) / np.sqrt(len(g)),
                    "all_valid": bool(g.valid.all()), "n_drops": len(g)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ plots
STYLE = {METHODS[0]: ("tab:gray", "o", "--"), METHODS[1]: ("tab:orange", "s", "--"), METHODS[2]: ("tab:blue", "^", "-"),
         METHODS[3]: ("tab:green", "v", "-"), METHODS[4]: ("tab:red", "D", "-"), METHODS[5]: ("tab:purple", "P", "-")}


def plot_metric(summary: pd.DataFrame, col: str, ylabel: str, fname: str, logy=False, err=None, title=None):
    import matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in METHODS:
        g = summary[summary.method == m].sort_values("power_dbm")
        c, mk, ls = STYLE[m]
        ax.errorbar(g.power_dbm, g[col], yerr=(1.96 * g[err] if err else None), color=c, marker=mk, ls=ls, ms=4, lw=1.4, capsize=2, label=m)
    ax.set_xlabel("BS transmit power P_max [dBm]"); ax.set_ylabel(ylabel); ax.grid(alpha=.3)
    if logy: ax.set_yscale("log")
    ax.set_title(title or ylabel); ax.legend(fontsize=7.5)
    fig.tight_layout()
    (RES / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RES / "figures" / fname, dpi=150)
    return fig


def plot_all(summary: pd.DataFrame):
    return {
        "sum_rate": plot_metric(summary, "sum_rate_mbps", "Sum rate [Mbps]", "sum_rate_vs_power.png", err="sum_rate_sem"),
        "qos": plot_metric(summary, "qos_pct", "Users with R_u >= 1 Mbps [%]", "qos_satisfaction_vs_power.png"),
        "jain": plot_metric(summary, "jain", "Jain's fairness index", "jain_fairness_vs_power.png"),
        "p5": plot_metric(summary, "p5_user_rate_mbps", "5th-percentile user rate [Mbps]", "p5_user_rate_vs_power.png"),
        "time": plot_metric(summary, "time_ms", "Computation time per drop [ms]", "computation_time_vs_power.png", logy=True),
        "sps": plot_metric(summary, "sentences_per_s", "Correctly delivered sentences per second", "sentences_per_second_vs_power.png", err="sentences_per_s_sem"),
    }
