"""Experiment drivers shared by ``main.py`` and the notebooks.

Two modes
---------
quick : 200 test sentences x 1 repetition, coarse SNR grids   (debug / demonstration)
full  : 2000 test sentences x 3-5 repetitions, fine SNR grids  (final plots / report)
Every output file is prefixed with its mode (``quick_`` / ``full_``) so the two can never
be confused. The MCS sweep is checkpointed row by row (CSV) and resumes automatically.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erfc
from tqdm.auto import tqdm

from . import channel as ch
from .amc import AMC, amc_with_estimation, build_lut
from .communication_system import MCS_TABLE, ConventionalLink, Frames
from .dataset import load_split
from .equalizer import zf_equalize
from .huffman import HuffmanCodec
from .metrics import SentenceEmbedder, text_quality
from .modulation import Modem

SEED = 20241111

CONFIG = {
    "quick": dict(n_sentences=200, reps_awgn=1, reps_rayleigh=1,
                  awgn_snr=list(np.arange(-2.0, 30.1, 2.0)),
                  rayleigh_snr=list(np.arange(0.0, 44.1, 4.0)),
                  uncoded_ebn0=list(np.arange(0.0, 9.1, 1.0)), uncoded_max_bits=2_000_000,
                  uncoded_min_errors=100),
    "full": dict(n_sentences=2000, reps_awgn=3, reps_rayleigh=5,
                 awgn_snr=list(np.arange(-2.0, 30.01, 0.5)),
                 rayleigh_snr=list(np.arange(0.0, 44.01, 2.0)),
                 uncoded_ebn0=list(np.arange(0.0, 10.01, 0.5)), uncoded_max_bits=100_000_000,
                 uncoded_min_errors=400),
}
BLER_TARGET = 0.10


def paths(root="."):
    root = Path(root)
    p = {"fig": root / "results/figures", "tab": root / "results/tables", "log": root / "results/logs",
         "data": root / "data"}
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p


def load_codec_and_test(root=".", n_sentences=None):
    data = Path(root) / "data"
    train, test = load_split("train", data), load_split("test", data)
    codec = HuffmanCodec.from_corpus(train)                     # training data only
    codec.save(data / "processed" / "huffman_codebook.json")
    return codec, (test if n_sentences is None else test[:n_sentences])


# ============================================================================= (a) uncoded
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def theory_bpsk_awgn(ebn0_db):
    return 0.5 * erfc(np.sqrt(ch.db_to_lin(ebn0_db)))


def theory_bpsk_rayleigh(snr_db):
    g = ch.db_to_lin(snr_db)                       # average Eb/N0 (= Es/N0 for BPSK)
    return 0.5 * (1 - np.sqrt(g / (1 + g)))


def run_uncoded_validation(mode="quick", root=".", seed=SEED, verbose=True) -> pd.DataFrame:
    """Uncoded BPSK/QPSK over AWGN vs 0.5 erfc(sqrt(Eb/N0)), plus BPSK over (per-symbol)
    Rayleigh fading with ZF vs the closed form 0.5(1 - sqrt(g/(1+g)))."""
    cfg = CONFIG[mode]
    rows = []
    rng = np.random.default_rng(seed)
    cases = [("BPSK", "awgn"), ("QPSK", "awgn"), ("BPSK", "rayleigh")]
    for mod, chan in cases:
        modem = Modem(mod)
        for eb in tqdm(cfg["uncoded_ebn0"], desc=f"uncoded {mod}/{chan}", disable=not verbose):
            es = ch.ebn0_to_esn0_db(eb, modem.bps, 1.0)         # uncoded: R = 1
            errs = bits = 0
            while bits < cfg["uncoded_max_bits"] and errs < cfg["uncoded_min_errors"]:
                nb = 1_000_000 // modem.bps * modem.bps
                tx = rng.integers(0, 2, nb)
                x = modem.modulate(tx)
                if chan == "awgn":
                    y, n0 = ch.awgn(x, es, rng)
                    xh = y
                else:
                    ids = np.arange(len(x))                      # fully independent fading per symbol
                    y, h, n0 = ch.rayleigh_block_fading(x, es, ids, len(x), rng)
                    xh, _ = zf_equalize(y, h)
                rx = modem.demodulate_hard(xh)
                errs += int(np.count_nonzero(rx != tx))
                bits += nb
            th = theory_bpsk_awgn(eb) if chan == "awgn" else theory_bpsk_rayleigh(eb)
            lo, hi = wilson(errs, bits)
            rows.append(dict(modulation=mod, channel=chan, ebn0_db=eb, esn0_db=es, n_bits=bits,
                             n_errors=errs, ber_sim=errs / bits, ci_lo=lo, ci_hi=hi, ber_theory=float(th)))
    df = pd.DataFrame(rows)
    df["inside_ci_or_5pct"] = ((df.ber_theory >= df.ci_lo) & (df.ber_theory <= df.ci_hi)) | \
                              (np.abs(df.ber_sim / df.ber_theory - 1) < 0.05)
    df.to_csv(paths(root)["tab"] / f"{mode}_uncoded_ber.csv", index=False)
    return df


# ============================================================================= (b)+(c) sweep
def _point_rng(seed, chan, mcs, snr):
    return np.random.default_rng(np.random.SeedSequence([seed, 0 if chan == "awgn" else 1, mcs,
                                                        int(round(snr * 10)) + 1000]))


def run_mcs_sweep(mode="quick", root=".", seed=SEED, channels=("awgn", "rayleigh"),
                  text_quality_metrics=True, resume=True, verbose=True) -> pd.DataFrame:
    """Coded BER, sentence BLER and text metrics for all MCS x SNR x channel.
    Each point sends the same test sentences ``reps`` times with fresh noise/fading."""
    cfg = CONFIG[mode]
    P = paths(root)
    out = P["tab"] / f"{mode}_mcs_sweep.csv"
    codec, test = load_codec_and_test(root, cfg["n_sentences"])
    frames = Frames.build(test, codec)
    emb = SentenceEmbedder() if text_quality_metrics else None
    done = set()
    rows = []
    if resume and out.exists():
        old = pd.read_csv(out)
        rows = old.to_dict("records")
        done = {(r["channel"], int(r["mcs"]), round(float(r["snr_db"]), 3)) for r in rows}
    jobs = [(c, m, s) for c in channels for m in MCS_TABLE
            for s in (cfg["awgn_snr"] if c == "awgn" else cfg["rayleigh_snr"])
            if (c, m.index, round(float(s), 3)) not in done]
    t0 = time.time()
    (P["log"] / f"{mode}_config.json").write_text(json.dumps(
        dict(mode=mode, seed=seed, embedder=emb.backend if emb else None, n_test_sentences=len(test),
             **{k: (v if not isinstance(v, list) else [float(x) for x in v]) for k, v in cfg.items()}), indent=2))
    for chan, m, snr in tqdm(jobs, desc=f"{mode} sweep", disable=not verbose):
        link = ConventionalLink(codec, m, chan)
        rng = _point_rng(seed, chan, m.index, snr)
        reps = cfg["reps_awgn"] if chan == "awgn" else cfg["reps_rayleigh"]
        be = nb = fe = nf = 0
        raw_e = raw_n = 0
        tq = {"bleu": [], "word_acc": [], "cos": []}
        for _ in range(reps):
            res, hyp = link.run(frames, snr, rng)
            be += int(res.bit_errors.sum())
            nb += int(frames.n_info.sum())
            raw_e += int(res.raw_bit_errors.sum())
            raw_n += int(res.raw_bits.sum())
            fe += int(sum(a != b for a, b in zip(frames.sentences, hyp)))
            nf += len(hyp)
            if text_quality_metrics:
                q = text_quality(frames.sentences, hyp, emb)
                for k in tq:
                    tq[k].append(q[k])
        lo, hi = wilson(fe, nf)
        row = dict(channel=chan, mcs=m.index, modulation=m.modulation, rate=m.rate,
                   efficiency=m.efficiency, snr_db=float(snr),
                   ebn0_db=float(ch.esn0_to_ebn0_db(snr, m.bps, float(m.code_rate))),
                   n_frames=nf, n_info_bits=nb, bit_errors=be, coded_ber=be / nb,
                   raw_ber=raw_e / raw_n, n_block_errors=fe, bler=fe / nf, bler_ci_lo=lo, bler_ci_hi=hi)
        if text_quality_metrics:
            for k in tq:
                row[k] = float(np.mean(np.concatenate(tq[k])))
        rows.append(row)
        if len(rows) % 10 == 0:                                   # checkpoint
            pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows).sort_values(["channel", "mcs", "snr_db"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    (P["log"] / f"{mode}_sweep_runtime.txt").write_text(f"last run: {len(jobs)} new points, {time.time()-t0:.1f}s\n")
    return df


# ============================================================================= (d) AMC
def run_amc(mode="quick", root=".", seed=SEED):
    P = paths(root)
    sweep = pd.read_csv(P["tab"] / f"{mode}_mcs_sweep.csv")
    out = {}
    for chan in sweep["channel"].unique():
        lut = build_lut(sweep, chan)
        lut.to_csv(P["tab"] / f"{mode}_amc_lut_{chan}.csv", index=False)
        amc = AMC(lut, BLER_TARGET)
        grid = np.sort(lut["snr_db"].unique())
        curve = amc.curve(grid)
        curve.to_csv(P["tab"] / f"{mode}_amc_curve_{chan}.csv", index=False)
        out[chan] = (lut, curve)
    if "awgn" in out:                                  # estimation robustness (AWGN LUT)
        amc = AMC(out["awgn"][0], BLER_TARGET)
        grid = np.sort(out["awgn"][0]["snr_db"].unique())
        est = pd.concat([amc_with_estimation(amc, grid, n, 200 if mode == "quick" else 1000, seed)
                         for n in (16, 128)], ignore_index=True)
        est.to_csv(P["tab"] / f"{mode}_amc_snr_estimation.csv", index=False)
        out["estimation"] = est
    return out


def run_all(mode="quick", root=".", figures=True):
    from . import plotting
    log = {}
    log["uncoded"] = run_uncoded_validation(mode, root)
    log["sweep"] = run_mcs_sweep(mode, root)
    log["amc"] = run_amc(mode, root)
    if figures:
        plotting.make_all_figures(mode, root)
    return log


def expected_sweep_points(mode: str) -> int:
    cfg = CONFIG[mode]
    return len(MCS_TABLE) * (len(cfg["awgn_snr"]) + len(cfg["rayleigh_snr"]))


def sweep_is_complete(mode: str, root=".") -> bool:
    f = Path(root) / "results/tables" / f"{mode}_mcs_sweep.csv"
    return f.exists() and len(pd.read_csv(f)) >= expected_sweep_points(mode)


def best_available_mode(root=".") -> str:
    """'full' once the full sweep has finished, otherwise 'quick' (clearly labelled in plots)."""
    return "full" if sweep_is_complete("full", root) else "quick"
