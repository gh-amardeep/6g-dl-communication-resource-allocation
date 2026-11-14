"""Correct sentences per second of an allocation, using the Task 1 AMC (unchanged) and its AWGN look-up table.

For every allocated RB n (user u, power p_n): SNR = p_n * a_un (Es/N0).  The RB gain is constant over the RB, so the
link is AWGN-equivalent at that SNR (Task 1: block fading + ZF) -> the AWGN LUT is used.  The Task 1 AMC rule picks the
highest-efficiency MCS with BLER <= 10 % (fallback MCS 1 + outage flag).  Assumptions: symbol rate of an RB = B (180 ksym/s);
information bits per sentence = mean Huffman-coded length of the Task 1 test sentences.
sentences/s of RB = B * eta(MCS) * (1 - BLER(SNR, MCS)) / bits_per_sentence;  BLER undefined (SNR below the table) -> 0 delivered.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from task1_conventional.amc import AMC
from task1_conventional.dataset import load_split
from task1_conventional.huffman import HuffmanCodec

from .system import P, Params


def load_amc(root=".") -> AMC:
    return AMC(pd.read_csv(Path(root) / "results/tables/full_amc_lut_awgn.csv"))


def mean_sentence_bits(root=".") -> float:
    """Mean information bits per test sentence (Huffman codebook of Task 1, read-only)."""
    codec = HuffmanCodec.load(Path(root) / "data/processed/huffman_codebook.json")
    test = load_split("test", Path(root) / "data")
    return float(np.mean([len(codec.encode(s)) for s in test]))


class AMCTable:
    """Vectorised look-up built by calling ``AMC.select`` once per LUT grid SNR (no new AMC rule)."""

    def __init__(self, amc: AMC):
        self.amc = amc
        g = amc.snr_grid
        sel = [amc.select(float(s)) for s in g]
        self.grid = g
        self.mcs = np.array([r["mcs"] for r in sel])
        self.eff = np.array([r["efficiency"] for r in sel])
        self.bler = np.array([r["bler"] for r in sel])
        self.outage = np.array([r["outage"] for r in sel])

    def lookup(self, snr_db: np.ndarray):
        """Same grid rule as ``AMC._grid_point``; SNR below the table -> (MCS 1, efficiency of MCS 1, BLER nan, outage)."""
        k = np.searchsorted(self.grid, snr_db + 1e-9, side="right") - 1
        below = k < 0
        k = np.clip(k, 0, None)
        mcs, eff, bler, out = self.mcs[k].copy(), self.eff[k].copy(), self.bler[k].copy(), self.outage[k].copy()
        mcs[below], bler[below], out[below] = 1, np.nan, True
        eff[below] = self.eff[0] if self.mcs[0] == 1 else 1.0
        return mcs, eff, bler, out


def sentences_per_second(assign, power, a, table: AMCTable, bits_per_sentence: float, p: Params = P):
    """Returns (total sentences/s, per-RB array, mcs per RB, outage per RB)."""
    owner = assign.argmax(axis=0)
    snr = power * a[owner, np.arange(a.shape[1])]
    with np.errstate(divide="ignore"):
        snr_db = np.where(snr > 0, 10 * np.log10(np.maximum(snr, 1e-300)), -np.inf)
    mcs, eff, bler, out = table.lookup(snr_db)
    ok = np.isfinite(snr_db) & ~np.isnan(bler)
    delivered = np.where(ok, p.B * eff * (1 - np.nan_to_num(bler, nan=1.0)) / bits_per_sentence, 0.0)
    return float(delivered.sum()), delivered, mcs, out
