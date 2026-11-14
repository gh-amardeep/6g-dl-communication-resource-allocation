"""Adaptive modulation and coding (AMC) driven by simulated look-up tables.

Rule: among the six MCSs choose the one with the highest nominal spectral efficiency
eta = bits/symbol x code rate whose sentence BLER (at the reported SNR) is <= 10 %.
If no MCS qualifies, fall back to the most robust one (MCS 1, QPSK 1/2) and flag an outage.

The table is read at the largest grid SNR that does not exceed the input SNR
(conservative, because BLER decreases with SNR, so a grid point below the true SNR can
only over-estimate the BLER). Nothing here is hard coded: the LUT is produced from the
Monte-Carlo sweep (``experiments.run_mcs_sweep``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .channel import db_to_lin
from .communication_system import MCS_TABLE

BLER_TARGET = 0.10
LUT_COLUMNS = ["snr_db", "mcs", "modulation", "rate", "efficiency", "ber", "bler",
               "bleu", "word_acc", "cos_sim"]


def build_lut(sweep: pd.DataFrame, channel: str) -> pd.DataFrame:
    d = sweep[sweep["channel"] == channel].copy()
    d = d.rename(columns={"cos": "cos_sim", "coded_ber": "ber"})
    return d[LUT_COLUMNS].sort_values(["snr_db", "mcs"]).reset_index(drop=True)


class AMC:
    def __init__(self, lut: pd.DataFrame, bler_target: float = BLER_TARGET):
        self.lut, self.target = lut, bler_target
        self.snr_grid = np.sort(lut["snr_db"].unique())
        self.by_snr = {s: g.set_index("mcs") for s, g in lut.groupby("snr_db")}
        self.fallback = min(m.index for m in MCS_TABLE)

    def _grid_point(self, snr_db: float):
        k = np.searchsorted(self.snr_grid, snr_db + 1e-9, side="right") - 1
        return None if k < 0 else float(self.snr_grid[k])

    def select(self, snr_db: float) -> dict:
        """Return {'mcs', 'efficiency', 'bler', 'outage', 'grid_snr_db'} for an SNR [dB]."""
        g = self._grid_point(snr_db)
        if g is None:                                          # below the table: most robust MCS
            return {"mcs": self.fallback, "efficiency": float(MCS_TABLE[0].efficiency),
                    "bler": np.nan, "outage": True, "grid_snr_db": np.nan}
        row = self.by_snr[g]
        ok = row[row["bler"] <= self.target]
        if ok.empty:
            m = self.fallback
            return {"mcs": m, "efficiency": float(row.loc[m, "efficiency"]),
                    "bler": float(row.loc[m, "bler"]), "outage": True, "grid_snr_db": g}
        m = int(ok["efficiency"].idxmax())
        return {"mcs": m, "efficiency": float(ok.loc[m, "efficiency"]),
                "bler": float(ok.loc[m, "bler"]), "outage": False, "grid_snr_db": g}

    def curve(self, snr_db_values) -> pd.DataFrame:
        rows = []
        for s in snr_db_values:
            r = self.select(float(s))
            r["snr_db"] = float(s)
            r["goodput"] = r["efficiency"] * (1 - r["bler"]) if not np.isnan(r["bler"]) else np.nan
            rows.append(r)
        return pd.DataFrame(rows)

    def bler_at(self, snr_db: float, mcs: int) -> float:
        g = self._grid_point(snr_db)
        return np.nan if g is None else float(self.by_snr[g].loc[mcs, "bler"])


def estimate_snr_db(true_snr_db: float, n_pilots: int, rng: np.random.Generator) -> float:
    """Data-aided ML SNR estimate on an AWGN link from ``n_pilots`` known unit-energy QPSK
    pilots: SNR_hat = Es / mean|y - x|^2 (the unbiased-variance estimate of the noise)."""
    n0 = 1.0 / float(db_to_lin(true_snr_db))
    noise = np.sqrt(n0 / 2) * (rng.standard_normal(n_pilots) + 1j * rng.standard_normal(n_pilots))
    n0_hat = np.mean(np.abs(noise) ** 2)
    return float(10 * np.log10(1.0 / n0_hat))


def amc_with_estimation(amc: AMC, snr_grid, n_pilots: int, n_trials: int, seed: int = 0) -> pd.DataFrame:
    """Robustness check: select with an *estimated* SNR, score with the LUT at the *true* SNR."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in snr_grid:
        eff, blr, viol = [], [], []
        for _ in range(n_trials):
            m = amc.select(estimate_snr_db(s, n_pilots, rng))["mcs"]
            b = amc.bler_at(s, m)
            e = [x for x in MCS_TABLE if x.index == m][0].efficiency
            eff.append(e)
            blr.append(b)
            viol.append(b > amc.target)
        rows.append({"snr_db": float(s), "n_pilots": n_pilots, "mean_efficiency": float(np.mean(eff)),
                     "mean_selected_bler": float(np.nanmean(blr)), "p_bler_violation": float(np.mean(viol))})
    return pd.DataFrame(rows)
