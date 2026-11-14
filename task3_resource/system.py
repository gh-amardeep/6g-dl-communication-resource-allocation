"""System parameters and channel-gain generation for Task 3.

Units: powers in WATT, bandwidth in Hz, rates in bit/s. ``a = g / (N0 B F)`` [1/W] is the
noise-normalised gain, so SNR = p * a and rate = B log2(1 + p a).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from task1_conventional.channel import rayleigh_gains          # reused Task 1 Rayleigh h ~ CN(0,1)


@dataclass(frozen=True)
class Params:
    U: int = 8                   # users
    N: int = 16                  # resource blocks
    B: float = 180e3             # bandwidth per RB [Hz]
    radius_m: float = 500.0
    d_min_m: float = 35.0
    pl_const_db: float = 128.1   # PL = 128.1 + 37.6 log10(d/km)
    pl_slope: float = 37.6
    shadow_db: float = 8.0       # log-normal shadowing std
    n0_dbm_hz: float = -174.0
    nf_db: float = 9.0
    r_min: float = 1e6           # minimum user rate [bit/s]

    @property
    def noise_w(self) -> float:
        """N0 * B * F in watt."""
        return 10 ** ((self.n0_dbm_hz + 10 * np.log10(self.B) + self.nf_db - 30) / 10)


P = Params()
POWER_DBM_GRID = np.arange(15, 41, 1.0)          # 15 ... 40 dBm


def dbm_to_w(p_dbm):
    return 10 ** ((np.asarray(p_dbm, dtype=float) - 30) / 10)


def draw_channels(n_drops: int, rng: np.random.Generator, p: Params = P) -> np.ndarray:
    """Noise-normalised gain tensor ``a`` of shape (n_drops, U, N).

    One drop = users uniform in the cell (distance in [35, 500] m), one log-normal shadowing
    value per user, and Rayleigh small-scale fading independent per user and RB.
    """
    d_km = np.sqrt(rng.uniform(p.d_min_m ** 2, p.radius_m ** 2, size=(n_drops, p.U, 1))) / 1e3
    pl_db = p.pl_const_db + p.pl_slope * np.log10(d_km)
    sh_db = p.shadow_db * rng.standard_normal((n_drops, p.U, 1))
    large = 10 ** (-(pl_db + sh_db) / 10)
    h = rayleigh_gains(n_drops * p.U * p.N, rng).reshape(n_drops, p.U, p.N)
    return large * np.abs(h) ** 2 / p.noise_w


def rb_rates(power_w: np.ndarray, a: np.ndarray, p: Params = P) -> np.ndarray:
    """Rate [bit/s] of every (user, RB) pair if that user used ``power_w`` (shape N or (U,N))."""
    return p.B * np.log2(1 + power_w * a)


def user_rates(assign: np.ndarray, power_w: np.ndarray, a: np.ndarray, p: Params = P) -> np.ndarray:
    """R_u = sum_n x_{u,n} B log2(1 + p_n a_{u,n}); ``assign`` is (U,N) binary, ``power_w`` (N,)."""
    return (assign * rb_rates(power_w[None, :], a, p)).sum(axis=1)
