"""Max-log LLR demapper for Gray BPSK / QPSK / 16-QAM / 64-QAM.

LLR_k = ln P(b_k = 0 | x_hat) / P(b_k = 1 | x_hat)  ~  ( min_{s: b_k=1} |x_hat - s|^2
                                                     - min_{s: b_k=0} |x_hat - s|^2 ) / N0_eff
Positive LLR => bit 0 more likely. N0_eff is the *complex* noise variance seen at the
equaliser output (N0 for AWGN, N0/|h|^2 after ZF on a fading channel).
Because square Gray QAM is separable, the minimisation is done per axis, which is exact.
"""
from __future__ import annotations

import numpy as np

from .modulation import Modem


def llr_maxlog(x_hat, n0_eff, modem: Modem, clip: float = 1e3) -> np.ndarray:
    """Return flat LLR vector of length len(x_hat) * bps (bit order of ``Modem.modulate``)."""
    x_hat = np.asarray(x_hat)
    n0 = np.broadcast_to(np.asarray(n0_eff, dtype=float), x_hat.shape)
    n0 = np.maximum(n0, 1e-12)
    pam = modem.levels * modem.scale
    m = modem.m_axis
    pattern = np.arange(len(pam))
    axes = [x_hat.real] if modem.is_real else [x_hat.real, x_hat.imag]
    cols = []
    for ax in axes:
        d2 = (ax[:, None] - pam[None, :]) ** 2                          # (N, 2^m)
        for b in range(m - 1, -1, -1):                                   # MSB first
            bit = (pattern >> b) & 1
            d0 = d2[:, bit == 0].min(axis=1)
            d1 = d2[:, bit == 1].min(axis=1)
            cols.append((d1 - d0) / n0)
    llr = np.stack(cols, axis=1).reshape(-1)
    return np.clip(llr, -clip, clip)
