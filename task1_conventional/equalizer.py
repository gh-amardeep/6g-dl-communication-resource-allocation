"""Zero-forcing equalisation  y = h x + n  ->  x_hat = y / h  = x + n/h.

Numerics: when |h| < eps the coefficient is replaced by eps * h/|h| (same phase, magnitude
floor), so the division can never overflow. The returned ``gain = |h_used|^2`` tells the
demapper that the post-equalisation noise has variance N0 / gain.

Limitations (see notebook 06): ZF removes the channel amplitude/phase but amplifies noise
by 1/|h|^2 (noise enhancement in deep fades); for a Rayleigh channel E[1/|h|^2] diverges, so
ZF without coding/interleaving is dominated by the rare deep fades. With a single antenna
ZF is just channel inversion and is not MMSE-optimal at low SNR.
"""
from __future__ import annotations

import numpy as np

H_FLOOR = 1e-3


def zf_equalize(y, h, eps: float = H_FLOOR):
    y = np.asarray(y)
    h = np.asarray(h, dtype=complex)
    mag = np.abs(h)
    phase = np.where(mag > 0, h / np.where(mag > 0, mag, 1.0), 1.0)
    h_used = np.where(mag < eps, eps * phase, h)
    return y / h_used, np.abs(h_used) ** 2
