"""Conventional baselines for the (7,4) experiments + a common Monte-Carlo evaluator.

Hamming(7,4), systematic: codeword c = [d1 d2 d3 d4 p1 p2 p3], BPSK 0 -> +1, 1 -> -1.
  * hard decision : sign() of every symbol, syndrome decoding (corrects 1 bit error);
  * ML decoding   : correlate the received vector with all 16 codewords (min Euclidean
                    distance), i.e. soft-decision maximum likelihood.
  * uncoded BPSK  : the 4 information bits are sent directly (Es = Eb).
All systems see the same information bits and the same Eb/N0, where Eb is the energy per
*information* bit (so coded systems spend Es = R*Eb = 4/7 Eb per transmitted symbol).
Task 1 functions are reused for the noise, the Rayleigh gains and the ZF equaliser.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

from task1_conventional.channel import (complex_gaussian, ebn0_to_esn0_db, noise_variance,
                                        rayleigh_gains)
from task1_conventional.equalizer import zf_equalize
from .autoencoder import M, int_to_bits

P = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
G = np.hstack([np.eye(4, dtype=np.uint8), P])            # 4 x 7
H = np.hstack([P.T, np.eye(3, dtype=np.uint8)])           # 3 x 7   (G H^T = 0)
MSG_BITS = int_to_bits(np.arange(M))                       # 16 x 4
CODEWORDS = (MSG_BITS @ G) % 2                             # 16 x 7
BPSK_CW = 1.0 - 2.0 * CODEWORDS                            # 16 x 7  (+-1)
RATE = 4 / 7


def hamming_encode(bits):
    return (np.asarray(bits) @ G) % 2


def hamming_hard_decode(r):
    """r: (N,7) real soft values -> (N,4) information bits (syndrome decoding)."""
    c = (np.asarray(r) < 0).astype(np.uint8)
    syn = (c @ H.T) % 2
    key = syn[:, 0] * 4 + syn[:, 1] * 2 + syn[:, 2]
    lut = np.full(8, -1)
    for j in range(7):
        lut[int(H[0, j]) * 4 + int(H[1, j]) * 2 + int(H[2, j])] = j
    pos = lut[key]
    fix = pos >= 0
    c[np.where(fix)[0], pos[fix]] ^= 1
    return c[:, :4]


def hamming_ml_decode(r):
    """Soft-decision ML (min Euclidean distance) -> (N,4) bits."""
    return MSG_BITS[np.argmax(np.asarray(r) @ BPSK_CW.T, axis=1)]


# ---------------------------------------------------------------------- theory
def q_bpsk_awgn(ebn0_db):
    return 0.5 * erfc(np.sqrt(10 ** (np.asarray(ebn0_db) / 10)))


def bpsk_rayleigh(ebn0_db):
    g = 10 ** (np.asarray(ebn0_db) / 10)
    return 0.5 * (1 - np.sqrt(g / (1 + g)))


# ---------------------------------------------------------------------- Monte Carlo
def _bits_errors(bits_hat, bits):
    e = bits_hat != bits
    return int(e.sum()), int(e.any(axis=1).sum())


def simulate_point(ebn0_db, n_blocks, rng, channel="awgn", ae=None, chunk=500_000):
    """One Eb/N0 point. Returns {method: (bit_errors, block_errors)} and n_blocks.
    ``ae``: trained (7,4) Autoencoder (or None)."""
    import torch
    res = {k: [0, 0] for k in ["uncoded_bpsk", "hamming_hard", "hamming_ml", "autoencoder"]}
    n0_c = noise_variance(float(ebn0_to_esn0_db(ebn0_db, 1, RATE)))   # coded symbols
    n0_u = noise_variance(float(ebn0_db))                              # uncoded BPSK
    done = 0
    if ae is not None:
        cb = ae.codebook().detach().numpy()
    while done < n_blocks:
        n = min(chunk, n_blocks - done)
        msg = rng.integers(0, M, n)
        bits = MSG_BITS[msg]
        x_ham = BPSK_CW[msg]
        x_unc = 1.0 - 2.0 * bits
        nz_c = complex_gaussian((n, 7), n0_c, rng)
        nz_u = complex_gaussian((n, 4), n0_u, rng)
        if channel == "awgn":
            r_ham, r_unc = (x_ham + nz_c).real, (x_unc + nz_u).real
            gain = None
            r_ae = (cb[msg] + nz_c).real if ae is not None else None
        else:
            h = rayleigh_gains(n, rng)[:, None]                       # one h per block
            gain = (np.abs(np.where(np.abs(h) < 1e-3, 1e-3, h)) ** 2)  # |h|^2 as used by ZF
            eq = lambda x, nz: zf_equalize(h * x + nz, h)[0].real       # y / h, real part
            r_ham, r_unc = eq(x_ham, nz_c), eq(x_unc, nz_u)
            r_ae = eq(cb[msg], nz_c) if ae is not None else None
        outs = {"uncoded_bpsk": (r_unc < 0).astype(np.uint8),
                "hamming_hard": hamming_hard_decode(r_ham),
                "hamming_ml": hamming_ml_decode(r_ham)}
        if ae is not None:
            t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
            m_hat = ae.predict(t(r_ae), None if gain is None else t(gain)).numpy()
            outs["autoencoder"] = MSG_BITS[m_hat]
        for k, bh in outs.items():
            be, ble = _bits_errors(bh, bits)
            res[k][0] += be; res[k][1] += ble
        done += n
    return res, n_blocks
