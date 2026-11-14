"""Channel models and SNR bookkeeping.

SNR definition used throughout (matches the assignment's gamma = p*g/(N0*B*F)):
    SNR = Es / N0   (average received symbol energy over *complex* noise variance N0)
With Es = 1 the complex noise sample is  n ~ CN(0, N0),  N0 = 1/SNR, i.e. N0/2 per real
dimension. Rayleigh fading h ~ CN(0, 1) has E|h|^2 = 1, so SNR is the *average* SNR; the
instantaneous SNR of a block is |h|^2 * SNR.

Conversions (bps = bits per modulation symbol, R = code rate = info bits / coded bit):
    Es/N0 = (Eb/N0) * bps * R          Eb/N0 [dB] = Es/N0 [dB] - 10 log10(bps * R)
Uncoded BPSK: bps*R = 1  -> Es/N0 = Eb/N0.      Uncoded QPSK: bps*R = 2.
"""
from __future__ import annotations

import numpy as np


def db_to_lin(x_db):
    return 10.0 ** (np.asarray(x_db, dtype=float) / 10.0)


def lin_to_db(x):
    return 10.0 * np.log10(np.asarray(x, dtype=float))


def esn0_to_ebn0_db(esn0_db, bps: int, rate: float = 1.0):
    return np.asarray(esn0_db, dtype=float) - 10.0 * np.log10(bps * rate)


def ebn0_to_esn0_db(ebn0_db, bps: int, rate: float = 1.0):
    return np.asarray(ebn0_db, dtype=float) + 10.0 * np.log10(bps * rate)


def noise_variance(snr_db: float, es: float = 1.0) -> float:
    """Complex noise variance N0 for a given Es/N0 [dB]."""
    return es / float(db_to_lin(snr_db))


def complex_gaussian(shape, var: float, rng: np.random.Generator) -> np.ndarray:
    """CN(0, var): real and imaginary parts each N(0, var/2)."""
    s = np.sqrt(var / 2.0)
    return s * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def awgn(x, snr_db: float, rng: np.random.Generator, es: float = 1.0):
    """y = x + n. Returns (y, N0)."""
    n0 = noise_variance(snr_db, es)
    x = np.asarray(x)
    return x + complex_gaussian(x.shape, n0, rng), n0


def rayleigh_gains(n_blocks: int, rng: np.random.Generator) -> np.ndarray:
    """Independent h ~ CN(0,1), one per resource block."""
    return complex_gaussian(n_blocks, 1.0, rng)


def block_ids(n_symbols_per_frame, block_len: int):
    """Global resource-block index of every symbol.

    Frames are concatenated; frame i owns ``n_symbols_per_frame[i]`` symbols which are
    cut into consecutive blocks of ``block_len`` symbols (one RB each). RBs are never
    shared between frames, and every RB has its own fading coefficient.
    Returns (ids[total_symbols], total_blocks)."""
    n_sym = np.asarray(n_symbols_per_frame, dtype=np.int64)
    n_blk = -(-n_sym // block_len)
    blk_off = np.concatenate([[0], np.cumsum(n_blk)[:-1]])
    sym_off = np.concatenate([[0], np.cumsum(n_sym)[:-1]])
    local = np.arange(n_sym.sum()) - np.repeat(sym_off, n_sym)
    return np.repeat(blk_off, n_sym) + local // block_len, int(n_blk.sum())


def rayleigh_block_fading(x, snr_db: float, ids, n_blocks: int, rng: np.random.Generator,
                          es: float = 1.0):
    """y = h x + n with h constant over each RB. Returns (y, h_per_symbol, N0)."""
    h_blk = rayleigh_gains(n_blocks, rng)
    h = h_blk[ids]
    n0 = noise_variance(snr_db, es)
    x = np.asarray(x)
    return h * x + complex_gaussian(x.shape, n0, rng), h, n0
