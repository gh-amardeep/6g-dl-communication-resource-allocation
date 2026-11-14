"""Gray-mapped BPSK / QPSK / 16-QAM / 64-QAM with unit average symbol energy (Es = 1).

Square QAM is the product of two Gray-coded PAM axes. The first m = bps/2 bits of a
symbol select the in-phase level, the next m bits the quadrature level (as in
IEEE 802.11). For an axis with 2^m levels the bit pattern g (MSB first) maps to level
index k with g = k XOR (k >> 1):
    m=1 : 0 -> -1, 1 -> +1
    m=2 : 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
    m=3 : 000 -> -7, 001 -> -5, 011 -> -3, 010 -> -1, 110 -> +1, 111 -> +3, 101 -> +5, 100 -> +7
Average energy of the un-normalised grid is 2(M-1)/3, so the scale is sqrt(3/(2(M-1))).
"""
from __future__ import annotations

import numpy as np

BITS_PER_SYMBOL = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6}


def gray_pam_levels(m: int) -> np.ndarray:
    """levels[g] = (odd-integer) amplitude of the Gray pattern g (m bits, MSB first)."""
    M = 1 << m
    levels = np.empty(M)
    for k in range(M):
        levels[k ^ (k >> 1)] = 2 * k - (M - 1)
    return levels


class Modem:
    def __init__(self, name: str):
        name = name.upper().replace("-", "")
        if name not in BITS_PER_SYMBOL:
            raise ValueError(f"unknown modulation {name}")
        self.name = name
        self.bps = BITS_PER_SYMBOL[name]
        self.is_real = name == "BPSK"
        self.m_axis = 1 if self.is_real else self.bps // 2
        self.M = 1 << self.bps
        self.levels = gray_pam_levels(self.m_axis)            # unnormalised
        if self.is_real:
            self.scale = 1.0
        else:
            self.scale = np.sqrt(3.0 / (2.0 * (self.M - 1)))   # Es = 1
        self._w = 1 << np.arange(self.m_axis - 1, -1, -1)       # MSB-first weights

    # ----------------------------------------------------------------- mapping
    def modulate(self, bits, pad: bool = True) -> np.ndarray:
        """Map bits -> symbols. An incomplete last group is zero-padded (``pad=True``)
        or rejected. The caller keeps the original bit count to strip the padding."""
        bits = np.asarray(bits, dtype=np.int64)
        rem = len(bits) % self.bps
        if rem:
            if not pad:
                raise ValueError("bit count is not a multiple of bits/symbol")
            bits = np.concatenate([bits, np.zeros(self.bps - rem, dtype=np.int64)])
        g = bits.reshape(-1, self.bps)
        if self.is_real:
            return self.levels[g[:, 0]].astype(complex)
        i_idx = g[:, : self.m_axis] @ self._w
        q_idx = g[:, self.m_axis:] @ self._w
        return (self.levels[i_idx] + 1j * self.levels[q_idx]) * self.scale

    def demodulate_hard(self, symbols) -> np.ndarray:
        """Minimum-distance hard decision -> bits (used for noiseless verification)."""
        s = np.asarray(symbols)
        cols = []
        axes = [s.real] if self.is_real else [s.real, s.imag]
        pam = self.levels * self.scale
        for ax in axes:
            idx = np.argmin(np.abs(ax[:, None] - pam[None, :]), axis=1)   # idx == Gray pattern g
            for b in range(self.m_axis - 1, -1, -1):
                cols.append((idx >> b) & 1)
        return np.stack(cols, axis=1).reshape(-1).astype(np.uint8)

    # ----------------------------------------------------------------- helpers
    def constellation(self):
        """Return (points[M], bit_table[M, bps]); row n corresponds to integer n (MSB first)."""
        n = np.arange(self.M)
        table = ((n[:, None] >> np.arange(self.bps - 1, -1, -1)) & 1).astype(np.uint8)
        return self.modulate(table.reshape(-1)), table

    def average_energy(self) -> float:
        pts, _ = self.constellation()
        return float(np.mean(np.abs(pts) ** 2))


def get_modem(name: str) -> Modem:
    return Modem(name)
