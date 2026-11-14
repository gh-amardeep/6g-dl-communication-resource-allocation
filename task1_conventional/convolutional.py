"""Rate-1/2, K=7 convolutional code (generators 133, 171 octal) with puncturing.

Convention (IEEE 802.11 / DVB style)
------------------------------------
The shift register holds the current input bit and the six previous bits:
    reg = [b_t, b_{t-1}, ..., b_{t-6}]  ->  integer (b_t << 6) | state
    state = (b_{t-1} << 5) | ... | b_{t-6}       (6 bits -> 64 states)
Generator polynomial g (7 bits, MSB first) taps the register MSB-first, i.e. the MSB
of g multiplies the *current* input. Output bit  c = parity(reg & g).
    next_state = (b_t << 5) | (state >> 1)
Coded stream order: [c^(133)_0, c^(171)_0, c^(133)_1, c^(171)_1, ...].
Termination: K-1 = 6 zero tail bits drive the encoder back to state 0, so a frame
of L information bits yields 2*(L+6) mother-code bits.

``trellis()`` is the *single* definition shared by the encoder tests and the
Viterbi decoder, which guarantees the two are consistent.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

K = 7
N_TAIL = K - 1
GENERATORS_OCT = (0o133, 0o171)
N_STATES = 1 << (K - 1)

# Puncturing masks over the mother stream [A0 B0 A1 B1 ...] (1 = transmit); IEEE 802.11a
PUNCTURE_MASKS = {
    "1/2": np.array([1, 1], dtype=bool),
    "2/3": np.array([1, 1, 1, 0], dtype=bool),               # keep A0 B0 A1, drop B1
    "3/4": np.array([1, 1, 1, 0, 0, 1], dtype=bool),         # keep A0 B0 A1 B2, drop B1 A2
}


def _taps(g: int) -> np.ndarray:
    """Tap vector h[k] multiplying x[t-k]  (k = 0 is the current input = MSB of g)."""
    return np.array([(g >> (K - 1 - k)) & 1 for k in range(K)], dtype=np.int64)


def encode_mother(bits, terminate: bool = True) -> np.ndarray:
    """Vectorised rate-1/2 encoding (via convolution mod 2). Output length 2*(L+6)."""
    x = np.asarray(bits, dtype=np.int64)
    if terminate:
        x = np.concatenate([x, np.zeros(N_TAIL, dtype=np.int64)])
    n = len(x)
    out = np.empty(2 * n, dtype=np.uint8)
    for i, g in enumerate(GENERATORS_OCT):
        out[i::2] = (np.convolve(x, _taps(g))[:n] & 1).astype(np.uint8)
    return out


def encode_reference(bits, terminate: bool = True) -> np.ndarray:
    """Slow shift-register encoder used only to cross-check ``encode_mother``."""
    x = list(map(int, bits)) + ([0] * N_TAIL if terminate else [])
    state = 0
    out = []
    for b in x:
        reg = (b << (K - 1)) | state
        for g in GENERATORS_OCT:
            out.append(bin(reg & g).count("1") & 1)
        state = (b << (K - 2)) | (state >> 1)
    return np.array(out, dtype=np.uint8)


def trellis():
    """Return (next_state[64,2], out_bits[64,2,2]) indexed by (state, input bit)."""
    nxt = np.zeros((N_STATES, 2), dtype=np.int64)
    out = np.zeros((N_STATES, 2, 2), dtype=np.int64)
    for s in range(N_STATES):
        for b in range(2):
            reg = (b << (K - 1)) | s
            nxt[s, b] = (b << (K - 2)) | (s >> 1)
            for i, g in enumerate(GENERATORS_OCT):
                out[s, b, i] = bin(reg & g).count("1") & 1
    return nxt, out


# --------------------------------------------------------------------------- puncturing
def _keep_index(rate: str, n_mother: int) -> np.ndarray:
    mask = PUNCTURE_MASKS[rate]
    reps = -(-n_mother // len(mask))
    return np.flatnonzero(np.tile(mask, reps)[:n_mother])


def puncture(mother_bits, rate: str) -> np.ndarray:
    mother_bits = np.asarray(mother_bits)
    return mother_bits[_keep_index(rate, len(mother_bits))]


def depuncture(values, rate: str, n_mother: int, fill: float = 0.0) -> np.ndarray:
    """Re-insert punctured positions. For soft decoding ``fill=0`` means "no information"
    (LLR = 0), so the Viterbi branch metric ignores the erased bits."""
    idx = _keep_index(rate, n_mother)
    if len(values) != len(idx):
        raise ValueError(f"expected {len(idx)} received values for rate {rate}, got {len(values)}")
    out = np.full(n_mother, fill, dtype=np.result_type(np.asarray(values).dtype, np.float32))
    out[idx] = values
    return out


def punctured_length(n_info: int, rate: str) -> int:
    return len(_keep_index(rate, 2 * (n_info + N_TAIL)))


def nominal_rate(rate: str) -> Fraction:
    """Asymptotic code rate = (1/2) * (mask length / mask ones)."""
    m = PUNCTURE_MASKS[rate]
    return Fraction(1, 2) * Fraction(len(m), int(m.sum()))


def effective_rate(n_info: int, rate: str) -> float:
    """Actual rate of a finite frame, including the 6 termination bits."""
    return n_info / punctured_length(n_info, rate)
