"""Batched soft-decision Viterbi decoder for the K=7 (133,171) code.

LLR convention: llr = ln P(c=0)/P(c=1)  (positive => bit 0).
Path metric (maximised): sum over coded bits of  llr * (1 - 2c)/2 ,
which is the log-likelihood up to a constant. Punctured (erased) bits carry
llr = 0 and therefore contribute nothing.

Frames of different length are decoded together: each frame's LLR row is zero
padded; at its own trellis end (T_i = L_i + 6) every state except state 0 is
forced to -inf (zero termination), after which the padded zero-LLR steps are
neutral and traceback necessarily passes through state 0 at T_i.
"""
from __future__ import annotations

import numpy as np

from .convolutional import N_STATES, N_TAIL, trellis

_NEG = np.float32(-1e9)


def _predecessor_tables():
    """For every next state: its two (previous state, input bit, coded-bit pair index).
    Derived from ``trellis()`` so the decoder cannot drift from the encoder."""
    nxt, out = trellis()
    prev = np.zeros((N_STATES, 2), dtype=np.int64)
    inbit = np.zeros(N_STATES, dtype=np.int64)
    cidx = np.zeros((N_STATES, 2), dtype=np.int64)      # 2*c_A + c_B
    filled = np.zeros(N_STATES, dtype=np.int64)
    for s in range(N_STATES):
        for b in range(2):
            ns = nxt[s, b]
            k = filled[ns]
            prev[ns, k] = s
            inbit[ns] = b
            cidx[ns, k] = 2 * out[s, b, 0] + out[s, b, 1]
            filled[ns] += 1
    assert (filled == 2).all()
    return prev, inbit, cidx


_PREV, _INBIT, _CIDX = _predecessor_tables()
# which of the two predecessor slots (0/1) was taken is what we store as the decision


def viterbi_decode(llr, n_info, chunk: int = 1000) -> np.ndarray:
    """Decode a batch.

    llr    : (B, 2*Tmax) float array of depunctured mother-code LLRs (zero padded)
    n_info : (B,) number of information bits of every frame (T_i = n_info + 6)
    returns: (B, Tmax - 6) uint8 decoded information bits (valid up to n_info[i]).
    """
    llr = np.asarray(llr, dtype=np.float32)
    n_info = np.asarray(n_info)
    out = [_decode_chunk(llr[i:i + chunk], n_info[i:i + chunk]) for i in range(0, len(llr), chunk)]
    return np.concatenate(out, axis=0) if len(out) > 1 else out[0]


def _decode_chunk(llr, n_info):
    B, two_t = llr.shape
    T = two_t // 2
    T_end = n_info + N_TAIL
    pm = np.full((B, N_STATES), _NEG, dtype=np.float32)
    pm[:, 0] = 0.0                                            # start in state 0
    dec = np.empty((T, B, N_STATES), dtype=np.uint8)
    ends: dict[int, np.ndarray] = {}
    for i, t in enumerate(T_end):
        ends.setdefault(int(t), []).append(i)
    ends = {t: np.array(v) for t, v in ends.items()}
    for t in range(T):
        a = llr[:, 2 * t] * 0.5
        b = llr[:, 2 * t + 1] * 0.5
        bm4 = np.stack([a + b, a - b, -a + b, -a - b], axis=1)   # index = 2*cA + cB
        cand = pm[:, _PREV] + bm4[:, _CIDX]                      # (B, 64, 2)
        sel = cand[:, :, 1] > cand[:, :, 0]
        dec[t] = sel
        pm = np.where(sel, cand[:, :, 1], cand[:, :, 0])
        if (t + 1) in ends:                                       # force termination state 0
            rows = ends[t + 1]
            keep = pm[rows, 0].copy()
            pm[rows] = _NEG
            pm[rows, 0] = keep
    state = np.argmax(pm, axis=1)
    bits = np.empty((T, B), dtype=np.uint8)
    ar = np.arange(B)
    for t in range(T - 1, -1, -1):
        bits[t] = _INBIT[state]
        slot = dec[t, ar, state]
        state = _PREV[state, slot]
    return bits[: T - N_TAIL].T.copy()


def viterbi_single(llr, n_info: int) -> np.ndarray:
    """Convenience wrapper for one frame (llr length must be 2*(n_info+6))."""
    llr = np.asarray(llr, dtype=np.float32)[None, :]
    return viterbi_decode(llr, np.array([n_info]))[0, :n_info]
