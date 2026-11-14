"""End-to-end conventional link:  text -> Huffman -> conv. enc. -> puncture -> QAM ->
channel -> ZF -> max-log LLR -> depuncture -> soft Viterbi -> Huffman -> text.

Frame = one sentence. Assumptions (documented in README):
  * the receiver knows each frame's information-bit count (frame length side information
    sent error-free on a control channel) - needed for trellis termination/depuncturing;
  * no bit interleaver (not in the specified chain); the Rayleigh channel is block fading:
    one independent h ~ CN(0,1) per resource block of ``rb_symbols`` symbols;
  * perfect CSI at the receiver (h known to the ZF equaliser / demapper).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from . import channel as ch
from . import convolutional as cv
from .demapper import llr_maxlog
from .equalizer import zf_equalize
from .huffman import HuffmanCodec
from .modulation import Modem
from .viterbi import viterbi_decode

RB_SYMBOLS = 84           # 12 subcarriers x 7 OFDM symbols (one LTE/NR-style resource block)


@dataclass(frozen=True)
class MCS:
    index: int
    modulation: str
    rate: str                    # "1/2", "2/3", "3/4"

    @property
    def name(self) -> str:
        return f"MCS{self.index}: {self.modulation} {self.rate}"

    @property
    def bps(self) -> int:
        return Modem(self.modulation).bps

    @property
    def code_rate(self) -> Fraction:
        return cv.nominal_rate(self.rate)

    @property
    def efficiency(self) -> float:
        """Nominal spectral efficiency [information bit / channel use] = bps * R."""
        return float(self.bps * self.code_rate)


MCS_TABLE = [MCS(1, "QPSK", "1/2"), MCS(2, "QPSK", "3/4"), MCS(3, "16QAM", "1/2"),
             MCS(4, "16QAM", "3/4"), MCS(5, "64QAM", "2/3"), MCS(6, "64QAM", "3/4")]


@dataclass
class Frames:
    """Sentences + their Huffman bits (computed once, reused over SNR/MCS/channel)."""
    sentences: list
    info: list
    n_info: np.ndarray
    info_pad: np.ndarray                    # (B, Lmax) zero padded
    ref_text: list = field(default_factory=list)   # codec.decode(codec.encode(s)) (== s when no UNK chars)
    _tx: dict = field(default_factory=dict)  # rate -> (list of punctured bit arrays)
    _chunks: dict = field(default_factory=dict)

    @classmethod
    def build(cls, sentences, codec: HuffmanCodec) -> "Frames":
        info = [codec.encode(s) for s in sentences]
        n = np.array([len(b) for b in info])
        pad = np.zeros((len(info), n.max()), dtype=np.uint8)
        for i, b in enumerate(info):
            pad[i, :len(b)] = b
        return cls(list(sentences), info, n, pad, [codec.decode(b) for b in info])

    def tx_bits(self, rate: str):
        if rate not in self._tx:
            self._tx[rate] = [cv.puncture(cv.encode_mother(b), rate) for b in self.info]
        return self._tx[rate]

    def subset(self, idx) -> "Frames":
        idx = np.asarray(idx)
        return Frames([self.sentences[i] for i in idx], [self.info[i] for i in idx],
                      self.n_info[idx], self.info_pad[idx][:, : self.n_info[idx].max()].copy(),
                      [self.ref_text[i] for i in idx])

    def chunks(self, size: int):
        if size not in self._chunks:
            self._chunks[size] = [self.subset(np.arange(i, min(i + size, len(self.sentences))))
                                  for i in range(0, len(self.sentences), size)]
        return self._chunks[size]


@dataclass
class LinkResult:
    decoded: np.ndarray            # (B, Lmax) decoded info bits
    bit_errors: np.ndarray         # (B,) info-bit errors per frame (after decoding)
    frame_error: np.ndarray        # (B,) bool, info bits not all correct
    raw_bit_errors: np.ndarray     # (B,) channel (pre-decoder) hard-decision errors on coded bits
    raw_bits: np.ndarray           # (B,) number of transmitted coded bits

    @property
    def ber(self) -> float:
        return float(self.bit_errors.sum() / self.n_info_total)

    n_info_total: int = 0


class ConventionalLink:
    def __init__(self, codec: HuffmanCodec, mcs: MCS, channel: str = "awgn",
                 rb_symbols: int = RB_SYMBOLS):
        if channel not in ("awgn", "rayleigh"):
            raise ValueError("channel must be 'awgn' or 'rayleigh'")
        self.codec, self.mcs, self.channel, self.rb = codec, mcs, channel, rb_symbols
        self.modem = Modem(mcs.modulation)

    # ----------------------------------------------------------------- physical layer
    def transmit_receive(self, frames: Frames, snr_db: float | None, rng: np.random.Generator,
                         chunk: int = 1000) -> LinkResult:
        """snr_db = Es/N0 in dB; ``None`` means a noiseless channel (h = 1 even for Rayleigh)."""
        if len(frames.sentences) > chunk:
            parts = [self.transmit_receive(c, snr_db, rng, chunk) for c in frames.chunks(chunk)]
            width = frames.info_pad.shape[1]
            dec = np.zeros((len(frames.sentences), width), dtype=np.uint8)
            o = 0
            for p in parts:
                dec[o:o + len(p.decoded), :p.decoded.shape[1]] = p.decoded
                o += len(p.decoded)
            return LinkResult(dec, np.concatenate([p.bit_errors for p in parts]),
                              np.concatenate([p.frame_error for p in parts]),
                              np.concatenate([p.raw_bit_errors for p in parts]),
                              np.concatenate([p.raw_bits for p in parts]),
                              n_info_total=int(frames.n_info.sum()))
        modem, rate, bps = self.modem, self.mcs.rate, self.modem.bps
        tx_list = frames.tx_bits(rate)
        n_tx = np.array([len(t) for t in tx_list])
        pad = (-n_tx) % bps
        flat = np.concatenate([np.concatenate([t, np.zeros(p, dtype=t.dtype)]) for t, p in zip(tx_list, pad)])
        x = modem.modulate(flat, pad=False)
        n_sym = (n_tx + pad) // bps
        # --- channel
        if snr_db is None:
            y, h, n0 = x, np.ones_like(x), 0.0
        elif self.channel == "awgn":
            y, n0 = ch.awgn(x, snr_db, rng)
            h = np.ones_like(x)
        else:
            ids, n_blk = ch.block_ids(n_sym, self.rb)
            y, h, n0 = ch.rayleigh_block_fading(x, snr_db, ids, n_blk, rng)
        # --- receiver front end
        x_hat, gain = zf_equalize(y, h)
        llr = llr_maxlog(x_hat, n0 / gain, modem)
        bit_off = np.concatenate([[0], np.cumsum(n_tx + pad)[:-1]])
        n_mother = 2 * (frames.n_info + cv.N_TAIL)
        Tmax = int(n_mother.max() // 2)
        mother = np.zeros((len(tx_list), 2 * Tmax), dtype=np.float32)
        raw_err = np.zeros(len(tx_list), dtype=np.int64)
        for i in range(len(tx_list)):
            l = llr[bit_off[i]: bit_off[i] + n_tx[i]]
            raw_err[i] = np.count_nonzero((l < 0) != (tx_list[i] == 1))
            mother[i, : n_mother[i]] = cv.depuncture(l, rate, int(n_mother[i]))
        dec = viterbi_decode(mother, frames.n_info)
        L = frames.info_pad.shape[1]
        dec = dec[:, :L]
        mask = np.arange(L)[None, :] < frames.n_info[:, None]
        bit_err = ((dec != frames.info_pad) & mask).sum(axis=1)
        return LinkResult(dec, bit_err, bit_err > 0, raw_err, n_tx,
                          n_info_total=int(frames.n_info.sum()))

    # ----------------------------------------------------------------- text level
    def reconstruct(self, frames: Frames, result: LinkResult) -> list:
        """Huffman-decode. Error-free frames are copied (decoding is lossless, verified in tests)."""
        out = []
        for i, s in enumerate(frames.sentences):
            if not result.frame_error[i]:
                out.append(frames.ref_text[i])
            else:
                out.append(self.codec.decode(result.decoded[i, : frames.n_info[i]]))
        return out

    def run(self, frames: Frames, snr_db, rng):
        res = self.transmit_receive(frames, snr_db, rng)
        return res, self.reconstruct(frames, res)
