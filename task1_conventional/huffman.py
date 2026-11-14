"""Character-level Huffman source coding.

The codebook is built from *training* text only. Characters never seen in training
are mapped to a reserved UNK symbol (which owns a code word, with frequency 1), so
encoding never fails; such a character is reconstructed as ``UNK_CHAR`` (lossy by
design, and reported by ``HuffmanCodec.unknown_count``).
"""
from __future__ import annotations

import heapq
import json
import math
from collections import Counter
from itertools import count
from pathlib import Path

import numpy as np

UNK_CHAR = "�"


class HuffmanCodec:
    def __init__(self, code: dict[str, str], freq: dict[str, int] | None = None):
        self.code = dict(code)                      # char -> '0101...'
        self.freq = dict(freq) if freq else {}
        self._code_bits = {c: np.frombuffer(s.encode(), dtype=np.uint8) - 48
                           for c, s in self.code.items()}
        # binary decoding tree stored in flat arrays: node -> (child0, child1), leaf char
        self._child = [[-1, -1]]
        self._leaf: list[str | None] = [None]
        for ch, s in self.code.items():
            node = 0
            for b in s:
                b = int(b)
                if self._child[node][b] < 0:
                    self._child.append([-1, -1])
                    self._leaf.append(None)
                    self._child[node][b] = len(self._child) - 1
                node = self._child[node][b]
            self._leaf[node] = ch

    # ------------------------------------------------------------------ build
    @classmethod
    def from_corpus(cls, sentences) -> "HuffmanCodec":
        freq = Counter()
        for s in sentences:
            freq.update(s)
        freq.pop(UNK_CHAR, None)
        freq[UNK_CHAR] = 1                           # reserved symbol for unseen characters
        tie = count()
        heap = [(f, next(tie), (ch,)) for ch, f in sorted(freq.items())]
        heapq.heapify(heap)
        depth_code = {ch: "" for ch in freq}
        while len(heap) > 1:
            f1, _, g1 = heapq.heappop(heap)
            f2, _, g2 = heapq.heappop(heap)
            for ch in g1:
                depth_code[ch] = "0" + depth_code[ch]
            for ch in g2:
                depth_code[ch] = "1" + depth_code[ch]
            heapq.heappush(heap, (f1 + f2, next(tie), g1 + g2))
        return cls(depth_code, dict(freq))

    # ------------------------------------------------------------- persistence
    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"code": self.code, "freq": self.freq},
                                         ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "HuffmanCodec":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["code"], d.get("freq"))

    # -------------------------------------------------------------- encode/decode
    def encode(self, text: str) -> np.ndarray:
        unk = self._code_bits[UNK_CHAR]
        parts = [self._code_bits.get(c, unk) for c in text]
        if not parts:
            return np.zeros(0, dtype=np.uint8)
        return np.concatenate(parts).astype(np.uint8)

    def decode(self, bits) -> str:
        """Greedy prefix decoding. A trailing incomplete code word is dropped."""
        out = []
        node = 0
        child, leaf = self._child, self._leaf
        for b in np.asarray(bits).tolist():
            node = child[node][b]
            if node < 0:                               # cannot happen for a complete code
                node = 0
                continue
            ch = leaf[node]
            if ch is not None:
                out.append(ch)
                node = 0
        return "".join(out)

    # ------------------------------------------------------------------ statistics
    def unknown_count(self, text: str) -> int:
        return sum(1 for c in text if c not in self.code or c == UNK_CHAR)

    @property
    def vocab_size(self) -> int:
        return len(self.code)

    def average_length(self) -> float:
        """Expected code length (bits/char) under the training distribution."""
        tot = sum(self.freq.values())
        return sum(self.freq[c] * len(self.code[c]) for c in self.freq) / tot

    def entropy(self) -> float:
        tot = sum(self.freq.values())
        return -sum(f / tot * math.log2(f / tot) for f in self.freq.values())

    def efficiency(self) -> float:
        return self.entropy() / self.average_length()

    def empirical_length(self, sentences) -> float:
        """Measured bits/char on an arbitrary set of sentences (e.g. the test set)."""
        n_bits = sum(len(self.encode(s)) for s in sentences)
        n_chars = sum(len(s) for s in sentences)
        return n_bits / n_chars

    def fixed_length(self) -> int:
        return math.ceil(math.log2(self.vocab_size))

    def verify_lossless(self, sentences) -> dict:
        """Encode -> decode every sentence; report mismatches (excluding UNK substitutions)."""
        bad = 0
        unk = 0
        for s in sentences:
            u = self.unknown_count(s)
            unk += u
            if self.decode(self.encode(s)) != "".join(UNK_CHAR if (c not in self.code) else c for c in s):
                bad += 1
        return {"n_sentences": len(sentences), "mismatches": bad, "unknown_chars": unk}
