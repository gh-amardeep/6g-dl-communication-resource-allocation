"""Link and text-quality metrics.

* BER / sentence BLER                      (bit-level / sentence-level)
* BLEU        : sacreBLEU sentence BLEU (13a tokenisation, exponential smoothing), range 0..1
* Word accuracy: from a minimum-edit-distance word alignment of reference vs hypothesis
      hits H, substitutions S, deletions D, insertions I  (N = H + S + D reference words)
      word accuracy = H / N        (insertions are *not* hits and do not change H/N;
                                    they are reported separately through WER = (S+D+I)/N)
* Cosine similarity of sentence embeddings (see ``SentenceEmbedder``).
"""
from __future__ import annotations

import re

import numpy as np


# ---------------------------------------------------------------------------- bit level
def ber(tx_bits, rx_bits) -> float:
    tx, rx = np.asarray(tx_bits), np.asarray(rx_bits)
    if tx.shape != rx.shape:
        raise ValueError("shape mismatch")
    return float(np.mean(tx != rx)) if tx.size else 0.0


def bler(tx_sentences, rx_sentences) -> float:
    """Fraction of sentences whose reconstruction differs from the transmitted one."""
    if len(tx_sentences) != len(rx_sentences):
        raise ValueError("length mismatch")
    return float(np.mean([a != b for a, b in zip(tx_sentences, rx_sentences)]))


# ---------------------------------------------------------------------------- BLEU
def sentence_bleu(ref: str, hyp: str) -> float:
    if ref == hyp:
        return 1.0
    if not hyp.strip():
        return 0.0
    from sacrebleu.metrics import BLEU
    global _BLEU
    try:
        _BLEU
    except NameError:
        _BLEU = BLEU(smooth_method="exp", effective_order=True)
    return float(_BLEU.sentence_score(hyp, [ref]).score) / 100.0


# ---------------------------------------------------------------------------- words
def word_alignment_counts(ref: str, hyp: str):
    """Levenshtein alignment on whitespace tokens -> (hits, subs, dels, ins, n_ref)."""
    r, h = ref.split(), hyp.split()
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = r[i - 1]
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ri != h[j - 1]))
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            S += int(r[i - 1] != h[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return n - S - D, S, D, I, n


def word_accuracy(ref: str, hyp: str) -> float:
    hits, _, _, _, n = word_alignment_counts(ref, hyp)
    return hits / n if n else 1.0


def wer(ref: str, hyp: str) -> float:
    _, S, D, I, n = word_alignment_counts(ref, hyp)
    return (S + D + I) / n if n else 0.0


# ---------------------------------------------------------------------------- embeddings
_TOKEN = re.compile(r"\w+|[^\w\s]")


class SentenceEmbedder:
    """Sentence embedding + cosine similarity.

    backend "spacy-md" (default, works offline once the wheel is installed):
        mean of the pretrained 300-d word vectors of spaCy ``en_core_web_md`` over *all*
        tokens (unknown / corrupted tokens contribute a zero vector but still count in the
        denominator, so garbled words pull the embedding towards 0). This is a pretrained
        word-vector average, i.e. weaker than a transformer sentence encoder.
    backend "sbert": ``sentence-transformers`` ``all-MiniLM-L6-v2`` if the package and the
        model are available (requires Hugging Face access).
    """

    def __init__(self, backend: str = "auto"):
        self.backend = None
        self._cache: dict[str, np.ndarray] = {}
        if backend in ("auto", "sbert"):
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.backend = "sbert-all-MiniLM-L6-v2"
            except Exception:
                if backend == "sbert":
                    raise
        if self.backend is None:
            import spacy
            nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "tagger", "lemmatizer",
                                                       "attribute_ruler"])
            self._vocab = nlp.vocab
            self._dim = nlp.vocab.vectors.shape[1]
            self.backend = "spacy-en_core_web_md-mean-word-vectors"

    def _word_vec(self, w: str) -> np.ndarray:
        v = self._cache.get(w)
        if v is None:
            lex = self._vocab[w]
            v = lex.vector.copy() if lex.has_vector else np.zeros(self._dim, dtype=np.float32)
            self._cache[w] = v
        return v

    def embed(self, sentences) -> np.ndarray:
        if self.backend.startswith("sbert"):
            return np.asarray(self._model.encode(list(sentences), show_progress_bar=False))
        out = np.zeros((len(sentences), self._dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            toks = _TOKEN.findall(s)
            if toks:
                out[i] = np.sum([self._word_vec(t) for t in toks], axis=0) / len(toks)
        return out

    def cosine(self, refs, hyps) -> np.ndarray:
        a, b = self.embed(refs), self.embed(hyps)
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        den = na * nb
        cos = np.where(den > 0, np.sum(a * b, axis=1) / np.where(den > 0, den, 1), 0.0)
        return np.clip(cos, -1.0, 1.0)


def text_quality(refs, hyps, embedder: SentenceEmbedder | None = None) -> dict:
    """Per-sentence arrays (bleu, word_acc, cos) for lists of equal length. Identical
    pairs short-circuit to 1.0, which is exact for all three metrics."""
    n = len(refs)
    bl, wa, cs = np.ones(n), np.ones(n), np.ones(n)
    bad = [i for i in range(n) if refs[i] != hyps[i]]
    for i in bad:
        bl[i] = sentence_bleu(refs[i], hyps[i])
        wa[i] = word_accuracy(refs[i], hyps[i])
    if bad and embedder is not None:
        cs[bad] = embedder.cosine([refs[i] for i in bad], [hyps[i] for i in bad])
    return {"bleu": bl, "word_acc": wa, "cos": cs}
