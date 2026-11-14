import numpy as np
import pytest
from task1_conventional.huffman import HuffmanCodec, UNK_CHAR

CORPUS = ["Hello, world!", "The quick brown fox jumps over the lazy dog 0123456789.", "a b c; d?"]


def test_prefix_free_and_roundtrip():
    h = HuffmanCodec.from_corpus(CORPUS)
    codes = list(h.code.values())
    assert not any(a != b and b.startswith(a) for a in codes for b in codes)   # prefix free
    for s in CORPUS:
        assert h.decode(h.encode(s)) == s


def test_unknown_characters_are_handled():
    h = HuffmanCodec.from_corpus(CORPUS)
    out = h.decode(h.encode("Hello é world"))
    assert out == f"Hello {UNK_CHAR} world"
    assert h.unknown_count("Hello é") == 1


def test_efficiency_between_entropy_and_entropy_plus_one():
    h = HuffmanCodec.from_corpus(CORPUS)
    assert h.entropy() <= h.average_length() < h.entropy() + 1
    assert 0 < h.efficiency() <= 1


def test_save_load(tmp_path):
    h = HuffmanCodec.from_corpus(CORPUS)
    h.save(tmp_path / "cb.json")
    h2 = HuffmanCodec.load(tmp_path / "cb.json")
    assert h2.code == h.code and h2.decode(h.encode(CORPUS[1])) == CORPUS[1]


def test_truncated_stream_drops_incomplete_codeword():
    h = HuffmanCodec.from_corpus(CORPUS)
    text = "Hello, world!"
    bits = h.encode(text)
    out = h.decode(bits[:-1])                       # last code word loses its final bit
    assert text.startswith(out) and len(out) == len(text) - 1


def test_codebook_uses_training_data_only():
    h = HuffmanCodec.from_corpus(["aab"])
    assert "z" not in h.code
