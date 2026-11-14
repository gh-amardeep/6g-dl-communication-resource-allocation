import numpy as np
import pytest

spacy = pytest.importorskip("spacy")
from task1_conventional.metrics import SentenceEmbedder


@pytest.fixture(scope="module")
def emb():
    try:
        return SentenceEmbedder("auto")
    except Exception as e:                                          # model not installed
        pytest.skip(f"no embedding backend: {e}")


def test_cosine_identity_similarity_and_range(emb):
    a = ["The president signed the new law on Friday."]
    same = emb.cosine(a, a)[0]
    para = emb.cosine(a, ["The president approved a new law on Friday."])[0]
    other = emb.cosine(a, ["Bananas ripen faster next to apples."])[0]
    garbled = emb.cosine(a, ["Thx prgsidwnt sigbed tte nea lax on Frxday."])[0]
    assert same == pytest.approx(1.0, abs=1e-5)
    assert para > other and para > garbled and -1 <= other <= 1
    assert "word-vector" in emb.backend or emb.backend.startswith("sbert")


def test_empty_sentence_gives_zero_similarity(emb):
    assert emb.cosine(["hello world"], [""])[0] == 0.0
