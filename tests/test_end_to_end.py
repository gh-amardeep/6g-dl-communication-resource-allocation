import numpy as np
import pytest
from pathlib import Path
from task1_conventional.communication_system import MCS_TABLE, ConventionalLink, Frames
from task1_conventional.dataset import load_split
from task1_conventional.huffman import HuffmanCodec
from task1_conventional.metrics import (ber, bler, sentence_bleu, word_accuracy, wer, text_quality,
                                        word_alignment_counts)
from task1_conventional.amc import AMC
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data/test/test.txt").exists(), reason="dataset not prepared")


@pytest.fixture(scope="module")
def setup():
    tr, te = load_split("train", ROOT / "data"), load_split("test", ROOT / "data")
    codec = HuffmanCodec.from_corpus(tr)
    return codec, te[:60]


def test_dataset_requirements_and_no_leakage():
    tr, te = load_split("train", ROOT / "data"), load_split("test", ROOT / "data")
    assert len(tr) >= 20000 and len(te) >= 2000
    assert not set(tr) & set(te)


def test_huffman_lossless_on_test_set(setup):
    codec, _ = setup
    te = load_split("test", ROOT / "data")
    assert codec.verify_lossless(te)["mismatches"] == 0


@pytest.mark.parametrize("channel", ["awgn", "rayleigh"])
@pytest.mark.parametrize("mcs", MCS_TABLE, ids=lambda m: m.name)
def test_noiseless_end_to_end_is_exact(setup, mcs, channel):
    codec, sents = setup
    fr = Frames.build(sents, codec)
    link = ConventionalLink(codec, mcs, channel)
    res, text = link.run(fr, None, np.random.default_rng(0))
    assert res.bit_errors.sum() == 0 and res.ber == 0.0
    assert text == sents and bler(sents, text) == 0.0
    # decoded bits really decode to the text (not only the shortcut path)
    assert all(codec.decode(res.decoded[i, : fr.n_info[i]]) == sents[i] for i in range(len(sents)))


@pytest.mark.parametrize("channel", ["awgn", "rayleigh"])
def test_high_snr_is_clean_and_low_snr_is_not(setup, channel):
    codec, sents = setup
    fr = Frames.build(sents, codec)
    link = ConventionalLink(codec, MCS_TABLE[0], channel)
    hi = link.run(fr, 45.0, np.random.default_rng(3))[0]
    lo = link.run(fr, -3.0, np.random.default_rng(3))[0]
    assert hi.ber <= lo.ber and lo.ber > 0.1 and hi.ber < 0.01


def test_bler_in_unit_interval_and_monotone_trend(setup):
    codec, sents = setup
    fr = Frames.build(sents, codec)
    link = ConventionalLink(codec, MCS_TABLE[2], "awgn")
    bl = []
    for snr in (0.0, 5.0, 10.0):
        _, t = link.run(fr, snr, np.random.default_rng(5))
        bl.append(bler(sents, t))
    assert all(0.0 <= b <= 1.0 for b in bl) and bl[0] >= bl[1] >= bl[2] and bl[0] == 1.0 and bl[2] == 0.0


def test_reproducible_with_fixed_seed(setup):
    codec, sents = setup
    fr = Frames.build(sents, codec)
    link = ConventionalLink(codec, MCS_TABLE[3], "rayleigh")
    a = link.run(fr, 14.0, np.random.default_rng(42))[0]
    b = link.run(fr, 14.0, np.random.default_rng(42))[0]
    c = link.run(fr, 14.0, np.random.default_rng(43))[0]
    assert np.array_equal(a.bit_errors, b.bit_errors) and np.array_equal(a.decoded, b.decoded)
    assert not np.array_equal(a.bit_errors, c.bit_errors)


def test_ber_function():
    assert ber([0, 1, 1, 0], [0, 1, 0, 0]) == 0.25
    assert bler(["a", "b"], ["a", "x"]) == 0.5


def test_text_metrics_known_values():
    assert sentence_bleu("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert 0 < sentence_bleu("the cat sat on the mat today", "the cat sat on a mat today") < 1
    assert word_alignment_counts("a b c d", "a x c d") == (3, 1, 0, 0, 4)          # one substitution
    assert word_alignment_counts("a b c d", "a c d") == (3, 0, 1, 0, 4)            # one deletion
    assert word_alignment_counts("a b c", "a b x c") == (3, 0, 0, 1, 3)            # one insertion
    assert word_accuracy("a b c d", "a x c d") == 0.75
    assert wer("a b c", "a b x c") == pytest.approx(1 / 3)
    q = text_quality(["same text", "other text"], ["same text", "otxer text"])
    assert q["bleu"][0] == 1.0 and q["bleu"][1] < 1 and q["word_acc"][1] == 0.5


def _lut(blers):
    rows = []
    for snr, bl in blers.items():
        for m, b in zip(MCS_TABLE, bl):
            rows.append(dict(snr_db=snr, mcs=m.index, modulation=m.modulation, rate=m.rate,
                             efficiency=m.efficiency, ber=b / 10, bler=b, bleu=1 - b, word_acc=1 - b, cos_sim=1 - b))
    return pd.DataFrame(rows)


def test_amc_picks_highest_efficiency_within_target_and_falls_back():
    lut = _lut({0: [1, 1, 1, 1, 1, 1], 5: [0.05, 0.5, 1, 1, 1, 1], 10: [0, 0.1, 0.09, 0.5, 1, 1],
                15: [0, 0, 0, 0.1001, 0.9, 1], 20: [0, 0, 0, 0, 0.0, 0.2]})
    a = AMC(lut, 0.1)
    assert a.select(0)["mcs"] == 1 and a.select(0)["outage"]                 # fallback
    assert a.select(5)["mcs"] == 1 and not a.select(5)["outage"]
    assert a.select(10)["mcs"] == 3                                          # MCS2 (<=0.1) eff 1.5 < MCS3 eff 2.0
    assert a.select(15)["mcs"] == 3                                          # MCS4 BLER 0.1001 > 10 %
    assert a.select(20)["mcs"] == 5
    assert a.select(12.3)["grid_snr_db"] == 10                               # conservative grid point
    assert a.select(-5)["outage"]
    assert [m.efficiency for m in MCS_TABLE] == [1.0, 1.5, 2.0, 3.0, 4.0, 4.5]
