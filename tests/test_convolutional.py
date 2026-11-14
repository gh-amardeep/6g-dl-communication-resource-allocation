import numpy as np
import pytest
from task1_conventional import convolutional as cv
from task1_conventional.viterbi import viterbi_single, viterbi_decode

RNG = np.random.default_rng(0)


def test_impulse_response_equals_generators():
    out = cv.encode_mother([1], terminate=True)           # 1 followed by 6 tail zeros
    a, b = out[0::2], out[1::2]
    assert list(a) == [1, 0, 1, 1, 0, 1, 1]               # 133 octal = 1011011
    assert list(b) == [1, 1, 1, 1, 0, 0, 1]               # 171 octal = 1111001


def test_fast_encoder_matches_shift_register():
    for n in (1, 7, 50, 333):
        b = RNG.integers(0, 2, n)
        assert np.array_equal(cv.encode_mother(b), cv.encode_reference(b))


def test_termination_returns_to_state_zero_and_length():
    b = RNG.integers(0, 2, 40)
    assert len(cv.encode_mother(b)) == 2 * (40 + 6)
    state = 0
    nxt, _ = cv.trellis()
    for x in list(b) + [0] * 6:
        state = nxt[state, x]
    assert state == 0


def test_trellis_matches_encoder():
    nxt, out = cv.trellis()
    b = RNG.integers(0, 2, 60)
    s, bits = 0, []
    for x in list(b) + [0] * 6:
        bits += list(out[s, x]); s = nxt[s, x]
    assert np.array_equal(np.array(bits), cv.encode_mother(b))


@pytest.mark.parametrize("rate,num,den", [("1/2", 1, 2), ("2/3", 2, 3), ("3/4", 3, 4)])
def test_puncturing_rate(rate, num, den):
    assert cv.nominal_rate(rate) == __import__("fractions").Fraction(num, den)
    b = RNG.integers(0, 2, 600)
    m = cv.encode_mother(b)
    p = cv.puncture(m, rate)
    assert abs(len(b) / len(p) - num / den) < 0.01        # only the 6 tail bits perturb it


@pytest.mark.parametrize("rate", ["1/2", "2/3", "3/4"])
def test_depuncture_places_values_and_erasures(rate):
    m = cv.encode_mother(RNG.integers(0, 2, 51)).astype(float)
    p = cv.puncture(m, rate)
    d = cv.depuncture(p + 1.0, rate, len(m))
    keep = np.flatnonzero(np.tile(cv.PUNCTURE_MASKS[rate], len(m))[: len(m)])
    assert np.allclose(d[keep], m[keep] + 1.0)
    assert np.count_nonzero(d) <= len(keep)               # erased positions are exactly 0 (no information)
    with pytest.raises(ValueError):
        cv.depuncture(p[:-1], rate, len(m))


@pytest.mark.parametrize("rate", ["1/2", "2/3", "3/4"])
@pytest.mark.parametrize("n", [1, 5, 100, 301])
def test_noiseless_viterbi_exact_recovery(rate, n):
    b = RNG.integers(0, 2, n)
    m = cv.encode_mother(b)
    llr = cv.depuncture(5.0 * (1 - 2.0 * cv.puncture(m, rate)), rate, len(m))
    assert np.array_equal(viterbi_single(llr, n), b)


def test_viterbi_is_maximum_likelihood_on_short_frames():
    import itertools
    k = 7
    cws = {bits: cv.encode_mother(bits) for bits in itertools.product([0, 1], repeat=k)}
    for _ in range(60):
        u = RNG.integers(0, 2, k)
        llr = 2 * ((1 - 2.0 * cv.encode_mother(u)) + RNG.normal(0, 1.1, 2 * (k + 6)))
        best = max(cws, key=lambda c: np.sum(llr * (1 - 2.0 * cws[c])))
        # Viterbi returns *a* maximiser; compare metrics (ties are measure-zero)
        dec = tuple(viterbi_single(llr, k))
        assert np.isclose(np.sum(llr * (1 - 2.0 * cws[dec])), np.sum(llr * (1 - 2.0 * cws[best])))


def test_batch_decoding_equals_single_with_mixed_lengths():
    lens = [10, 37, 5, 64]
    Tmax = max(lens) + 6
    llr = np.zeros((len(lens), 2 * Tmax), dtype=np.float32)
    bits = []
    for i, n in enumerate(lens):
        b = RNG.integers(0, 2, n); bits.append(b)
        m = cv.encode_mother(b)
        llr[i, : len(m)] = 2 * ((1 - 2.0 * m) + RNG.normal(0, 0.7, len(m)))
    dec = viterbi_decode(llr, np.array(lens))
    for i, n in enumerate(lens):
        assert np.array_equal(dec[i, :n], viterbi_single(llr[i, : 2 * (n + 6)], n))


def test_viterbi_corrects_errors_that_hard_threshold_cannot_fix():
    b = RNG.integers(0, 2, 200)
    m = cv.encode_mother(b)
    llr = 4.0 * (1 - 2.0 * m)
    flip = RNG.choice(len(m), 8, replace=False)           # a few isolated wrong (but weak) bits
    llr[flip] *= -0.3
    assert np.array_equal(viterbi_single(llr, 200), b)
