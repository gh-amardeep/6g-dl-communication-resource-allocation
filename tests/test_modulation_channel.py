import numpy as np
import pytest
from task1_conventional import channel as ch
from task1_conventional.demapper import llr_maxlog
from task1_conventional.equalizer import zf_equalize
from task1_conventional.modulation import Modem

RNG = np.random.default_rng(1)
NAMES = ["BPSK", "QPSK", "16QAM", "64QAM"]


@pytest.mark.parametrize("name", NAMES)
def test_unit_average_energy_and_size(name):
    m = Modem(name)
    pts, tab = m.constellation()
    assert len(pts) == m.M and len(set(np.round(pts, 9))) == m.M
    assert np.isclose(np.mean(np.abs(pts) ** 2), 1.0)


@pytest.mark.parametrize("name", NAMES)
def test_noiseless_roundtrip(name):
    m = Modem(name)
    bits = RNG.integers(0, 2, m.bps * 500)
    assert np.array_equal(m.demodulate_hard(m.modulate(bits)), bits)


@pytest.mark.parametrize("name", ["QPSK", "16QAM", "64QAM"])
def test_gray_property_nearest_neighbours_differ_by_one_bit(name):
    m = Modem(name)
    pts, tab = m.constellation()
    d = np.abs(pts[:, None] - pts[None, :]); np.fill_diagonal(d, np.inf)
    dmin = d.min()
    for i in range(m.M):
        for j in np.flatnonzero(np.isclose(d[i], dmin)):
            assert np.sum(tab[i] != tab[j]) == 1


def test_known_16qam_points():
    m = Modem("16QAM")
    s = m.modulate([0, 0, 0, 0])[0]                       # 00 -> -3 on both axes
    assert np.isclose(s, (-3 - 3j) / np.sqrt(10))
    s = m.modulate([1, 0, 0, 1])[0]                       # I=10 -> +3, Q=01 -> -1
    assert np.isclose(s, (3 - 1j) / np.sqrt(10))


@pytest.mark.parametrize("name", NAMES)
def test_incomplete_bit_group_is_padded(name):
    m = Modem(name)
    bits = RNG.integers(0, 2, m.bps * 10 + (m.bps - 1 if m.bps > 1 else 0))
    s = m.modulate(bits)
    assert len(s) == -(-len(bits) // m.bps)
    assert np.array_equal(m.demodulate_hard(s)[: len(bits)], bits)
    if m.bps > 1:
        with pytest.raises(ValueError):
            m.modulate(bits, pad=False)


def test_awgn_noise_variance():
    x = np.ones(400_000, dtype=complex)
    y, n0 = ch.awgn(x, 5.0, RNG)
    n = y - x
    assert np.isclose(n0, 10 ** -0.5)
    assert np.isclose(np.mean(np.abs(n) ** 2), n0, rtol=0.02)
    assert np.isclose(np.var(n.real), n0 / 2, rtol=0.03) and np.isclose(np.var(n.imag), n0 / 2, rtol=0.03)
    assert abs(np.mean(n)) < 0.01


def test_snr_conversions():
    assert np.isclose(ch.esn0_to_ebn0_db(10, 4, 0.5), 10 - 10 * np.log10(2))
    assert np.isclose(ch.ebn0_to_esn0_db(ch.esn0_to_ebn0_db(7.3, 6, 3 / 4), 6, 3 / 4), 7.3)
    assert np.isclose(ch.esn0_to_ebn0_db(4.0, 1, 1.0), 4.0)           # BPSK uncoded: Es = Eb


def test_rayleigh_statistics_and_block_structure():
    h = ch.rayleigh_gains(300_000, RNG)
    assert np.isclose(np.mean(np.abs(h) ** 2), 1.0, rtol=0.02)
    assert np.isclose(np.mean(np.abs(h)), np.sqrt(np.pi) / 2, rtol=0.02)   # Rayleigh mean (sigma^2 = 1/2)
    ids, nblk = ch.block_ids([10, 5, 84, 85], 84)
    assert nblk == 1 + 1 + 1 + 2 and ids[0] == ids[9] == 0 and ids[10] == 1 and ids[15] == 2
    assert ids[15 + 83] == 2 and ids[15 + 84] == 3 and ids[-1] == 4
    x = np.ones(len(ids), dtype=complex)
    y, hs, n0 = ch.rayleigh_block_fading(x, 200.0, ids, nblk, RNG)          # (almost) no noise
    for b in range(nblk):
        assert np.allclose(hs[ids == b], hs[ids == b][0])                    # constant inside an RB
    assert len({hs[ids == b][0] for b in range(nblk)}) == nblk               # independent across RBs
    assert np.allclose(y, hs * x, atol=1e-6)


def test_zf_equalizer_inverts_channel_and_handles_deep_fade():
    x = Modem("QPSK").modulate(RNG.integers(0, 2, 200))
    h = ch.rayleigh_gains(len(x), RNG)
    xh, gain = zf_equalize(h * x, h)
    assert np.allclose(xh, x) and np.allclose(gain, np.abs(h) ** 2)
    h0 = np.array([0.0 + 0j, 1e-12 + 0j, 1e-9j])
    xh, gain = zf_equalize(np.array([1.0, 1.0, 1.0], dtype=complex), h0)
    assert np.all(np.isfinite(xh)) and np.all(gain > 0)


@pytest.mark.parametrize("name", NAMES)
def test_llr_sign_convention_noiseless(name):
    m = Modem(name)
    bits = RNG.integers(0, 2, m.bps * 400)
    llr = llr_maxlog(m.modulate(bits), 0.05, m)
    assert len(llr) == len(bits)
    assert np.array_equal((llr < 0).astype(int), bits)                       # LLR>0 <=> bit 0
    assert np.all(llr[bits == 0] > 0) and np.all(llr[bits == 1] < 0)


@pytest.mark.parametrize("name", ["QPSK", "16QAM"])
def test_llr_matches_exact_formula_for_qpsk_and_scales_with_noise(name):
    m = Modem(name)
    pts, tab = m.constellation()
    y = np.array([0.3 + 0.1j, -0.7 + 0.4j])
    n0 = 0.2
    llr = llr_maxlog(y, n0, m).reshape(2, m.bps)
    for i, yy in enumerate(y):
        d2 = np.abs(yy - pts) ** 2
        for k in range(m.bps):
            ref = (d2[tab[:, k] == 1].min() - d2[tab[:, k] == 0].min()) / n0
            assert np.isclose(llr[i, k], ref)
    assert np.allclose(llr_maxlog(y, 2 * n0, m).reshape(2, m.bps) * 2, llr)


def test_llr_with_zf_uses_channel_gain():
    m = Modem("QPSK")
    x = m.modulate([0, 1, 1, 0])
    h = np.array([1.0 + 0j, 0.1 + 0j])
    xh, gain = zf_equalize(h * x + 0.02, h)
    llr = llr_maxlog(xh, 0.05 / gain, m).reshape(2, 2)
    assert np.abs(llr[1]).max() < np.abs(llr[0]).min()                       # weak channel => unreliable
