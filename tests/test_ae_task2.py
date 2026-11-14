"""Tests for Exercise 1 / Task 2 (autoencoders, Hamming baselines, channels)."""
import numpy as np
import pytest
import torch

from task2_autoencoder import baselines as bl
from task2_autoencoder.autoencoder import (Autoencoder, M, awgn_channel, bits_to_int, int_to_bits,
                                           load_model, rayleigh_channel, save_model,
                                           sigma_from_esn0_db, train)


def test_dimensions_and_power():
    for n, p in [(7, 1.0), (2, 0.5)]:
        m = Autoencoder(n, power=p)
        cb = m.codebook()
        assert cb.shape == (M, n)
        assert float((cb ** 2).mean()) == pytest.approx(p, rel=1e-5)
        msg = torch.randint(0, M, (32,))
        assert m.encode(msg).shape == (32, n)
        assert m(msg, awgn_channel(0.1)).shape == (32, M)
    m = Autoencoder(7, rx_extra=1)
    assert m(torch.randint(0, M, (5,)), rayleigh_channel(0.1)).shape == (5, M)


def test_bits_int_roundtrip():
    msg = np.arange(M)
    assert (bits_to_int(int_to_bits(msg)) == msg).all()


def test_hamming_code():
    d = [(a != b).sum() for i, a in enumerate(bl.CODEWORDS) for j, b in enumerate(bl.CODEWORDS) if i < j]
    assert min(d) == 3
    assert ((bl.G @ bl.H.T) % 2 == 0).all()
    assert (bl.hamming_hard_decode(bl.BPSK_CW) == bl.MSG_BITS).all()      # noiseless
    assert (bl.hamming_ml_decode(bl.BPSK_CW) == bl.MSG_BITS).all()
    r = bl.BPSK_CW.copy(); r[:, 2] *= -1                                  # any single error corrected
    assert (bl.hamming_hard_decode(r) == bl.MSG_BITS).all()


def test_awgn_noise_variance():
    torch.manual_seed(0)
    sigma = sigma_from_esn0_db(3.0)
    x = torch.zeros(200000, 7)
    y, _ = awgn_channel(sigma)(x)
    assert float(y.std()) == pytest.approx(sigma, rel=0.01)
    assert sigma ** 2 == pytest.approx(1 / (2 * 10 ** 0.3), rel=1e-9)


def test_rayleigh_and_zf():
    torch.manual_seed(1)
    x = torch.ones(200000, 7)
    y, csi = rayleigh_channel(0.0)(x)                  # no noise -> ZF restores x exactly
    assert torch.allclose(y, x, atol=1e-4)
    assert float(csi.mean()) == pytest.approx(1.0, rel=0.02)   # E|h|^2 = 1


def test_loss_backprop_and_training_reduces_loss():
    torch.manual_seed(0)
    m = Autoencoder(7)
    msg = torch.randint(0, M, (64,))
    loss = torch.nn.functional.cross_entropy(m(msg, awgn_channel(0.3)), msg)
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    hist = train(m, awgn_channel, (4, 4), 4 / 7, steps=300, seed=0)
    assert hist[-1][1] < hist[0][1]


def test_noiseless_reconstruction_after_training():
    m = Autoencoder(7)
    train(m, awgn_channel, (6, 6), 4 / 7, steps=1500, seed=0)
    pred = m.predict(m.codebook().detach())
    assert (pred == torch.arange(M)).all()


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = Autoencoder(2, power=0.5)
    save_model(m, tmp_path / "m.pt", meta={"a": 1})
    m2 = load_model(tmp_path / "m.pt")
    assert torch.allclose(m.codebook(), m2.codebook())
    assert m2.power == 0.5 and m2.meta == {"a": 1}


def test_ber_and_reproducibility():
    r1, n = bl.simulate_point(4.0, 50000, np.random.default_rng(5), "awgn")
    r2, _ = bl.simulate_point(4.0, 50000, np.random.default_rng(5), "awgn")
    assert r1 == r2                                                       # fixed seed
    for k, (be, ble) in r1.items():
        if k == "autoencoder":
            continue
        assert 0 <= be <= 4 * n and 0 <= ble <= n and ble <= be
    r0, _ = bl.simulate_point(60.0, 20000, np.random.default_rng(1), "awgn")   # essentially noiseless
    assert all(v[0] == 0 for k, v in r0.items() if k != "autoencoder")
    # uncoded BPSK vs theory
    big, nb = bl.simulate_point(2.0, 400000, np.random.default_rng(2), "awgn")
    assert big["uncoded_bpsk"][0] / (4 * nb) == pytest.approx(bl.q_bpsk_awgn(2.0), rel=0.05)
    bigr, nb = bl.simulate_point(10.0, 400000, np.random.default_rng(2), "rayleigh")
    assert bigr["uncoded_bpsk"][0] / (4 * nb) == pytest.approx(bl.bpsk_rayleigh(10.0), rel=0.05)
