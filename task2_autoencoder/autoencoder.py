"""Small autoencoders for Exercise 1 / Task 2 (PyTorch, CPU).

Both models map one of M = 16 messages (4 information bits) to n real channel symbols
(n = 7 for the (7,4) code, n = 2 = one complex symbol for the (2,4) code) and back:

    one-hot(16) -> Dense(16, ReLU) -> Dense(n) -> power normalisation -> channel
                -> Dense(16, ReLU) -> Dense(16) -> softmax over the 16 messages

Power normalisation: the codebook (the 16 transmit vectors) is scaled so that the average
energy per *real* symbol, taken over the 16 equiprobable messages, is exactly 1:
    mean_{m, i} x_{m,i}^2 = power (1 for (7,4); 0.5 for (2,4), i.e. |x_complex|^2 = 1 on average).
Training uses cross-entropy between the softmax output and the true message index.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from task1_conventional.channel import ebn0_to_esn0_db, noise_variance   # reuse Task 1 SNR bookkeeping

M = 16          # messages = 2^4 information bits
K_BITS = 4


def int_to_bits(msg, k: int = K_BITS):
    """Message index -> k bits, MSB first. (numpy in, numpy out)"""
    msg = np.asarray(msg)
    return ((msg[..., None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8)


def bits_to_int(bits):
    bits = np.asarray(bits, dtype=np.int64)
    return (bits * (1 << np.arange(bits.shape[-1] - 1, -1, -1))).sum(-1)


class Autoencoder(nn.Module):
    def __init__(self, n_real: int, hidden: int = 16, rx_extra: int = 0, power: float = 1.0):
        """n_real: number of real channel symbols per message.
        rx_extra: extra receiver inputs (1 = the fading power |h|^2 used as CSI).
        power: average energy per real symbol (1 for the (7,4) code; 0.5 for the (2,4) code so
        that the single complex symbol has average power 1)."""
        super().__init__()
        self.n_real, self.hidden, self.rx_extra, self.power = n_real, hidden, rx_extra, power
        self.enc = nn.Sequential(nn.Linear(M, hidden), nn.ReLU(), nn.Linear(hidden, n_real))
        self.dec = nn.Sequential(nn.Linear(n_real + rx_extra, hidden), nn.ReLU(), nn.Linear(hidden, M))
        self.register_buffer("eye", torch.eye(M))

    # ------------------------------------------------------------------ transmitter
    def codebook(self) -> torch.Tensor:
        """(16, n_real) transmit vectors, average energy ``power`` per real symbol."""
        raw = self.enc(self.eye)
        return raw / torch.sqrt((raw ** 2).mean()) * (self.power ** 0.5)

    def encode(self, msg: torch.Tensor) -> torch.Tensor:
        return self.codebook()[msg]

    # ------------------------------------------------------------------ receiver
    def decode_logits(self, r: torch.Tensor, csi: torch.Tensor | None = None) -> torch.Tensor:
        if self.rx_extra:
            r = torch.cat([r, csi], dim=-1)
        return self.dec(r)

    def forward(self, msg, channel):
        """channel(x) -> (r, csi_or_None). Returns logits."""
        r, csi = channel(self.encode(msg))
        return self.decode_logits(r, csi)

    @torch.no_grad()
    def predict(self, r: torch.Tensor, csi: torch.Tensor | None = None) -> torch.Tensor:
        return self.decode_logits(r, csi).argmax(-1)


# ---------------------------------------------------------------------- channels (torch)
# Same SNR definitions as task1_conventional/channel.py: for a code of rate R = 4/n_real,
# Es/N0 = Eb/N0 * R per real symbol (Es = 1), real-noise variance sigma^2 = N0/2 = 1/(2 Es/N0).
def esn0_from_ebn0_db(ebn0_db, rate: float):
    """Es/N0 [dB] for one *real* symbol with Es = 1: Es/N0 = Eb/N0 * R (bits/symbol = 1)."""
    return float(ebn0_to_esn0_db(ebn0_db, bps=1, rate=rate))


def sigma_from_esn0_db(esn0_db: float) -> float:
    """Noise std per real dimension for Es = 1 (complex variance N0 -> N0/2 per real dim)."""
    return float(np.sqrt(noise_variance(esn0_db) / 2.0))


def awgn_channel(sigma: float):
    def ch(x):
        return x + sigma * torch.randn_like(x), None
    return ch


def rayleigh_channel(sigma: float):
    """One Rayleigh gain per message block (7 symbols). h = |CN(0,1)| so E[h^2] = 1; the
    phase is removed by coherent reception. Receiver CSI: h known -> ZF  x_hat = y / h.
    Returns the equalised vector and the CSI h^2 (needed to know how noisy x_hat is)."""
    def ch(x):
        hr, hi = torch.randn(x.shape[0], 1) / np.sqrt(2), torch.randn(x.shape[0], 1) / np.sqrt(2)
        h = torch.sqrt(hr ** 2 + hi ** 2).clamp_min(1e-3)
        y = h * x + sigma * torch.randn_like(x)
        return y / h, h ** 2
    return ch


# ---------------------------------------------------------------------- training
def train(model: Autoencoder, channel_factory, ebn0_db_range, rate: float, steps: int = 4000,
          batch: int = 1024, lr: float = 3e-3, seed: int = 0, log_every: int = 200):
    """Train with cross-entropy. Each step draws a random Eb/N0 in ``ebn0_db_range``
    (a (lo, hi) tuple; lo == hi gives a fixed training SNR). Returns the loss history."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = []
    for s in range(steps):
        lo, hi = ebn0_db_range
        ebn0 = lo + (hi - lo) * float(torch.rand(1).item())
        ch = channel_factory(sigma_from_esn0_db(esn0_from_ebn0_db(ebn0, rate)))
        msg = torch.randint(0, M, (batch,))
        loss = F.cross_entropy(model(msg, ch), msg)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % log_every == 0 or s == steps - 1:
            hist.append((s, float(loss.item())))
    return hist


def save_model(model: Autoencoder, path, meta: dict | None = None):
    torch.save({"state": model.state_dict(), "n_real": model.n_real, "hidden": model.hidden,
                "rx_extra": model.rx_extra, "power": model.power, "meta": meta or {}}, path)


def load_model(path) -> Autoencoder:
    d = torch.load(path, map_location="cpu", weights_only=False)
    m = Autoencoder(d["n_real"], d["hidden"], d["rx_extra"], d.get("power", 1.0))
    m.load_state_dict(d["state"]); m.eval()
    m.meta = d["meta"]
    return m
