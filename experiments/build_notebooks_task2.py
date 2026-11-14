"""Generate the three Exercise-1 Task-2 notebooks (run: python experiments/build_notebooks_task2.py)."""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "Exercise1_Task2"
OUT.mkdir(parents=True, exist_ok=True)
KS = {"display_name": "Python (Exercise1)", "language": "python", "name": "exercise1"}
SETUP = '''import os, sys
from pathlib import Path
ROOT = Path.cwd().parent.parent if Path.cwd().parent.name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
%matplotlib inline
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from task2_autoencoder import experiments as ex, baselines as bl
from task2_autoencoder.autoencoder import load_model
pd.set_option("display.width", 160)
print("results ->", ex.RES, "| models ->", ex.MODELS)'''


def nb(name, cells):
    n = nbf.v4.new_notebook(metadata={"kernelspec": KS})
    n.cells = [nbf.v4.new_markdown_cell(c) if k == "m" else nbf.v4.new_code_cell(c) for k, c in cells]
    nbf.write(n, OUT / name)


nb("01_autoencoder_7_4_AWGN.ipynb", [
("m", """# 01 - (7,4) autoencoder over AWGN
**Objective.** Train a small autoencoder that sends 4 information bits over 7 real channel uses and compare its BER with uncoded BPSK, Hamming(7,4) with hard-decision decoding and Hamming(7,4) with ML (soft) decoding.

**How it works.** The 4 bits select one of 16 messages (one-hot vector). A dense encoder (16 -> 16 ReLU -> 7) outputs 7 real numbers; they are scaled so the average energy per symbol is 1 (power normalisation). Gaussian noise is added, and a dense decoder (7 -> 16 ReLU -> 16 softmax) estimates which of the 16 messages was sent. Training minimises the cross-entropy between the softmax and the true message.

**SNR.** Eb/N0 is the energy per *information* bit. A coded symbol carries Es = (4/7) Eb, so Es/N0 = Eb/N0 + 10 log10(4/7) dB; noise std per real dimension = sqrt(N0/2). Task 1 helpers (`noise_variance`, `ebn0_to_esn0_db`) are reused."""),
("c", SETUP),
("m", "## 1. Train (cross-entropy, random Eb/N0 in 0-8 dB per batch, seed fixed)"),
("c", """model, hist = ex.train_model("ae74_awgn")
print("config:", ex.CFG["ae74_awgn"])
h = pd.DataFrame(hist, columns=["step", "loss"]); display(h.iloc[[0, len(h)//2, -1]])
plt.figure(figsize=(5, 3)); plt.plot(h.step, h.loss); plt.xlabel("step"); plt.ylabel("cross-entropy"); plt.title("Training loss"); plt.grid(alpha=.3); plt.show()
cb = model.codebook().detach().numpy()
print("codebook", cb.shape, "| average energy per symbol =", round(float((cb**2).mean()), 4))"""),
("m", "## 2. Evaluate on identical information bits (and identical noise for the coded systems)\nAt each Eb/N0, 2.5 million messages (10 million information bits) are sent; the lowest plotted points (8 dB) have only ~50-100 bit errors and are statistically weak; points with zero errors are not plotted."),
("c", """ae = load_model(ex.MODELS / "ae74_awgn.pt")
grid = np.arange(0, 9, 1.0)
df = ex.eval_74(ae, "awgn", grid, 2_500_000)
df.to_csv(ex.TAB / "ae74_awgn_results.csv", index=False)
display(df.pivot(index="ebn0_db", columns="method", values="ber").round(6))"""),
("c", """fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
ex.plot_curves(df, "ber", "(7,4) over AWGN: BER", ax=axs[0])
ex.plot_curves(df, "bler", "(7,4) over AWGN: block (4-bit message) error rate", ylabel="BLER", ax=axs[1])
axs[0].semilogy(grid, bl.q_bpsk_awgn(grid), "k:", lw=1, label="BPSK theory"); axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(ex.FIG / "ae74_awgn_ber_bler.png", dpi=150); plt.show()"""),
("m", "## 3. Discussion"),
("c", """p = df.pivot(index="ebn0_db", columns="method", values="ber")
def snr_at(col, target):   # Eb/N0 at which the BER curve crosses `target` (log-linear interpolation)
    d = p[col][p[col] > 0]; return float(np.interp(np.log10(target), np.log10(d.values[::-1]), d.index.values[::-1]))
t = 1e-3
print({c: round(snr_at(c, t), 2) for c in ["uncoded_bpsk", "hamming_hard", "hamming_ml", "autoencoder"]}, "dB needed for BER = 1e-3")"""),
("m", """* The autoencoder is clearly better than hard-decision Hamming decoding (about 0.9-1.1 dB less Eb/N0 for BER = 1e-2 / 1e-3 in the run above) because it, like ML, uses soft information. It is also better than uncoded BPSK for BER = 1e-3 by about 1.2 dB, but at very low SNR (about 1 dB and below) it is worse than uncoded BPSK.
* It does **not** reach Hamming-ML: it needs roughly 0.2-0.3 dB more Eb/N0 for the same BER (numbers printed above). Hamming-ML is the optimum decoder for the Hamming code, and the AE learns its code and decoder by gradient descent from noisy samples, so it can approach this reference but nothing forces it to match it; in this run it did not beat it.
* Coded curves start above uncoded BPSK at 0 dB because the code spends 7/4 times more symbols per information bit, so each symbol has less energy (Es = 4/7 Eb). Hamming-ML overtakes uncoded BPSK at about 1 dB, the AE at about 2 dB, and hard-decision Hamming only at about 6 dB."""),
])

nb("02_autoencoder_7_4_Rayleigh.ipynb", [
("m", """# 02 - (7,4) autoencoder over Rayleigh fading with receiver CSI
**Objective.** Repeat the (7,4) comparison when each 7-symbol block experiences one Rayleigh gain h (|h| Rayleigh, E|h|^2 = 1, i.e. block fading) and the receiver knows h.

**Receiver.** Zero-forcing equalisation x_hat = y/h (Task 1 `zf_equalize`, with the same magnitude floor). After ZF the noise std is sigma/|h|, so the AE decoder additionally receives the CSI |h|^2 as a ninth input; for Hamming-ML the per-block factor |h|^2 does not change the arg-min, so correlation on x_hat is still ML. Hard decision uses sign(x_hat). Uncoded BPSK uses the same fading realisations.

**Expected theory.** Uncoded BPSK with ZF/coherent detection: BER = 0.5 (1 - sqrt(g/(1+g))), g = Eb/N0 (slope 1/SNR, no diversity)."""),
("c", SETUP),
("m", "## 1. Train (random Eb/N0 in 0-20 dB per batch)"),
("c", """model, hist = ex.train_model("ae74_rayleigh")
print("config:", ex.CFG["ae74_rayleigh"])
h = pd.DataFrame(hist, columns=["step", "loss"]); display(h.iloc[[0, len(h)//2, -1]])"""),
("m", "## 2. Evaluate (2 million messages per point)"),
("c", """ae = load_model(ex.MODELS / "ae74_rayleigh.pt")
grid = np.arange(0, 25, 2.0)
dr = ex.eval_74(ae, "rayleigh", grid, 2_000_000)
dr.to_csv(ex.TAB / "ae74_rayleigh_results.csv", index=False)
display(dr.pivot(index="ebn0_db", columns="method", values="ber").round(6))
print("max |sim - theory| / theory for uncoded BPSK:", float((abs(dr[dr.method=='uncoded_bpsk'].ber.values / bl.bpsk_rayleigh(grid) - 1)).max()))"""),
("c", """fig, ax = plt.subplots(figsize=(6.4, 4.4))
ex.plot_curves(dr, "ber", "(7,4) over Rayleigh block fading (CSI at Rx, ZF)", ax=ax)
ax.semilogy(grid, bl.bpsk_rayleigh(grid), "k:", lw=1, label="BPSK Rayleigh theory"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(ex.FIG / "ae74_rayleigh_ber.png", dpi=150); plt.show()"""),
("m", "## 3. AWGN versus Rayleigh (AE and Hamming-ML)\nThe AWGN results come from notebook 01 (`ae74_awgn_results.csv`)."),
("c", """da = pd.read_csv(ex.TAB / "ae74_awgn_results.csv")
both = pd.concat([da, dr])
fig, ax = plt.subplots(figsize=(6.8, 4.6))
for ch, ls in (("awgn", "-"), ("rayleigh", "--")):
    for k in ("uncoded_bpsk", "hamming_ml", "autoencoder"):
        d = both[(both.channel == ch) & (both.method == k) & (both.ber > 0)]
        c, mk, _ = ex.STYLE[k]
        ax.semilogy(d.ebn0_db, d.ber, color=c, marker=mk, ls=ls, ms=3, label=f"{ex.LABELS[k]} ({ch})")
ax.set_xlabel("Eb/N0 [dB]"); ax.set_ylabel("BER"); ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=7); ax.set_title("AWGN (solid) vs Rayleigh (dashed)")
fig.tight_layout(); fig.savefig(ex.FIG / "ae74_awgn_vs_rayleigh.png", dpi=150); plt.show()
p = dr.pivot(index="ebn0_db", columns="method", values="ber"); display((p.div(p.hamming_ml, axis=0)).round(2).rename_axis("BER / BER(Hamming ML)"))"""),
("m", """* Fading destroys the exponential waterfall of AWGN: BER falls only like 1/SNR (about one decade per 10 dB) because a deep fade of the *single* gain hits all 7 symbols of the block together.
* Because one gain is shared by the whole codeword there is **no diversity** to exploit, and the code pays the 4/7 energy penalty. In this block-fading model uncoded BPSK is therefore *better* than every coded system: its BER is about 5-13 % lower than Hamming-ML (ratio 0.87-0.96 in the last table). A coding/diversity gain would require independent fades across the codeword (interleaving), which is outside the specified model.
* The AE with CSI follows the shape of Hamming-ML but stays 14-25 % above its BER at every point (ratio column; the gap grows slightly for Eb/N0 above its 0-20 dB training range), and is only slightly better than hard-decision Hamming. It does not beat ML."""),
])

nb("03_autoencoder_2_4_constellation.ipynb", [
("m", """# 03 - (2,4) autoencoder: learning a 16-point constellation
**Objective.** Map 4 bits to **one complex symbol** (two real numbers: I and Q), power-normalised to E|x|^2 = 1, and compare the learned constellation with Gray 16-QAM (Task 1 modem) over AWGN.

**SNR.** One symbol carries 4 bits: Es/N0 = 4 Eb/N0 (+6.02 dB); complex noise N0 = 1/(Es/N0). 16-QAM uses nearest-neighbour (ML) detection and Gray labelling. The AE decoder outputs one of 16 messages; message-bit labels are plain binary (no Gray), which matters for BER but not for the symbol error rate (SER)."""),
("c", SETUP),
("m", "## 1. Train (Eb/N0 random in 4-12 dB)"),
("c", """model, hist = ex.train_model("ae24_awgn")
print("config:", ex.CFG["ae24_awgn"])
h = pd.DataFrame(hist, columns=["step", "loss"]); display(h.iloc[[0, len(h)//2, -1]])
cb = model.codebook().detach().numpy(); c = cb[:, 0] + 1j * cb[:, 1]
print("average |x|^2 =", round(float(np.mean(abs(c)**2)), 4))"""),
("m", "## 2. Constellations"),
("c", """ex.plot_constellations(model, ex.FIG / "ae24_constellations.png"); plt.show()
pd.DataFrame({"message": range(16), "I": c.real, "Q": c.imag}).to_csv(ex.TAB / "ae24_learned_constellation.csv", index=False)"""),
("m", "## 3. BER and SER versus Eb/N0 (2 million symbols per point)"),
("c", """grid = np.arange(0, 14, 1.0)
d24 = ex.eval_24(model, grid, 2_000_000)
d24.to_csv(ex.TAB / "ae24_vs_16qam_results.csv", index=False)
display(d24.pivot(index="ebn0_db", columns="method", values=["ber", "ser"]).round(6))
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
ex.plot_curves(d24, "ber", "(2,4) AE vs 16-QAM: BER", ax=axs[0], methods=["autoencoder", "qam16"])
ex.plot_curves(d24, "ser", "(2,4) AE vs 16-QAM: symbol error rate", ylabel="SER", ax=axs[1], methods=["autoencoder", "qam16"])
fig.tight_layout(); fig.savefig(ex.FIG / "ae24_vs_16qam_ber_ser.png", dpi=150); plt.show()"""),
("m", "## 4. Explanation"),
("c", """from task1_conventional.modulation import get_modem
q = np.asarray(get_modem("16QAM").constellation()[0]).ravel()
dmin = lambda p: min(abs(a - b) for i, a in enumerate(p) for b in p[i+1:])
print(f"min distance  AE: {dmin(c):.3f}   16-QAM: {dmin(q):.3f}  (both at average power 1)")
print("peak |x|^2    AE: %.2f   16-QAM: %.2f" % ((abs(c)**2).max(), (abs(q)**2).max()))"""),
("m", """* **Learning.** The encoder only receives gradients through the noisy channel: points that are close are confused by the decoder and increase the cross-entropy, so training pushes them apart while the power constraint stops them from spreading out indefinitely. The result (see the figure) is an irregular packing: one point near the origin and the other 15 spread around it in a roughly circular/hexagonal pattern, instead of a square grid.
* **Differences from 16-QAM.** The AE has no square structure. Its minimum distance is actually slightly smaller than 16-QAM's (printed above, 0.60 vs 0.63 at equal average power), but its symbol error rate is lower: it needs about 0.3 dB less Eb/N0 than 16-QAM for SER = 1e-2 and about 0.2 dB less for SER = 1e-3 (interpolated from the table), presumably because fewer points have many close neighbours (the corner points of 16-QAM waste energy). 16-QAM has an explicit Gray labelling that makes neighbouring points differ in one bit, so its BER is about SER/4; the AE was trained on message-level cross-entropy and its bit labelling is arbitrary, so its **BER is worse than 16-QAM even though its SER is better**. Training with a bit-wise loss or a Gray-aware labelling would be needed to close this gap; we did not do it to keep the model simple.
* Every plotted point has at least 273 bit errors (2 million symbols per point), so the curves are statistically reasonable over 0-13 dB. For BER the AE is worse: about 0.7 dB more Eb/N0 for BER = 1e-2 and 0.3 dB for 1e-3."""),
])
print("ok")
