"""Generates the eleven teaching notebooks in ../notebooks (run once; then execute them).

    python experiments/build_notebooks.py
"""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"
OUT.mkdir(exist_ok=True)

SETUP = '''import os, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))          # project root = import root
%matplotlib inline
import numpy as np, pandas as pd, matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
FIG, TAB = ROOT / "results/figures", ROOT / "results/tables"
print("project root:", ROOT)'''

MODE_CELL = '''from task1_conventional import experiments as ex, plotting
MODE = ex.best_available_mode(ROOT)      # "full" once `python main.py --experiment all --mode full` has finished
# MODE = "quick"                         # <- uncomment to force the small debugging data set
print("Using", MODE.upper(), "results", "" if MODE == "full" else "(QUICK MODE: small sample, NOT final results)")
sweep = pd.read_csv(TAB / f"{MODE}_mcs_sweep.csv")
cfg = ex.CONFIG[MODE]
print(f"{len(sweep)} (channel, MCS, SNR) points; {cfg['n_sentences']} sentences x {cfg['reps_awgn']} (AWGN) / {cfg['reps_rayleigh']} (Rayleigh) repetitions per point")'''


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata["kernelspec"] = {"name": "exercise1", "display_name": "Python (Exercise1)", "language": "python"}
    nbf.write(n, OUT / name)


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

# ---------------------------------------------------------------- 00
nb("00_environment_check.ipynb", [
    md("""# 00 - Environment check
**Objective.** Confirm that this notebook runs in the dedicated `.venv` / `Python (Exercise1)` kernel and that every dependency of Task 1 is importable.

Everything in this project is CPU-only (NumPy/SciPy vectorised code); no GPU or PyTorch is needed for Task 1.
Tasks 2 and 3 of the assignment will add PyTorch later - it is deliberately *not* required here."""),
    code(SETUP),
    md("## 1. Interpreter and kernel"),
    code('''import platform, sys, importlib
print("python     :", sys.version.split()[0], "|", platform.platform())
print("executable :", sys.executable)
in_venv = Path(sys.executable).resolve().as_posix().startswith((ROOT / ".venv").resolve().as_posix())
print("running inside project .venv:", in_venv)
try:
    import ipykernel; print("ipykernel  :", ipykernel.__version__)
except Exception as e: print(e)'''),
    md("## 2. Required packages"),
    code('''req = ["numpy", "scipy", "matplotlib", "pandas", "sklearn", "pytest", "tqdm", "sacrebleu", "spacy", "nbformat"]
rows = []
for m in req:
    try:
        mod = importlib.import_module(m); rows.append((m, getattr(mod, "__version__", "?"), "OK"))
    except Exception as e:
        rows.append((m, "-", f"MISSING ({e.__class__.__name__})"))
df = pd.DataFrame(rows, columns=["package", "version", "status"]); display(df)
assert (df.status == "OK").all(), "install missing packages: pip install -r requirements.txt"'''),
    md("""## 3. Sentence-embedding backend (Task 1c)
Hugging Face models cannot be downloaded in every environment. The project therefore uses the **spaCy `en_core_web_md` pretrained 300-d word vectors (mean-pooled)** as its default embedding, and switches to `sentence-transformers/all-MiniLM-L6-v2` automatically when that package and model are available. The active backend is printed below and recorded with the results."""),
    code('''from task1_conventional.metrics import SentenceEmbedder
emb = SentenceEmbedder()
print("embedding backend:", emb.backend)
print("cos(similar)   = %.3f" % emb.cosine(["The jury said the election was fair."], ["The jury declared the election fair."])[0])
print("cos(unrelated) = %.3f" % emb.cosine(["The jury said the election was fair."], ["Bananas ripen quickly in warm rooms."])[0])'''),
    md("## 4. Project modules"),
    code('''import task1_conventional as t1
from task1_conventional import (dataset, huffman, convolutional, modulation, channel, equalizer, demapper,
                                viterbi, communication_system, metrics, amc, experiments, plotting)
print("task1_conventional", t1.__version__, "- all modules import")
(ROOT / "results/logs").mkdir(parents=True, exist_ok=True)
with open(ROOT / "results/logs/environment.txt", "w") as f:
    f.write(f"python {sys.version}\\n" + "\\n".join(f"{r[0]}=={r[1]}" for r in rows) + f"\\nembedding_backend={emb.backend}\\n")
print("saved results/logs/environment.txt")'''),
])

# ---------------------------------------------------------------- 01
nb("01_dataset_preparation.ipynb", [
    md("""# 01 - Dataset preparation
**Objective.** Build a reproducible English text corpus with **>= 20 000 training** and **>= 2 000 test** sentences, strictly separated, and document it honestly.

**Concept.** The source encoder (Huffman) learns character statistics from *training* text only. If test sentences leaked into that training set the codebook would be tuned to them and the measured bits/character would be optimistic. We therefore split **by document** (a document never appears on both sides), then remove any sentence that is nevertheless identical in both sets.

**Source.** Brown Corpus from the NLTK data repository (`raw.githubusercontent.com/nltk/nltk_data`). If the download is impossible, place `brown.zip` in `data/raw/` (documented offline fallback)."""),
    code(SETUP),
    md("## 1. Build (or reuse) the dataset"),
    code('''import json
from task1_conventional.dataset import prepare_dataset, load_split
stats_file = ROOT / "data/processed/dataset_stats.json"
if not (ROOT / "data/train/train.txt").exists():
    stats = prepare_dataset(ROOT / "data")
else:
    stats = json.loads(stats_file.read_text())
display(pd.Series(stats, name="value").to_frame())'''),
    code('''train, test = load_split("train", ROOT / "data"), load_split("test", ROOT / "data")
assert len(train) >= 20000 and len(test) >= 2000, "assignment size requirement"
assert not set(train) & set(test), "train/test leakage"
print(f"train = {len(train)} sentences | test = {len(test)} sentences | exact overlap = {len(set(train) & set(test))}")
for s in test[:5]: print(" ", s)'''),
    md("""## 2. Preprocessing performed
1. POS tags removed, tokens **de-tokenised** (punctuation re-attached; quotes and contractions repaired). Capitalisation, digits and punctuation are **kept**.
2. Only sentences of **30-120 characters** with printable ASCII are kept (bounds the frame length, i.e. the number of resource blocks per sentence).
3. Exact duplicates removed; split by document; test = seeded random sample of 2 000 sentences from the held-out documents."""),
    md("## 3. Character statistics"),
    code('''from collections import Counter
cnt = Counter("".join(train))
chars = sorted(cnt, key=cnt.get, reverse=True)
print("training character vocabulary size:", len(chars), "| test vocabulary:", len(set("".join(test))))
print("characters in test but never in train:", sorted(set("".join(test)) - set(chars)))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
top = chars[:40]
ax[0].bar(range(len(top)), [cnt[c] for c in top]); ax[0].set_xticks(range(len(top))); ax[0].set_xticklabels([repr(c)[1:-1] if c != " " else "SP" for c in top], fontsize=7)
ax[0].set_yscale("log"); ax[0].set_title("40 most frequent characters (training)"); ax[0].set_ylabel("count")
ax[1].hist([len(s) for s in train], bins=45, alpha=0.6, label="train", density=True)
ax[1].hist([len(s) for s in test], bins=45, alpha=0.6, label="test", density=True)
ax[1].set_title("Sentence length"); ax[1].set_xlabel("characters"); ax[1].legend()
fig.tight_layout(); fig.savefig(FIG / "dataset_overview.png", dpi=150); plt.show()'''),
    md("""## 4. Limitations (please state them in the report)
* The Brown corpus is 1960s American English (news, fiction, academic prose): vocabulary and style differ from modern chat/web text.
* Character statistics are therefore specific to this corpus; the Huffman efficiency would change for other domains.
* Length filtering (30-120 chars) removes very short and very long sentences, so BLER values refer to *medium-length* sentences - longer sentences have higher BLER at the same SNR.
* Detokenisation is rule-based; a few artefacts (e.g. `Ill..`) remain.
* ASCII only, no accented characters."""),
])

# ---------------------------------------------------------------- 02
nb("02_huffman_source_coding.ipynb", [
    md("""# 02 - Character-level Huffman source coding
**Objective.** Compress sentences into bits with a variable-length prefix code built from training statistics, and verify that decoding is lossless.

**Concept.** Huffman coding gives frequent characters short code words. The expected length $\\bar L$ satisfies $H \\le \\bar L < H+1$ where $H=-\\sum p_i\\log_2 p_i$ is the source entropy (bits/char). **Efficiency** $=H/\\bar L$. A fixed-length code would need $\\lceil\\log_2 |V|\\rceil$ bits/char.
Characters never seen in training map to a reserved `UNK` symbol (reconstructed as U+FFFD) so encoding never fails."""),
    code(SETUP),
    code('''from task1_conventional.dataset import load_split
from task1_conventional.huffman import HuffmanCodec, UNK_CHAR
train, test = load_split("train", ROOT / "data"), load_split("test", ROOT / "data")
codec = HuffmanCodec.from_corpus(train)            # TRAINING DATA ONLY
codec.save(ROOT / "data/processed/huffman_codebook.json")
print("vocabulary (incl. UNK):", codec.vocab_size)'''),
    md("## 1. Code words"),
    code('''cb = pd.DataFrame({"char": [repr(c)[1:-1] if c not in (" ", UNK_CHAR) else ("SP" if c == " " else "UNK") for c in codec.code],
                   "freq": [codec.freq[c] for c in codec.code], "code": list(codec.code.values())})
cb["length"] = cb.code.str.len(); cb = cb.sort_values("freq", ascending=False).reset_index(drop=True)
display(cb.head(12)); display(cb.tail(4))
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.loglog(cb.freq, cb.length, "o"); ax.set_xlabel("training frequency"); ax.set_ylabel("code length [bits]")
ax.set_title("Frequent characters get short codes"); plt.show()'''),
    md("## 2. Lossless round trip on multiple sentences"),
    code('''for s in test[:3]:
    b = codec.encode(s); r = codec.decode(b)
    print(f"{len(s):3d} chars -> {len(b):3d} bits | ok={r == s} | {''.join(map(str, b[:40]))}...")
rep = codec.verify_lossless(test); print(rep)
assert rep["mismatches"] == 0
tr_rep = codec.verify_lossless(train[:5000]); print("first 5000 training sentences:", tr_rep)
print("unknown-char handling:", repr(codec.decode(codec.encode("caf\\u00e9"))))'''),
    md("## 3. Efficiency"),
    code('''res = {
 "entropy H [bit/char] (training pmf)": codec.entropy(),
 "average code length [bit/char] (training pmf)": codec.average_length(),
 "efficiency H / L": codec.efficiency(),
 "measured bits/char on TEST set": codec.empirical_length(test),
 f"fixed-length code ceil(log2 {codec.vocab_size})": codec.fixed_length(),
}
r = pd.Series(res, name="value").to_frame(); display(r)
bits_per_sentence = np.array([len(codec.encode(s)) for s in test])
print("test sentence: mean %.1f bits, min %d, max %d" % (bits_per_sentence.mean(), bits_per_sentence.min(), bits_per_sentence.max()))
print("compression vs fixed-length: %.1f %%" % (100 * (1 - res["measured bits/char on TEST set"] / codec.fixed_length())))
r.to_csv(TAB / "huffman_statistics.csv")'''),
    md("""**Take-away.** The code is within a fraction of a bit of the entropy and compresses ~36 % relative to a fixed-length code. Because Huffman output is *highly compressed*, every residual bit error is damaging: a single flipped bit desynchronises the prefix code and garbles the characters that follow. This is the root of the cliff effect studied in notebook 09."""),
])

# ---------------------------------------------------------------- 03
nb("03_convolutional_coding.ipynb", [
    md("""# 03 - Convolutional coding, puncturing and Viterbi decoding
**Objective.** Implement and verify the rate-1/2, K=7, generators (133, 171)$_8$ code, its punctured rates 2/3 and 3/4 and the soft-decision Viterbi decoder.

**Concept.** The encoder is a 6-bit shift register (64 states). Each input bit produces two coded bits
$c_A=\\text{parity}(\\text{reg}\\,\\&\\,133_8)$, $c_B=\\text{parity}(\\text{reg}\\,\\&\\,171_8)$. Six zero **tail bits** return the register to state 0 (termination), so $L$ information bits give $2(L+6)$ coded bits.

**Puncturing** deletes coded bits following a periodic mask, raising the rate: the 2/3 mask keeps 3 of every 4 coded bits ($\\tfrac12\\cdot\\tfrac43=\\tfrac23$), the 3/4 mask keeps 4 of every 6 ($\\tfrac12\\cdot\\tfrac32=\\tfrac34$). The decoder re-inserts **zero LLRs** (no information) at deleted positions ("depuncturing") and runs the *same* rate-1/2 trellis.

**LLR convention.** $\\text{LLR}=\\ln\\frac{P(c=0)}{P(c=1)}$, positive means bit 0. Viterbi maximises $\\sum \\text{LLR}\\cdot(1-2c)/2$."""),
    code(SETUP),
    code('''from task1_conventional import convolutional as cv
from task1_conventional.viterbi import viterbi_single, viterbi_decode
rng = np.random.default_rng(0)'''),
    md("## 1. Encoder verification: impulse response = generator polynomials"),
    code('''out = cv.encode_mother([1]); A, B = out[0::2], out[1::2]
print("133 octal =", format(0o133, "07b"), "-> branch A response", "".join(map(str, A)))
print("171 octal =", format(0o171, "07b"), "-> branch B response", "".join(map(str, B)))
assert "".join(map(str, A)) == format(0o133, "07b") and "".join(map(str, B)) == format(0o171, "07b")
b = rng.integers(0, 2, 100)
assert np.array_equal(cv.encode_mother(b), cv.encode_reference(b)); print("fast encoder == shift-register reference encoder")
nxt, o = cv.trellis(); s = 0
for x in list(b) + [0]*6: s = nxt[s, x]
print("final state after termination:", s, "| coded length", len(cv.encode_mother(b)), "= 2*(100+6)")'''),
    md("## 2. Puncturing patterns and effective rate"),
    code('''rows = []
for r, m in cv.PUNCTURE_MASKS.items():
    rows.append(dict(rate=r, mask="".join(map(str, m.astype(int))), period_coded_bits=len(m), kept=int(m.sum()),
                     nominal_rate=str(cv.nominal_rate(r)),
                     eff_rate_L=" / ".join(f"{L}: {cv.effective_rate(L, r):.4f}" for L in (50, 350, 1000))))
display(pd.DataFrame(rows))'''),
    md("""The **effective** rate of a finite frame is slightly below nominal because the 6 tail bits are also transmitted (and all coded bits of the tail are kept in the mother stream). For the sentence lengths used here (~330 info bits) this costs about 1.7 %. The AMC rule uses the *nominal* rate $m\\cdot R$ for spectral efficiency; the effective rate is reported for completeness."""),
    md("## 3. Noiseless recovery at all three rates"),
    code('''for rate in cv.PUNCTURE_MASKS:
    ok = True
    for n in (1, 6, 7, 50, 333):
        b = rng.integers(0, 2, n); m = cv.encode_mother(b)
        llr = cv.depuncture(8.0 * (1 - 2.0 * cv.puncture(m, rate)), rate, len(m))
        ok &= np.array_equal(viterbi_single(llr, n), b)
    print(f"rate {rate}: noiseless exact recovery for all tested lengths ->", ok)'''),
    md("""## 4. Is the Viterbi decoder really *maximum likelihood*?
For short frames we can brute-force **all** $2^k$ codewords and pick the best soft metric. Viterbi must achieve the same metric."""),
    code('''import itertools
k = 8; cws = {bits: cv.encode_mother(bits) for bits in itertools.product([0, 1], repeat=k)}
agree = 0; trials = 300
for _ in range(trials):
    u = rng.integers(0, 2, k); x = 1 - 2.0 * cv.encode_mother(u)
    llr = 2 * (x + rng.normal(0, 1.0, len(x)))
    d = tuple(viterbi_single(llr, k))
    best = max(cws, key=lambda c: np.sum(llr * (1 - 2.0 * cws[c])))
    agree += np.isclose(np.sum(llr*(1-2.0*cws[d])), np.sum(llr*(1-2.0*cws[best])))
print(f"Viterbi metric == brute-force ML metric in {agree}/{trials} noisy trials")'''),
    md("""## 5. Coding gain: BPSK + soft Viterbi over AWGN
Quick Monte-Carlo (400 frames x 500 bits per point). Axis = $E_b/N_0$, where for rate $R$ BPSK: $E_s/N_0 = R\\,E_b/N_0$ (a *coded* bit carries only $R$ information bits of energy)."""),
    code('''from task1_conventional.modulation import Modem
from task1_conventional.demapper import llr_maxlog
from task1_conventional import channel as ch
from scipy.special import erfc
bpsk = Modem("BPSK"); eb_grid = np.arange(0, 8.1, 1.0); nfr, L = 400, 500
res = {}
for rate in cv.PUNCTURE_MASKS:
    R = float(cv.nominal_rate(rate)); ber_l = []
    for eb in eb_grid:
        es = ch.ebn0_to_esn0_db(eb, 1, R); errs = 0
        tx = rng.integers(0, 2, (nfr, L)); T = L + 6
        llr = np.zeros((nfr, 2 * T), dtype=np.float32)
        for i in range(nfr):
            m = cv.encode_mother(tx[i]); p = cv.puncture(m, rate)
            y, n0 = ch.awgn(bpsk.modulate(p), es, rng)
            llr[i] = cv.depuncture(llr_maxlog(y, n0, bpsk), rate, len(m))
        dec = viterbi_decode(llr, np.full(nfr, L))[:, :L]
        ber_l.append(np.mean(dec != tx))
    res[rate] = ber_l
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(eb_grid, 0.5 * erfc(np.sqrt(10 ** (eb_grid / 10))), "k--", label="uncoded BPSK (theory)")
for rate, v in res.items(): ax.semilogy(eb_grid, np.where(np.array(v) > 0, v, np.nan), "o-", label=f"coded R={rate}")
ax.set_ylim(1e-6, 1); ax.set_xlabel("Eb/N0 [dB]"); ax.set_ylabel("BER (info bits)"); ax.legend(); ax.set_title("Soft-decision Viterbi, K=7 (133,171)")
fig.savefig(FIG / "conv_code_gain_bpsk.png", dpi=150); plt.show()
print("(points with zero errors are omitted; each point is based on", nfr * L, "bits - resolution ~", 1 / (nfr * L), ")")'''),
    md("Higher rates (more puncturing) lose coding gain and need larger $E_b/N_0$ for the same BER - the price of the higher spectral efficiency."),
])

# ---------------------------------------------------------------- 04
nb("04_modulation_demodulation.ipynb", [
    md("""# 04 - Gray-coded QPSK, 16-QAM and 64-QAM
**Objective.** Build the mapper/demapper, verify Gray labelling and energy normalisation, and confirm the simulated uncoded BER against theory.

**Concept.** Square $M$-QAM = two independent Gray-coded PAM axes. The first $m=\\log_2M/2$ bits select the in-phase level and the next $m$ bits the quadrature level. Levels are the odd integers $\\pm1,\\pm3,\\dots$ scaled by $\\sqrt{3/(2(M-1))}$ so that $E_s=1$. Gray labelling means nearest neighbours differ in **one** bit, so the most likely symbol error costs a single bit error.

$P_b \\approx \\dfrac{4}{k}\\Bigl(1-\\dfrac1{\\sqrt M}\\Bigr)Q\\Bigl(\\sqrt{\\dfrac{3k}{M-1}\\dfrac{E_b}{N_0}}\\Bigr)$, $k=\\log_2M$ (exact for QPSK; tight approximation for 16/64-QAM at moderate/high SNR)."""),
    code(SETUP),
    code('''from task1_conventional.modulation import Modem
from task1_conventional import channel as ch
from task1_conventional import plotting
from scipy.special import erfc
rng = np.random.default_rng(1)'''),
    md("## 1. Constellations"),
    code('''fig = plotting.plot_constellations(FIG / "constellations.png"); plt.show()'''),
    md("## 2. Verification checklist"),
    code('''rows = []
for n in ["BPSK", "QPSK", "16QAM", "64QAM"]:
    m = Modem(n); pts, tab = m.constellation()
    d = np.abs(pts[:, None] - pts[None, :]); np.fill_diagonal(d, np.inf); dmin = d.min()
    gray = all(np.sum(tab[i] != tab[j]) == 1 for i in range(m.M) for j in np.flatnonzero(np.isclose(d[i], dmin)))
    bits = rng.integers(0, 2, m.bps * 1000)
    rows.append(dict(modulation=n, bits_per_symbol=m.bps, points=m.M, Es=round(float(np.mean(abs(pts)**2)), 6),
                     d_min=round(float(dmin), 4), gray_neighbours_1bit=gray,
                     noiseless_roundtrip=bool(np.array_equal(m.demodulate_hard(m.modulate(bits)), bits))))
display(pd.DataFrame(rows))'''),
    md("### Incomplete bit groups\nA bit stream whose length is not a multiple of $\\log_2 M$ is **zero-padded**; the caller keeps the true length and drops the padding bits after demapping (done by the link in notebook 07)."),
    code('''m = Modem("16QAM"); bits = np.array([1, 0, 1, 1, 0, 1])        # 6 bits = 1.5 symbols
s = m.modulate(bits); print("symbols:", len(s), "| recovered:", m.demodulate_hard(s)[:len(bits)], "| padded bits added:", len(s) * m.bps - len(bits))'''),
    md("## 3. Uncoded BER vs theory (AWGN)"),
    code('''def ber_theory(mod, ebn0_db):
    k = Modem(mod).bps; M = 2 ** k; g = 10 ** (np.asarray(ebn0_db) / 10)
    Q = lambda x: 0.5 * erfc(x / np.sqrt(2))
    return 4 / k * (1 - 1 / np.sqrt(M)) * Q(np.sqrt(3 * k / (M - 1) * g))
fig, ax = plt.subplots(figsize=(7.5, 5)); eb = np.arange(0, 16.1, 1.0)
for n, c in zip(["QPSK", "16QAM", "64QAM"], ["C0", "C1", "C2"]):
    mo = Modem(n); sim = []
    for e in eb:
        es = ch.ebn0_to_esn0_db(e, mo.bps, 1.0); errs = bits = 0
        while errs < 200 and bits < 6_000_000:
            tx = rng.integers(0, 2, 600_000); y, _ = ch.awgn(mo.modulate(tx), es, rng)
            errs += np.count_nonzero(mo.demodulate_hard(y) != tx); bits += len(tx)
        sim.append(errs / bits)
    ax.semilogy(eb, sim, "o", color=c, ms=4, label=f"{n} simulated"); ax.semilogy(eb, ber_theory(n, eb), "-", color=c, label=f"{n} theory")
ax.set_ylim(1e-6, 1); ax.set_xlabel("Eb/N0 [dB]"); ax.set_ylabel("uncoded BER"); ax.legend(ncol=2, fontsize=8)
ax.set_title("Gray QAM, hard decision, AWGN"); fig.savefig(FIG / "uncoded_qam_ber.png", dpi=150); plt.show()'''),
    md("Simulation and theory agree (QPSK exactly; 16/64-QAM up to the nearest-neighbour approximation at low $E_b/N_0$). Higher-order modulation carries more bits per symbol but needs much more $E_b/N_0$ for the same BER - the trade-off AMC exploits."),
])

# ---------------------------------------------------------------- 05
nb("05_channel_models.ipynb", [
    md("""# 05 - Channel models and SNR bookkeeping
**Objective.** Implement AWGN and Rayleigh block fading, verify their statistics, and state exactly how $\\text{SNR}$, $E_s/N_0$, $E_b/N_0$, modulation order and code rate relate.

## SNR definitions (used everywhere in this project)
* **SNR $=E_s/N_0$**: received energy per *channel symbol* over the *complex* noise variance $N_0$ (this is the assignment's $\\gamma=p\\,g/(N_0BF)$ for one RB). With $E_s=1$: $n\\sim\\mathcal{CN}(0,N_0)$, $N_0=1/\\text{SNR}$, i.e. $N_0/2$ per real dimension.
* $E_b/N_0 = \\dfrac{E_s/N_0}{m\\,R}$ with $m$ = bits/symbol and $R$ = code rate (info bits per coded bit). In dB: $E_b/N_0 = E_s/N_0 - 10\\log_{10}(mR)$.
* Uncoded BPSK: $mR=1\\Rightarrow E_s/N_0=E_b/N_0$. Uncoded QPSK: $mR=2$, so at equal $E_s/N_0$ QPSK has 3 dB lower $E_b/N_0$ (same BER-vs-$E_b/N_0$ curve as BPSK).
* **Rayleigh block fading**: $y = h x + n$, $h\\sim\\mathcal{CN}(0,1)$ constant over one resource block ($84$ symbols = 12 subcarriers x 7 OFDM symbols) and **independent between RBs**; $E|h|^2=1$ so SNR is the *average*, the instantaneous SNR of an RB is $|h|^2\\,\\text{SNR}$. A sentence of ~350-1000 symbols spans several RBs. Perfect CSI at the receiver is assumed."""),
    code(SETUP),
    code('''from task1_conventional import channel as ch
from task1_conventional.communication_system import MCS_TABLE
rng = np.random.default_rng(2)'''),
    md("## 1. AWGN: noise variance check"),
    code('''x = np.ones(1_000_000, dtype=complex)
rows = []
for snr in (0, 5, 10, 20):
    y, n0 = ch.awgn(x, snr, rng); n = y - x
    rows.append(dict(snr_dB=snr, N0_target=1 / 10 ** (snr / 10), var_complex=np.mean(abs(n) ** 2), var_real=np.var(n.real) * 2, var_imag=np.var(n.imag) * 2))
display(pd.DataFrame(rows))'''),
    md("## 2. Rayleigh fading statistics"),
    code('''h = ch.rayleigh_gains(500_000, rng); g = abs(h) ** 2
print("E|h|^2 = %.4f (target 1) | E|h| = %.4f (Rayleigh: sqrt(pi)/2 = %.4f)" % (g.mean(), abs(h).mean(), np.sqrt(np.pi) / 2))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(g, bins=100, range=(0, 5), density=True, alpha=0.6, label="simulated |h|^2"); xs = np.linspace(0, 5, 200)
ax[0].plot(xs, np.exp(-xs), "k", label="exp(-x) (theory)"); ax[0].set_title("Power gain |h|^2 ~ Exp(1)"); ax[0].legend()
ax[1].semilogy(np.sort(g), 1 - np.arange(1, len(g) + 1) / len(g), label="simulated"); ax[1].semilogy(xs, np.exp(-xs), "k--", label="exp(-x)")
ax[1].set_xlim(0, 8); ax[1].set_ylim(1e-5, 1); ax[1].set_title("P(|h|^2 > x)"); ax[1].legend(); plt.show()
print("P(deep fade |h|^2 < 0.01) = %.4f (theory 1-exp(-0.01) = %.4f)" % ((g < 0.01).mean(), 1 - np.exp(-0.01)))'''),
    md("## 3. Block structure: independent fading per resource block"),
    code('''n_sym = [400, 350, 700]; ids, nblk = ch.block_ids(n_sym, 84)
x = np.ones(len(ids), dtype=complex); y, hs, n0 = ch.rayleigh_block_fading(x, 20.0, ids, nblk, rng)
print("frames (symbols):", n_sym, "-> resource blocks:", nblk)
fig, ax = plt.subplots(figsize=(10, 3.5)); ax.plot(abs(hs)); ax.set_xlabel("symbol index (3 sentences concatenated)"); ax.set_ylabel("|h|")
ax.set_title("|h| is piecewise constant over 84-symbol RBs; new RB -> new independent fade")
for b in np.cumsum(n_sym)[:-1]: ax.axvline(b, color="r", ls=":")
plt.show()'''),
    md("## 4. $E_s/N_0$ vs $E_b/N_0$ for the six MCSs"),
    code('''rows = []
for m in MCS_TABLE:
    for es in (5, 10, 20):
        rows.append(dict(MCS=m.name, bits_per_symbol=m.bps, code_rate=str(m.code_rate), mR=m.efficiency, EsN0_dB=es, EbN0_dB=round(float(ch.esn0_to_ebn0_db(es, m.bps, float(m.code_rate))), 2)))
t = pd.DataFrame(rows); display(t.pivot(index="MCS", columns="EsN0_dB", values="EbN0_dB"))
t.to_csv(TAB / "esn0_ebn0_conversion.csv", index=False)'''),
    md("**Reading it:** at the same $E_s/N_0$ a high-order, high-rate MCS has a *much lower* $E_b/N_0$ (each bit gets less energy). All plots in this project use $E_s/N_0$ (the assignment's per-RB SNR $\\gamma$) on the x-axis; the $E_b/N_0$ equivalent is stored in every CSV (`ebn0_db`)."),
    md("## 5. Uncoded BPSK over Rayleigh fading vs closed form (validation of fading + ZF)"),
    code('''from task1_conventional.equalizer import zf_equalize
from task1_conventional.modulation import Modem
bpsk = Modem("BPSK"); snr = np.arange(0, 21, 2.0); sim = []
for s in snr:
    tx = rng.integers(0, 2, 2_000_000); x = bpsk.modulate(tx)
    y, hh, _ = ch.rayleigh_block_fading(x, s, np.arange(len(x)), len(x), rng); xh, _ = zf_equalize(y, hh)
    sim.append(np.mean(bpsk.demodulate_hard(xh) != tx))
g = 10 ** (snr / 10); th = 0.5 * (1 - np.sqrt(g / (1 + g)))
fig, ax = plt.subplots(figsize=(6.5, 4.2)); ax.semilogy(snr, sim, "o", label="simulated (ZF)"); ax.semilogy(snr, th, "k-", label="0.5(1-sqrt(g/(1+g)))")
ax.set_xlabel("average SNR [dB]"); ax.set_ylabel("BER"); ax.legend(); ax.set_title("Rayleigh, uncoded BPSK: only ~1/(4g) decay (diversity order 1)"); plt.show()
print("max relative error vs theory: %.3f" % np.max(abs(np.array(sim) / th - 1)))'''),
])

# ---------------------------------------------------------------- 06
nb("06_equalization_and_soft_demapping.ipynb", [
    md("""# 06 - Zero-forcing equalisation and max-log soft demapping
**Objective.** Undo the fading with a ZF equaliser and turn the equalised symbols into **bit LLRs** that carry the *reliability* of each bit to the Viterbi decoder.

## Zero forcing
$y=hx+n \\;\\Rightarrow\\; \\hat x=y/h = x + n/h$. The channel is removed completely, but the noise is scaled: its variance becomes $N_0/|h|^2$.
* **Deep fades** ($|h|\\to0$) blow the noise up. Numerically we floor $|h|\\ge\\varepsilon=10^{-3}$ (keeping the phase) so no division overflows.
* **Limitations:** noise enhancement (for Rayleigh, $E[1/|h|^2]=\\infty$); with one antenna ZF is plain channel inversion and is not MMSE-optimal at low SNR; it needs perfect CSI.

## Max-log LLR
$$\\text{LLR}_k=\\ln\\frac{P(b_k=0|\\hat x)}{P(b_k=1|\\hat x)}\\approx\\frac{\\min_{s:b_k=1}|\\hat x-s|^2-\\min_{s:b_k=0}|\\hat x-s|^2}{N_{0,\\text{eff}}},\\qquad N_{0,\\text{eff}}=\\frac{N_0}{|h|^2}.$$
Positive $\\Rightarrow$ bit 0. Dividing by $N_{0,\\text{eff}}$ is what makes a bit received in a deep fade have an LLR near 0 ("don't trust me"), which is exactly what lets the Viterbi decoder ride through faded blocks."""),
    code(SETUP),
    code('''from task1_conventional import channel as ch
from task1_conventional.equalizer import zf_equalize
from task1_conventional.demapper import llr_maxlog
from task1_conventional.modulation import Modem
rng = np.random.default_rng(3)'''),
    md("## 1. ZF on a faded constellation"),
    code('''m = Modem("16QAM"); tx = rng.integers(0, 2, 4 * 3000); x = m.modulate(tx); snr = 22.0
ids = np.arange(len(x)) // 1; y, h, n0 = ch.rayleigh_block_fading(x, snr, ids, len(x), rng); xh, gain = zf_equalize(y, h)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].scatter(y.real, y.imag, s=3, alpha=0.4); ax[0].set_title("received y = h x + n"); lim = 4
ax[1].scatter(xh.real, xh.imag, s=3, alpha=0.3, c=np.log10(abs(h)), cmap="viridis"); ax[1].set_title("after ZF (colour = log10|h|)")
ax[2].scatter(xh.real, xh.imag, s=3, alpha=0.3); ax[2].set_title("after ZF, zoom"); ax[2].set_xlim(-1.5, 1.5); ax[2].set_ylim(-1.5, 1.5)
for a in ax[:2]: a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
for a in ax: a.set_aspect("equal")
plt.show()
print("ZF restores the grid for strong |h| but smears points from deep fades far away (noise enhancement).")'''),
    md("## 2. Noise enhancement and the deep-fade guard"),
    code('''err = abs(xh - x) ** 2
bins = [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 10]; cat = pd.cut(abs(h) ** 2, bins)
tab = pd.DataFrame({"|h|^2 bin": cat, "mean |xhat - x|^2": err, "theory N0/|h|^2 (mean)": n0 / abs(h) ** 2}).groupby("|h|^2 bin", observed=True).mean()
display(tab)
h0 = np.array([0j, 1e-12 + 0j, 1e-4 + 0j, 1.0 + 0j]); xz, gz = zf_equalize(np.ones(4, complex), h0)
print("guarded division with h =", h0, "\\n  x_hat =", xz, "\\n  finite:", np.all(np.isfinite(xz)))'''),
    md("## 3. LLR sign convention (known bits, noiseless and noisy)"),
    code('''rows = []
for n in ["BPSK", "QPSK", "16QAM", "64QAM"]:
    mo = Modem(n); b = rng.integers(0, 2, mo.bps * 2000); s = mo.modulate(b)
    l_clean = llr_maxlog(s, 0.01, mo); y_, n0_ = ch.awgn(s, 12.0, rng); l_noisy = llr_maxlog(y_, n0_, mo)
    rows.append(dict(modulation=n, noiseless_sign_correct=bool(np.all((l_clean < 0) == (b == 1))),
                     frac_sign_errors_12dB=float(np.mean((l_noisy < 0) != (b == 1))),
                     mean_LLR_given_bit0=float(l_noisy[b == 0].mean()), mean_LLR_given_bit1=float(l_noisy[b == 1].mean())))
display(pd.DataFrame(rows))'''),
    md("## 4. Reliability differs per bit position (16-QAM, Gray)"),
    code('''mo = Modem("16QAM"); b = rng.integers(0, 2, 4 * 100000); y_, n0_ = ch.awgn(mo.modulate(b), 12.0, rng)
L = llr_maxlog(y_, n0_, mo).reshape(-1, 4); B = b.reshape(-1, 4)
fig, ax = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
for k in range(4):
    ax[k].hist(L[B[:, k] == 0, k], bins=80, alpha=0.6, label="bit 0"); ax[k].hist(L[B[:, k] == 1, k], bins=80, alpha=0.6, label="bit 1")
    ax[k].set_title(f"bit {k} ({'sign/MSB' if k % 2 == 0 else 'inner/LSB'} of axis)\\nsign err = {np.mean((L[:, k] < 0) != (B[:, k] == 1)):.3f}", fontsize=9)
ax[0].legend(); plt.show()
print("The 2nd bit of every axis (inner bit) is less reliable: its two decision regions are closer together.")'''),
    md("## 5. LLR magnitude follows the channel quality"),
    code('''mo = Modem("QPSK"); b = rng.integers(0, 2, 2 * 50000); x = mo.modulate(b); ids = np.arange(len(x))
y, h, n0 = ch.rayleigh_block_fading(x, 10.0, ids, len(x), rng); xh, gain = zf_equalize(y, h)
L_zf = llr_maxlog(xh, n0 / gain, mo).reshape(-1, 2); L_naive = llr_maxlog(xh, n0, mo).reshape(-1, 2)   # naive: ignores |h|
B = b.reshape(-1, 2); bins = pd.cut(abs(h) ** 2, [0, 0.05, 0.2, 0.5, 1, 2, 10])
out = pd.DataFrame({"|h|^2": bins, "mean |LLR| (with CSI)": abs(L_zf[:, 0]), "bit err": (L_zf[:, 0] < 0) != (B[:, 0] == 1)}).groupby("|h|^2", observed=True).mean()
display(out)
print("With CSI the weak-channel bits get small |LLR| although their hard-decision error rate is high; ignoring CSI (N0 instead of N0/|h|^2)\\nwould give them large, *wrong-confidence* LLRs and degrade the Viterbi decoder - this is why the demapper receives the equaliser gain.")'''),
])

# ---------------------------------------------------------------- 07
nb("07_end_to_end_communication.ipynb", [
    md("""# 07 - The complete end-to-end link
**Objective.** Connect every block and follow *one sentence* through the entire chain, then check noiseless exactness for all six MCSs on both channels.

```
text -> Huffman -> conv. encoder (K=7, 133/171) -> puncturing -> Gray QAM -> AWGN | Rayleigh block fading
     -> ZF equaliser -> max-log LLR -> depuncturing -> soft Viterbi -> Huffman decoder -> text
```
**Assumptions.** One sentence = one frame; the receiver knows the frame's number of information bits (side information, needed for trellis termination); no interleaver (not part of the specified chain); perfect CSI; each frame uses whole RBs of 84 symbols."""),
    code(SETUP),
    code('''from task1_conventional.dataset import load_split
from task1_conventional.huffman import HuffmanCodec
from task1_conventional.communication_system import MCS_TABLE, ConventionalLink, Frames
from task1_conventional import convolutional as cv
from task1_conventional.metrics import bler
train, test = load_split("train", ROOT / "data"), load_split("test", ROOT / "data")
codec = HuffmanCodec.from_corpus(train)'''),
    md("## 1. One sentence, stage by stage (MCS4: 16-QAM 3/4, AWGN, 14 dB)"),
    code('''from task1_conventional.modulation import Modem
from task1_conventional import channel as ch
from task1_conventional.equalizer import zf_equalize
from task1_conventional.demapper import llr_maxlog
from task1_conventional.viterbi import viterbi_single
mcs = MCS_TABLE[3]; modem = Modem(mcs.modulation); rng = np.random.default_rng(7)
s = test[3]; print("1 text          :", s)
info = codec.encode(s); print(f"2 Huffman       : {len(info)} bits")
mother = cv.encode_mother(info); print(f"3 conv. encoder : {len(mother)} bits = 2*({len(info)}+6) (rate 1/2 incl. 6 tail bits)")
tx = cv.puncture(mother, mcs.rate); print(f"4 puncturing    : {len(tx)} bits (rate {mcs.rate}; info/coded = {len(info)/len(tx):.4f})")
sym = modem.modulate(tx); print(f"5 modulation    : {len(sym)} {mcs.modulation} symbols ({len(tx)} bits, {(-len(tx)) % modem.bps} padding bit(s))")
y, n0 = ch.awgn(sym, 14.0, rng); print(f"6 channel       : AWGN, Es/N0 = 14 dB, N0 = {n0:.4f}")
xh, gain = zf_equalize(y, np.ones_like(y)); llr = llr_maxlog(xh, n0 / gain, modem)[:len(tx)]
print(f"7 ZF + max-log  : {len(llr)} LLRs, raw channel bit errors = {np.sum((llr < 0) != (tx == 1))}")
dep = cv.depuncture(llr, mcs.rate, len(mother)); print(f"8 depuncturing  : {len(dep)} soft values ({np.sum(dep == 0)} erasures)")
dec = viterbi_single(dep, len(info)); print(f"9 Viterbi       : info bit errors = {np.sum(dec != info)}")
print("10 Huffman dec. :", codec.decode(dec)); print("   identical    :", codec.decode(dec) == s)'''),
    md("## 2. Noiseless exactness: all MCS x both channels"),
    code('''fr = Frames.build(test[:300], codec); rows = []
for m in MCS_TABLE:
    for chn in ("awgn", "rayleigh"):
        res, txt = ConventionalLink(codec, m, chn).run(fr, None, np.random.default_rng(0))
        rows.append(dict(MCS=m.name, channel=chn, info_bit_errors=int(res.bit_errors.sum()), BLER=bler(fr.sentences, txt)))
t = pd.DataFrame(rows); display(t); assert (t.info_bit_errors == 0).all() and (t.BLER == 0).all()
print("All noiseless transmissions reproduce the text exactly.")'''),
    md("## 3. How a sentence degrades with SNR (MCS4, AWGN) - first look at the cliff"),
    code('''link = ConventionalLink(codec, MCS_TABLE[3], "awgn"); one = Frames.build([test[3]], codec)
for snr in (6, 9, 10, 11, 12, 14):
    res, txt = link.run(one, float(snr), np.random.default_rng(100 + snr))
    print(f"{snr:3d} dB | info bit errors {int(res.bit_errors[0]):3d} | {txt[0]}")'''),
    md("## 4. Throughput of the simulator (CPU)"),
    code('''import time
fr = Frames.build(test, codec); link = ConventionalLink(codec, MCS_TABLE[2], "rayleigh"); link.run(fr, 20.0, np.random.default_rng(0))
t0 = time.time(); res, txt = link.run(fr, 20.0, np.random.default_rng(1)); dt = time.time() - t0
print(f"{len(test)} sentences, MCS3 / Rayleigh: {dt:.2f} s  ({len(test) / dt:.0f} sentences/s, vectorised batch Viterbi)")'''),
])

# ---------------------------------------------------------------- 08
nb("08_ber_bler_analysis.ipynb", [
    md("""# 08 - Coded BER and sentence BLER for the six MCSs (Task 1a, 1b)
**Objective.** (a) Validate the simulator with uncoded BER against theory, then (b) evaluate coded BER and sentence BLER for QPSK 1/2, QPSK 3/4, 16-QAM 1/2, 16-QAM 3/4, 64-QAM 2/3, 64-QAM 3/4 over AWGN and Rayleigh block fading.

**Definitions.**
* *Coded BER* = fraction of wrong **information bits** after Viterbi decoding (the Huffman bits).
* *Sentence BLER* = fraction of sentences whose reconstructed text differs from the transmitted text.
* Each Monte-Carlo point sends the same test sentences several times with fresh noise/fading realisations; every point has a fixed seed so results are reproducible. 95 % Wilson confidence intervals are stored in the CSV.

> Results come from `python main.py --experiment mcs --mode quick|full`. The cell below states which mode is loaded."""),
    code(SETUP),
    code(MODE_CELL),
    md("## 1. Validation first: uncoded BER vs theory (Task 1a)"),
    code('''unc = pd.read_csv(TAB / f"{MODE}_uncoded_ber.csv")
plotting.plot_uncoded(unc, MODE, FIG / f"{MODE}_uncoded_ber.png"); plt.show()
unc["rel_err_%"] = 100 * (unc.ber_sim / unc.ber_theory - 1)
print("points:", len(unc), "| within 95% CI or 5 %:", int(unc.inside_ci_or_5pct.sum()), "| max |rel. error| for points with >=400 errors: %.2f %%" % unc[unc.n_errors >= 400]["rel_err_%"].abs().max())
display(unc[["modulation", "channel", "ebn0_db", "n_bits", "n_errors", "ber_sim", "ber_theory", "rel_err_%"]].iloc[::4])
assert unc.inside_ci_or_5pct.all(), "validation failed - do not trust the coded results"'''),
    md("Validation passed (BPSK and QPSK over AWGN, BPSK over Rayleigh with ZF) - the noise scaling, fading generator, equaliser and BER counting are correct, so the coded experiments can be trusted."),
    md("## 2. Coded BER and sentence BLER vs SNR"),
    code('''figs = plotting.plot_ber_bler(sweep, MODE, str(FIG / MODE)); plt.show()'''),
    md("### AWGN vs Rayleigh, per MCS"),
    code('''for k in ("coded_ber_awgn_vs_rayleigh", "bler_awgn_vs_rayleigh"): display(figs[k])'''),
    md("## 3. Quantitative summary from the data"),
    code('''def snr_at(d, col, target):
    """smallest SNR (linear interpolation in the SNR grid) at which `col` first falls to <= target"""
    d = d.sort_values("snr_db"); s, v = d.snr_db.values, d[col].values
    k = np.flatnonzero(v <= target)
    if len(k) == 0: return np.nan
    k = k[0]
    if k == 0: return float(s[0])
    return float(s[k - 1] + (v[k - 1] - target) / (v[k - 1] - v[k]) * (s[k] - s[k - 1]))
rows = []
for m in sorted(sweep.mcs.unique()):
    r = dict(MCS=plotting.LABEL[m])
    for chn in ("awgn", "rayleigh"):
        d = sweep[(sweep.channel == chn) & (sweep.mcs == m)]
        r[f"SNR@BLER=10% {chn} [dB]"] = snr_at(d, "bler", 0.10); r[f"SNR@BLER=1% {chn} [dB]"] = snr_at(d, "bler", 0.01)
    rows.append(r)
summary = pd.DataFrame(rows).set_index("MCS").round(2); display(summary); summary.to_csv(TAB / f"{MODE}_snr_thresholds.csv")
a, r_ = summary["SNR@BLER=10% awgn [dB]"], summary["SNR@BLER=10% rayleigh [dB]"]
print("\\nSNR needed for BLER<=10 %: Rayleigh minus AWGN per MCS [dB]:", (r_ - a).round(1).tolist())
print("Spacing of AWGN thresholds between consecutive MCSs [dB]:", np.diff(a.values).round(1).tolist())'''),
    md("""### Reading the plots
* **AWGN.** Every curve is a very steep "waterfall" (a few dB wide): above the threshold SNR the sentence is almost always perfect, below it almost always lost. Higher-order/higher-rate MCSs are shifted right by the amounts printed above.
* **Rayleigh block fading.** The curves are far shallower and need tens of dB more SNR for the same BLER. With one fade per RB and no interleaver, a sentence that falls into a deep-faded RB loses a burst of bits that the convolutional code cannot repair; the outage probability of a Rayleigh channel decays only like $1/\\text{SNR}$ (10 dB per decade), which is exactly the slope visible in the BLER curves at high SNR.
* **Coded BER vs BLER.** BLER is much larger than BER where errors are bursty (see notebook 09).

## 4. Statistical reliability"""),
    code('''s = sweep.copy(); s["resolution"] = 1 / s.n_frames
low = s[(s.bler > 0) & (s.n_block_errors < 10)]
print(f"{len(s)} points; frames per point: AWGN {s[s.channel == 'awgn'].n_frames.iloc[0]}, Rayleigh {s[s.channel == 'rayleigh'].n_frames.iloc[0]}")
print("smallest non-zero BLER resolvable: AWGN %.1e, Rayleigh %.1e" % (1 / s[s.channel == 'awgn'].n_frames.iloc[0], 1 / s[s.channel == 'rayleigh'].n_frames.iloc[0]))
print(f"points with 1-9 block errors (low statistical confidence, relative CI > ~60 %): {len(low)} of {len(s)}")
print("BLER = 0 means 'no block error observed in N frames'; the 95 % upper bound is about 3/N.")
if MODE != "full": print("\\n*** QUICK MODE: only", s.n_frames.iloc[0], "frames per point - results are for debugging, not for claims. ***")
display(s[["channel", "mcs", "snr_db", "n_frames", "n_block_errors", "bler", "bler_ci_lo", "bler_ci_hi"]].query("0 < bler < 1").head(12))'''),
])

# ---------------------------------------------------------------- 09
nb("09_text_quality_analysis.ipynb", [
    md("""# 09 - Text quality and the cliff effect (Task 1c)
**Objective.** Measure *what the user actually reads*: BLEU, word accuracy and sentence-embedding cosine similarity of the reconstructed text versus SNR, and explain the cliff effect with measured numbers.

## Metrics (as implemented in `task1_conventional/metrics.py`)
* **BLEU** - sacreBLEU sentence BLEU (up to 4-grams, 13a tokenisation, exponential smoothing), scaled to 0-1; identical sentences = 1.
* **Word accuracy** - minimum-edit-distance alignment of reference and hypothesis words (whitespace tokens). With $H$ hits, $S$ substitutions, $D$ deletions, $I$ insertions and $N=H+S+D$ reference words: **word accuracy $=H/N$**. A substituted or deleted word is a miss; an inserted word is not a hit and does not change $H/N$ but is visible in $\\text{WER}=(S+D+I)/N$.
* **Cosine similarity of sentence embeddings** - the active backend is printed below. Default: mean of pretrained spaCy `en_core_web_md` word vectors (a *pretrained embedding*, not a lexical-overlap metric, but weaker than a transformer sentence encoder; unknown/garbled tokens count as zero vectors)."""),
    code(SETUP),
    code(MODE_CELL),
    code('''from task1_conventional.metrics import SentenceEmbedder
print("embedding backend used for the stored results:", ex.json.loads((ROOT / "results/logs" / f"{MODE}_config.json").read_text())["embedder"])'''),
    md("## 1. Quality vs SNR"),
    code('''figs = plotting.plot_text_quality(sweep, MODE, str(FIG / MODE)); plt.show()'''),
    md("## 2. The cliff effect with numbers"),
    code('''def snr_at(d, col, target, rising=False):
    d = d.sort_values("snr_db"); s, v = d.snr_db.values, d[col].values
    k = np.flatnonzero(v >= target) if rising else np.flatnonzero(v <= target)
    if len(k) == 0: return np.nan
    k = k[0]
    if k == 0: return float(s[0])
    return float(s[k - 1] + (v[k - 1] - target) / (v[k - 1] - v[k]) * (s[k] - s[k - 1]))
rows = []
for chn in ("awgn", "rayleigh"):
    for m in sorted(sweep.mcs.unique()):
        d = sweep[(sweep.channel == chn) & (sweep.mcs == m)]
        s90, s10, s1 = snr_at(d, "bler", 0.9), snr_at(d, "bler", 0.1), snr_at(d, "bler", 0.01)
        rows.append(dict(channel=chn, MCS=plotting.LABEL[m], SNR_BLER90=s90, SNR_BLER10=s10, SNR_BLER1=s1, cliff_width_90_to_10_dB=s10 - s90, cliff_width_10_to_1_dB=s1 - s10))
cliff = pd.DataFrame(rows).round(2); display(cliff); cliff.to_csv(TAB / f"{MODE}_cliff_widths.csv", index=False)
aw = cliff[cliff.channel == "awgn"]
print("AWGN: BLER falls from 90 %% to 10 %% within %.1f-%.1f dB (median %.1f dB) for the six MCSs." % (aw.cliff_width_90_to_10_dB.min(), aw.cliff_width_90_to_10_dB.max(), aw.cliff_width_90_to_10_dB.median()))'''),
    md("### One MCS in detail: BER, BLER and text quality on the same SNR axis"),
    code('''for chn in ("awgn", "rayleigh"):
    plotting.plot_cliff(sweep, 4, MODE, FIG / f"{MODE}_cliff_mcs4_{chn}.png", chn); plt.show()
d = sweep[(sweep.channel == "awgn") & (sweep.mcs == 4)].sort_values("snr_db")
zone = d[(d.bler > 0.001) & (d.bler < 0.999)][["snr_db", "raw_ber", "coded_ber", "bler", "bleu", "word_acc", "cos"]]
display(zone.round(4))'''),
    md("""### Why is there a cliff? (explanation, with the measured evidence above)
1. **Viterbi decoding has a sharp threshold.** Above a critical $E_s/N_0$ the channel errors are sparse and the code corrects *all* of them (BER $\\approx0$); below it, error events become long bursts and the decoder fails over many bits at once. The coded-BER curve therefore drops over only a few dB (table above: raw channel BER changes slowly while coded BER collapses).
2. **Source coding amplifies errors.** The Huffman stream has almost no redundancy left, and a prefix code loses synchronisation after a single wrong bit - the remainder of the sentence becomes garbage. So *any* residual bit error makes the sentence wrong (BLER) and a burst of errors destroys most of it (BLEU, word accuracy $\\to0$).
3. **Sentence metrics are thresholded versions of the BER.** If bit errors were independent, BLER$\\approx 1-(1-\\text{BER})^{N}$ for $N\\approx$ 350 bits would already be a steep function of BER (a 10x BER change, $10^{-4}\\to10^{-3}$, moves BLER from ~3 % to ~30 %); real Viterbi errors come in bursts, which makes the transition even sharper in SNR.
4. Hence a **small SNR change (a few dB) flips the user experience from perfect to unusable**, and the SNR margin needed for reliable text is much larger than the margin for a given BER. Compare the *cliff width* columns above with the spacing between the MCS thresholds printed in notebook 08.
5. In Rayleigh fading the cliff is *smeared* (much wider) because the SNR of each RB is random; the sentence succeeds only when none of its RBs fades deeply.

### The embedding metric is not the same cliff
Cosine similarity degrades **more gracefully** and has a high floor. Let us measure that floor instead of guessing:"""),
    code('''import itertools
from task1_conventional.dataset import load_split
test = load_split("test", ROOT / "data"); rng = np.random.default_rng(0)
emb = SentenceEmbedder(); idx = rng.permutation(len(test))[:400]
rand_cos = emb.cosine([test[i] for i in idx[:200]], [test[i] for i in idx[200:]])
print("embedding backend:", emb.backend)
print("cosine similarity between RANDOM UNRELATED test sentences: mean %.3f, 5-95 %% range [%.3f, %.3f]" % (rand_cos.mean(), *np.percentile(rand_cos, [5, 95])))
low = sweep[(sweep.snr_db == sweep.snr_db.min()) & (sweep.channel == "awgn")][["mcs", "bleu", "word_acc", "cos"]]
print("At the lowest AWGN SNR in the sweep (complete garbage):"); display(low.round(3))
print("-> cosine similarity of garbled text stays near the unrelated-sentence baseline (>0), while BLEU / word accuracy go to ~0.")'''),
    md("""**Interpretation.** Mean-pooled word vectors share a large common component, so even unrelated sentences have a clearly positive cosine; garbled text is no worse than that. Cosine similarity is therefore only informative *above* this baseline and should be read together with BLEU/word accuracy. A transformer sentence encoder (`all-MiniLM-L6-v2`, used automatically when available) has a lower baseline.

## 3. Examples of reconstructed text (MCS4, AWGN)"""),
    code('''from task1_conventional.dataset import load_split
from task1_conventional.huffman import HuffmanCodec
from task1_conventional.communication_system import MCS_TABLE, ConventionalLink, Frames
from task1_conventional.metrics import text_quality
codec = HuffmanCodec.from_corpus(load_split("train", ROOT / "data")); fr = Frames.build(test[10:13], codec)
link = ConventionalLink(codec, MCS_TABLE[3], "awgn")
for snr in (8.0, 9.0, 10.0, 12.0):
    res, hyp = link.run(fr, snr, np.random.default_rng(int(snr * 10)))
    q = text_quality(fr.sentences, hyp, emb)
    print(f"--- {snr} dB: BER={res.ber:.3f}")
    for a, b, bl, wa, c in zip(fr.sentences, hyp, q["bleu"], q["word_acc"], q["cos"]):
        print(f"  REF: {a}\\n  HYP: {b}\\n       BLEU={bl:.2f} word-acc={wa:.2f} cos={c:.2f}")'''),
])

# ---------------------------------------------------------------- 10
nb("10_adaptive_modulation_coding.ipynb", [
    md("""# 10 - Adaptive modulation and coding (Task 1d)
**Objective.** Use the simulated BLER/BER/text-quality tables to build an AMC rule: *choose the highest-rate MCS whose sentence BLER is $\\le10\\,\\%$*.

## Mechanism
1. **SNR input/estimation** - the receiver reports $E_s/N_0$ (here: the true value, or an estimate from known pilots, see section 4).
2. **Look-up** - read the row of the LUT at the largest grid SNR $\\le$ the reported SNR (conservative: BLER decreases with SNR, so a lower grid point can only over-estimate BLER).
3. **Threshold check** - keep MCSs with BLER $\\le 0.10$.
4. **Selection** - among them take the highest nominal spectral efficiency $\\eta=m\\cdot R$ [info bit/channel use]: 1, 1.5, 2, 3, 4, 4.5 for MCS 1-6.
5. **Fallback** - if none qualifies, use the most robust MCS (MCS1, QPSK 1/2) and flag *outage*.
6. **Throughput** - reported both as $\\eta$ and as goodput $\\eta(1-\\text{BLER})$.

**No value is typed in by hand:** the LUT is the Monte-Carlo sweep of notebook 08."""),
    code(SETUP),
    code(MODE_CELL),
    code('''from task1_conventional.amc import AMC, build_lut, amc_with_estimation, estimate_snr_db
from task1_conventional.communication_system import MCS_TABLE
out = ex.run_amc(MODE, ROOT)                          # (re)builds LUT CSVs from the sweep
luts = {c: out[c][0] for c in ("awgn", "rayleigh")}; curves = {c: out[c][1] for c in ("awgn", "rayleigh")}'''),
    md("## 1. Look-up tables (SNR, MCS, BER, BLER, text similarity)"),
    code('''print("LUT columns:", list(luts["awgn"].columns))
display(luts["awgn"].query("snr_db in [8, 10, 12, 14, 16, 18]").round(4).head(18))
for c in luts: print(f"saved results/tables/{MODE}_amc_lut_{c}.csv  ({len(luts[c])} rows)")'''),
    md("## 2. Selection in action"),
    code('''amc = AMC(luts["awgn"], 0.10)
for snr in (-5, 0, 3.2, 6, 9.5, 13, 17.4, 25):
    r = amc.select(snr); print(f"SNR {snr:6.1f} dB -> MCS{r['mcs']}  eta={r['efficiency']:.1f}  BLER(table)={r['bler']}  outage={r['outage']}")'''),
    md("## 3. AMC curves (AWGN and Rayleigh LUTs)"),
    code('''figs = plotting.plot_amc(curves, MODE, str(FIG / MODE)); plt.show()'''),
    code('''for chn, cv in curves.items():
    sw = cv[cv.mcs.diff().fillna(0) != 0][["snr_db", "mcs", "efficiency", "bler"]]
    print(f"--- {chn}: switching points (SNR grid where the selected MCS changes)"); display(sw.round(4))
    print("outage SNRs (no MCS meets BLER<=10%):", cv[cv.outage].snr_db.tolist())
    print("max BLER of selected MCS over non-outage SNRs: %.4f  (target 0.10)" % cv[~cv.outage].bler.max())
    assert (cv[~cv.outage].bler <= 0.10 + 1e-12).all()'''),
    md("""### Which LUT should Task 3 use?
In Task 3 every RB $n$ of user $u$ has its own *instantaneous* SNR $\\gamma_{u,n}$ (path loss, shadowing and one Rayleigh fade, constant within the RB). Conditioned on that fade the RB is an **AWGN channel** at $\\gamma_{u,n}$, so the **AWGN LUT** is the right table for per-RB link adaptation. The Rayleigh LUT here answers a different question - the BLER averaged over fading at a given *average* SNR (what a long-term link-adaptation loop would see)."""),
    md("## 4. Robustness: selecting with an *estimated* SNR (AWGN)\nThe estimator averages $|y-x|^2$ over $N_p$ known pilot symbols; its error shrinks like $1/\\sqrt{N_p}$. Selection uses the estimate, the outcome is scored with the table at the **true** SNR."),
    code('''est = out["estimation"]
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for n, g in est.groupby("n_pilots"):
    ax[0].plot(g.snr_db, g.mean_efficiency, label=f"{n} pilots"); ax[1].semilogy(g.snr_db, g.p_bler_violation.replace(0, np.nan), label=f"{n} pilots"); ax[2].plot(g.snr_db, g.mean_selected_bler, label=f"{n} pilots")
base = curves["awgn"]; ax[0].step(base.snr_db, base.efficiency, "k--", where="post", label="perfect SNR"); ax[0].set_title("mean efficiency"); ax[0].legend()
ax[1].set_title("P(selected MCS violates BLER<=10%)"); ax[2].set_title("mean BLER of selected MCS"); ax[2].axhline(0.1, color="k", ls=":")
for a in ax: a.set_xlabel("true SNR [dB]")
fig.tight_layout(); fig.savefig(FIG / f"{MODE}_amc_snr_estimation.png", dpi=150); plt.show()
print(est.groupby("n_pilots")[["mean_efficiency", "p_bler_violation"]].mean().round(4))'''),
    md("""**Reading it.** With few pilots the estimate is noisy; because the BLER cliff is so steep (notebook 09), an SNR error of only 1-2 dB just below a switching point can select an MCS that violates the 10 % target. More pilots (or a back-off margin of ~1-2 dB) trade a little spectral efficiency for a lower violation probability."""),
    md("## 5. Summary table for the report"),
    code('''thr = []
for chn, cv in curves.items():
    for m in MCS_TABLE:
        s = cv[(cv.mcs == m.index) & (~cv.outage)]
        thr.append(dict(channel=chn, MCS=m.name, efficiency=m.efficiency, first_SNR_selected_dB=s.snr_db.min() if len(s) else np.nan))
thr = pd.DataFrame(thr); display(thr.pivot(index="MCS", columns="channel", values="first_SNR_selected_dB")); thr.to_csv(TAB / f"{MODE}_amc_switching_snr.csv", index=False)'''),
])
print("notebooks written to", OUT)
