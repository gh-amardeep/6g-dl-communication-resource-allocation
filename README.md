# Exercise 1 – Task 1: Conventional End-to-End Text Link

Bit-level downlink for English sentences: **text → character-level Huffman → rate-1/2 K=7 (133,171) convolutional code
(punctured to 2/3, 3/4) → Gray QPSK/16-QAM/64-QAM → AWGN / Rayleigh block fading → ZF equaliser → max-log LLR demapper →
depuncturing → soft-decision Viterbi → Huffman decoder → text.**

Scope of the Task 1 sections: Phase 1 (environment) and Task 1 (a)–(d). Task 2 (autoencoder) and Task 3 (resource allocation) are documented in their own
sections at the end of this file; the modules are independent of any learning code, so they can be reused later
(`AMCController` is the rule Task 3 will apply per RB).

## Progress checklist

- [x] Environment, `.venv`, Jupyter kernel `Python (Exercise1)`, `requirements.txt`
- [x] Dataset: 25,932 train / 2,000 test sentences (no overlap, split by document)
- [x] Huffman encoding/decoding (lossless, UNK handling)
- [x] Convolutional coding at rates 1/2, 2/3, 3/4
- [x] Gray QPSK / 16-QAM / 64-QAM
- [x] AWGN and Rayleigh block-fading channels
- [x] ZF equalisation
- [x] Max-log soft LLR demapping
- [x] Soft-decision Viterbi decoding
- [x] Task 1(a): uncoded BPSK/QPSK/Rayleigh BER matches theory (63/63 points within 95 % CI or 5 %)
- [x] Task 1(b): six MCSs, coded BER + sentence BLER, AWGN and Rayleigh (full mode)
- [x] Task 1(c): BLEU, word accuracy, embedding cosine similarity, cliff-effect analysis
- [x] Task 1(d): AMC look-up tables from simulation + AMC figures
- [x] All 11 notebooks execute top-to-bottom in the `Python (Exercise1)` kernel
- [x] 81 pytest tests pass
- [x] Task 2 (autoencoder physical layer): see section "Exercise 1 – Task 2" below (90 tests at Checkpoint 02)
- [x] Task 3 (DNN multi-user resource allocation): see section "Exercise 1 – Task 3" below
- [ ] Not done (out of scope): slides, paper, Exercise 2

## System architecture

```
 text ─► Huffman enc ─► conv enc (K=7,133/171, 6 tail bits) ─► puncture (1/2, 2/3, 3/4)
                                                                      │
                                                            Gray QPSK / 16-QAM / 64-QAM (Es=1)
                                                                      │
                                            AWGN  or  Rayleigh block fading (h~CN(0,1) per RB of 84 symbols)
                                                                      │
 text ◄─ Huffman dec ◄─ soft Viterbi ◄─ depuncture ◄─ max-log LLR ◄─ ZF equaliser (x̂=y/h, gain=|h|²)
```

## Folder structure

```
notebooks/            00…10 notebooks (educational, executed, outputs saved)
task1_conventional/   reusable modules (see below)
data/{train,test,processed,raw}   corpus, Huffman codebook, dataset_stats.json
results/{figures,tables,logs}     every output; files are prefixed quick_ or full_
experiments/          build_notebooks.py (generates notebooks), run_notebooks.py (executes them)
tests/                pytest suite
main.py               terminal runner
```

## Environment setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # see the spaCy model note inside the file
python -m ipykernel install --user --name exercise1 --display-name "Python (Exercise1)"
jupyter lab                            # choose kernel "Python (Exercise1)"
```

## Notebook sequence

Run in order: 00 environment → 01 dataset → 02 Huffman → 03 convolutional/Viterbi → 04 modulation → 05 channels →
06 ZF + LLR → 07 end-to-end → 08 BER/BLER → 09 text quality → 10 AMC. Notebooks 08–10 read the CSVs produced by
`main.py` and use **full** results when present (`experiments.best_available_mode`), otherwise quick results with a
visible "QUICK MODE" warning. To re-execute all: `python experiments/run_notebooks.py`.

## Modules

| Module | Content |
|---|---|
| `dataset.py` | Brown corpus download (offline fallback: `data/raw/brown.zip`), cleaning, split by document |
| `huffman.py` | frequencies, tree, codebook save/load, UNK symbol, entropy/efficiency |
| `convolutional.py` | encoder, puncturing patterns, depuncturing (erasures = LLR 0), termination |
| `viterbi.py` | soft-decision 64-state Viterbi, traceback, terminated trellis |
| `modulation.py` | Gray BPSK/QPSK/16/64-QAM, mapper/demapper, padding of incomplete groups |
| `channel.py` | AWGN, Rayleigh block fading, SNR/Eb/N0/Es/N0 conversions |
| `equalizer.py` | ZF with magnitude floor against deep fades |
| `demapper.py` | max-log LLR (LLR>0 ⇒ bit 0), noise variance scaled by 1/\|h\|² |
| `communication_system.py` | MCS table and the full link (one sentence = one frame) |
| `metrics.py` | BER, BLER, BLEU (sacreBLEU), word accuracy (edit-distance alignment), embedding cosine |
| `amc.py` | LUT-driven AMC controller, fallback, throughput, SNR-estimation test |
| `experiments.py`, `plotting.py` | quick/full drivers with checkpointing, all figures |

Puncturing: mother rate 1/2 sends 2 coded bits per info bit. Pattern 2/3 keeps 3 of 4 coded bits per 2 info bits
(rate 2/3); pattern 3/4 keeps 4 of 6 per 3 info bits (rate 3/4). Fewer transmitted bits for the same information ⇒ higher
rate, less redundancy, worse error correction. Es/N0 = Eb/N0 · (bits/symbol) · R.

## Running experiments

```bash
python main.py --experiment dataset
python main.py --experiment ber  --mode quick      # Task 1(a)
python main.py --experiment mcs  --mode quick      # 1(b)+(c) sweep
python main.py --experiment text-quality --mode quick
python main.py --experiment amc  --mode quick      # 1(d)
python main.py --experiment all  --mode full       # final results (~25 min on 2 CPU cores, resumable)
python -m pytest                                    # tests
```

*Quick* = 200 sentences, coarse grids (debug only). *Full* = 2,000 test sentences × 3 repetitions (AWGN, 6,000 frames per
point) or × 5 (Rayleigh, 10,000 frames), 0.5 dB AWGN grid, 2 dB Rayleigh grid. Seeds and configs are saved in
`results/logs/{quick,full}_config.json`. The sweep checkpoints row by row.

## Outputs

* `results/tables/full_*.csv`: `uncoded_ber`, `mcs_sweep` (BER, BLER with CI, BLEU, word acc, cosine), `amc_lut_{awgn,rayleigh}`,
  `amc_curve_*`, `snr_thresholds`, `cliff_widths`, `amc_switching_snr`, `amc_snr_estimation`
* `results/figures/full_*.png`: uncoded BER, coded BER, BLER, AWGN-vs-Rayleigh, BLEU/word-acc/cosine vs SNR, cliff plots, AMC plots
* `results/tables/quick_*`, `quick_*.png`: debugging-mode outputs, **not final**

## Results summary (from `results/tables/full_*.csv`)

* Dataset: Brown corpus, 30–120 printable-ASCII characters per sentence; 25,932 train / 2,000 test; 84 training characters
  (+UNK = 85 codewords); mean 73 chars/sentence. Huffman: entropy 4.470, average length 4.498 bit/char, efficiency 99.4 %
  (7 bit/char fixed-length).
* SNR for BLER = 10 % (interpolated, Es/N0): AWGN 2.4 / 5.0 / 7.3 / 10.9 / 14.3 / 15.9 dB for MCS 1–6;
  Rayleigh 16.1 / 17.8 / 19.3 / 21.5 / 24.0 / 25.5 dB.
* Cliff: in AWGN BLER drops 90 %→10 % within about 2.0–2.4 dB (and 10 %→1 % in ~1 dB); in Rayleigh the same fall needs
  ~14 dB and 10 %→1 % ~10 dB, because deep fades (not average noise) cause the errors.
* AMC (BLER ≤ 10 %): AWGN switches up at 2.5, 5.0, 7.5, 11.0, 14.5, 16.0 dB (grid values); Rayleigh at 18, 20, 22, 24, 26 dB for
  MCS 2–6 (grid step 2 dB). Below the threshold the controller falls back to MCS 1 and flags an outage.

## Assumptions and limitations

1. **SNR** is Es/N0 (average for Rayleigh), matching γ = pg/(N0·B·F) of the assignment. Eb/N0 is used only for uncoded validation.
2. **Frame = one sentence.** The receiver knows the information-bit count (error-free side information); needed for termination
   and depuncturing. No CRC and no bit interleaver (neither is in the specified chain).
3. **Block fading**: one independent h ~ CN(0,1) per resource block of 84 symbols; perfect CSI at the receiver. Without an
   interleaver, coded Rayleigh performance is dominated by fades hitting a whole codeword, hence the high required SNR.
4. ZF uses a magnitude floor on |h| against near-zero gains.
5. **Embedding metric**: the Hugging Face / sentence-transformers hosts are not reachable from this environment, so the
   similarity uses spaCy `en_core_web_md` **mean-pooled pretrained word vectors**. These are real pretrained embeddings but static
   word vectors, not a contextual sentence encoder (no word-order information); absolute cosine values are therefore
   higher/less discriminating than with e.g. MiniLM. Replace `SentenceEmbedder` if you can download one.
6. Dataset: 1960s American English (Brown), sentence length-filtered; vocabulary is not representative of modern text. Test
   characters unseen in training: none (UNK path is tested separately).
7. BLER resolution is 1/6000 (AWGN) and 1/10000 (Rayleigh); values below that are reported as 0. Spectral efficiency in AMC is the
   nominal bits/symbol × rate (information bits per modulation symbol, tail/side-information overhead ignored); goodput = η·(1−BLER).
8. AMC reads the LUT at the largest grid SNR not above the input SNR (conservative).
9. Results come from one master seed (20241111); confidence intervals for BLER are stored in the CSV.


---

# Exercise 1 – Task 2: Autoencoder Physical Layer

Small PyTorch (CPU) autoencoders trained end-to-end over a channel. Task 1 code is only *reused*
(`noise_variance`, `ebn0_to_esn0_db`, `complex_gaussian`, `rayleigh_gains`, `zf_equalize`, `get_modem("16QAM")`), never modified.

## Files

```
task2_autoencoder/{autoencoder,baselines,experiments}.py   model, Hamming/BPSK/16-QAM baselines, training + evaluation
notebooks/Exercise1_Task2/01_autoencoder_7_4_AWGN.ipynb
notebooks/Exercise1_Task2/02_autoencoder_7_4_Rayleigh.ipynb
notebooks/Exercise1_Task2/03_autoencoder_2_4_constellation.ipynb
models/autoencoder/{ae74_awgn,ae74_rayleigh,ae24_awgn}.pt + *_config.json
results/exercise1_task2/figures/   5 PNG        results/exercise1_task2/tables/   7 CSV
tests/test_ae_task2.py                                      9 Task 2 tests
experiments/build_notebooks_task2.py                        generates the notebooks (edit text there, rebuild, re-execute)
```

## How to run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU build; other requirements as above
python -m pytest tests/test_ae_task2.py
jupyter lab        # kernel "Python (Exercise1)", open notebooks/Exercise1_Task2/ and run 01 -> 02 -> 03
```

The notebooks load the saved weights/CSVs when present. Retraining uses the settings below (seed 20241113).
`experiments/run_notebooks.py` covers only the Task 1 notebooks; Task 2 notebooks were executed with nbclient
(kernel `exercise1`, working directory = the notebook folder).

## Settings

| Item | (7,4) AWGN | (7,4) Rayleigh | (2,4) AWGN |
|---|---|---|---|
| Model | one-hot(16) → Dense16+ReLU → Dense n → power normalisation → channel → Dense16+ReLU → Dense16 | same; decoder also gets CSI \|h\|² | same, n = 2 (I, Q) |
| Loss / optimiser | cross-entropy on message index; Adam, lr 3e-3, cosine, batch 1024 | same | same |
| Training Eb/N0, steps | 0–8 dB, 6000 | 0–20 dB, 8000 | 4–12 dB, 6000 |
| Evaluation blocks per SNR | 2.5 M | 2 M | 2 M |
| Power normalisation | codebook-level, 1 per real symbol | same | \|x\|² = 1 per complex symbol |

SNR is Eb/N0 per information bit; Es/N0 = Eb/N0 + 10·log10(rate) with rate 4/7 bit per real symbol for (7,4) and
4 bit per complex symbol for (2,4); noise variance per real dimension is N0/2. Rayleigh = one gain h ~ CN(0,1) per
7-symbol block (block fading, no diversity), coherent receiver with perfect CSI, ZF equalisation x̂ = y/h (magnitude floor
1e-3). Baselines: uncoded BPSK, Hamming (7,4) hard decision, Hamming (7,4) ML (correlation over the 16 codewords),
and normalised Gray 16-QAM for (2,4). The autoencoder bit labelling is plain binary (not Gray). BER/BLER use identical
information-bit sequences and identical noise for all methods at each SNR.

## Results (from `results/exercise1_task2/tables/*.csv`)

* **(7,4) AWGN** – Eb/N0 needed for BER 1e-2 / 1e-3: autoencoder 3.64 / 5.53 dB, Hamming-ML 3.33 / 5.31, Hamming hard
  4.55 / 6.63, uncoded BPSK 4.30 / 6.77. Hamming-ML beats BPSK from about 1 dB, the autoencoder from about 2 dB, hard
  decision only from about 6 dB. The autoencoder BLER nearly overlaps Hamming-ML.
* **(7,4) Rayleigh (block fading)** – at 0 / 10 / 20 / 24 dB the BERs are autoencoder 0.192 / 0.030 / 3.25e-3 / 1.31e-3,
  Hamming-ML 0.168 / 0.025 / 2.62e-3 / 1.05e-3, uncoded BPSK 0.146 / 0.023 / 2.48e-3 / 0.98e-3. Uncoded BPSK is best, the
  autoencoder is 14–25 % above Hamming-ML, and hard decision is about equal to the autoencoder. Uncoded BER matches the theory
  0.5(1−√(γ/(1+γ))).
* **(2,4) vs 16-QAM (AWGN)** – Eb/N0 for SER 1e-2 / 1e-3: autoencoder 9.31 / 11.33 dB vs 16-QAM 9.62 / 11.57 (autoencoder
  about 0.3 / 0.2 dB better); for BER 1e-2 / 1e-3: autoencoder 8.57 / 10.83 vs 16-QAM 7.88 / 10.49 (autoencoder about
  0.7 / 0.3 dB worse). Minimum distance: autoencoder 0.599, 16-QAM 0.632 (|x|² = 1).

## Explanations

**1. How the (7,4) autoencoder works.** Each of the 16 four-bit messages is a one-hot vector. The encoder network maps it
to 7 real numbers; these are normalised so that the average power over the 16-codeword codebook is 1 per symbol, so the
network cannot gain SNR by scaling up. The channel adds Gaussian noise (and, in Experiment 2, a Rayleigh gain that the
receiver removes by ZF). The decoder network outputs 16 scores, softmax + cross-entropy trains both networks jointly by
backpropagation through the channel, and the decoded message is the arg-max. In effect it learns a 16-point codebook in R⁷ and a decision rule for it.

**2. Comparison with uncoded BPSK and Hamming.** Uncoded BPSK sends 4 bits in 4 symbols; the (7,4) schemes spend 7 symbols
for the same bits (rate 4/7), which costs energy per bit but adds redundancy. On AWGN, Hamming-ML gains about 1.5 dB over
BPSK at BER 1e-3 (5.31 vs 6.77 dB), the autoencoder about 1.2 dB (5.53 dB), and hard-decision Hamming only 0.1 dB because
discarding the soft values loses most of the coding gain; below about 6 dB hard decision is worse than uncoded BPSK. The
autoencoder uses the soft values and lands close to ML: 0.2 dB behind at BER 1e-3 and 0.3 dB at BER 1e-2.

**3. Effect of Rayleigh fading.** BER falls only slowly with SNR (about one decade per 10 dB, against steep curves in AWGN),
because errors are dominated by deep fades of the single block gain and not by noise: when |h| is small the whole
7-symbol block is lost. With one gain per block and no diversity, the 7-symbol code cannot average across fades, and the
three redundant symbols spend energy that does not help. Hence here uncoded BPSK is best and the coded schemes are worse
at equal Eb/N0 (opposite to AWGN). The autoencoder is 14–25 % above Hamming-ML (hard decision about equal to it) even with CSI at its decoder.

**4. How the (2,4) autoencoder learns its constellation.** The encoder's two outputs are the I and Q coordinates of a
single symbol per message. Under the power constraint |x|² = 1 on average, training pushes the 16 points apart where noise
would otherwise confuse them: this maximises the minimum Euclidean distance and so lowers the symbol error probability,
while the decoder learns the matching decision regions. The learned points (`ae24_learned_constellation.csv`) form an irregular, non-grid pattern: about ten points on an outer
ring (radius ≈1.1–1.3), five at radius ≈0.6–0.8 and one near the origin (see `ae24_constellations.png`), with SER below 16-QAM.

**5. Why the autoencoder does not necessarily beat ML decoding of Hamming (7,4).** ML decoding of a given codebook is the
optimum receiver for that codebook in AWGN, and the Hamming code is already a good 16-codeword code of length 7. The autoencoder
can at best match that decoder for *its own* codebook; it has finite capacity, is trained with stochastic gradient descent
over a mixture of SNRs rather than at each point, and can end in a local optimum. Its codebook may be slightly better or
worse than Hamming's at a given SNR, but here it stayed 0.2–0.3 dB behind Hamming-ML. Note also that the (2,4) autoencoder
wins in SER but loses in BER because its labels are not Gray, so a symbol error often flips several bits.

**6. Limits of longer block lengths.** A message-level autoencoder needs one output class (and one-hot input) per message, 2^k
of them: 16 for k = 4 but about a million for k = 20. The softmax/one-hot layers and the number of training examples needed
grow exponentially, and the network cannot see all codewords during training, so it generalises poorly; ML decoding of
random-like codes is also exponentially complex. Practical long codes need structure (convolutional/LDPC/polar graphs, bitwise
outputs), which is why this exercise stays at k = 4.

## Task 2 limitations

Highest-SNR points have few errors (e.g. AWGN at 8 dB: 93 autoencoder vs 51 Hamming-ML bit errors), so the CIs there are
wide; the other points have more. Rayleigh is block fading only, with perfect CSI. Single master seed. Task 2 notebooks are
executed with outputs embedded; they were not re-executed in later sessions (results unchanged).


---

# Exercise 1 – Task 3: DNN-based multi-user resource allocation

8 users, 16 RBs of 180 kHz, cell radius 500 m (users uniform, d ≥ 35 m), path loss 128.1 + 37.6 log10(d/km) dB, 8 dB shadowing,
Rayleigh fading per user and RB (Task 1 `rayleigh_gains`), N0 = −174 dBm/Hz, NF 9 dB (noise per RB −112.45 dBm), R_min = 1 Mbps,
P_max = 15…40 dBm (1 dB steps). Task 1 and Task 2 were not modified. The assignment PDF was not available in the chat; the
specification used is the Task 3 prompt (parameters, six methods, metrics, AMC).

## Files

```
task3_resource/system.py        parameters, channel gains (noise-normalised a = g/(N0 B F)), rates
task3_resource/allocation.py    methods 1-4 (round-robin, strongest user + WF, MILP, MILP + KKT water-filling)
task3_resource/dnn.py           DNN model, unsupervised training, prediction, save/load
task3_resource/amc_eval.py      sentences per second via the Task 1 AMC class + AWGN LUT
task3_resource/experiments.py   six-method evaluation on identical drops, summary, plots
experiments/run_task3.py        runs the sweep (300 drops x 26 power levels; about 5 min, CPU)   experiments/build_notebooks_task3.py
notebooks/Exercise1_Task3/01..05_*.ipynb   (kernel "Python (Exercise1)"; notebooks only load the sweep results)
models/resource_allocation/dnn_alloc.pt (+ .json)
results/exercise1_task3/figures/  9 PNG        results/exercise1_task3/tables/  CSV files        results/exercise1_task3/user_rates_mbps.npz
tests/test_task3.py             15 tests
```

Run: `python experiments/run_task3.py 300` (writes all CSVs and the six metric plots), then open the notebooks 01 → 05.
Retraining the DNN: `dnn.train(steps=12000, lam=5.0, hidden=512, lr=1e-3, batch=512, save_to=...)` (about 6 min on one CPU core).

## Implementation

* **Problem**: max Σ x_un B log2(1 + p_n a_un) s.t. Σ_u x_un ≤ 1, Σ p_n ≤ P_max, R_u ≥ R_min, x binary, p ≥ 0 (powers in W).
* **1 Round-robin + equal power**: RB n → user n mod 8, p_n = P_max/16.
* **2 Strongest user + water-filling**: arg-max user per RB, p_n = [L − 1/a_n]⁺ with Σp = P_max (no QoS enforcement).
* **3 MILP + equal power**: SciPy `milp` (HiGHS) with the constants r_un = B log2(1 + P_max/16 · a_un); one user per RB and Σ_n r_un x_un ≥ R_min for every user.
  If infeasible, `qos_feasible = False` is reported and a best-effort assignment (most satisfied users, then sum rate) is returned.
* **4 MILP + KKT water-filling**: assignment of method 3; Lagrangian Σ_u (1+μ_u)R_u − ν(Σp − P_max) gives p_n = [L_u − 1/a_n]⁺ with
  L_u = (1+μ_u)B/(ν ln 2). A slack QoS has μ_u = 0 (L_u = L); an active QoS has L_u = λ_u, the level giving exactly R_min.
  So L_u = max(L, λ_u) and the scalar L is found by bisection so that Σp = P_max. If Σ_u P_u^min > P_max the QoS cannot be met for that
  assignment: R_min is guaranteed to the cheapest users that fit the budget and the drop is flagged infeasible.
* **5 DNN**: MLP 129 → 512 → 512 → 144 (403 k parameters), input 10·log10(a) (standardised) plus P_max in dBm (an extra scalar input so one model serves the whole sweep –
  a deviation from "H only"), output assignment logits (softmax over users per RB, hard arg-max at test time; straight-through gradient in training) and power
  fractions (softmax, p_n = P_max · fraction_n). So every RB has exactly one user, p_n ≥ 0 and Σp = P_max. Loss L = −SumRate[Mbps] + λ Σ_u max(0, 1 − R_u)², λ = 5,
  trained 12 000 steps (batch 512, Adam 1e-3, cosine) on freshly generated channels with P_max ~ U(15, 40) dBm; no MILP labels. λ ∈ {5, 10, 20, 60} and hidden size
  256/512 were compared in short trials on held-out drops; λ = 5 with 512 units gave the best sum rate at similar QoS.
* **6 Hybrid**: DNN assignment + the KKT water-filling of method 4.
* **Evaluation**: 300 channel drops, identical for all methods and all power levels. Sum rate, % users with R_u ≥ 1 Mbps (users with no RB count as violations),
  Jain's index per drop (mean), 5th percentile of all user rates (pooled over drops and users), median computation time per drop (CPU, 1 thread; method 4 = MILP + KKT,
  method 6 = DNN + KKT), correct sentences per second.
* **AMC**: Task 1 `AMC` class and AWGN look-up table, unchanged. For each RB, SNR = p_n · a_un; the rule (highest efficiency with BLER ≤ 10 %) selects the MCS;
  sentences/s of an RB = B · η · (1 − BLER) / b̄ with B = 180 ksym/s per RB (assumption), η the efficiency of the chosen MCS, b̄ = 322.1 information bits per sentence
  (mean Huffman length of the Task 1 test sentences). RBs below the table's SNR range or with zero power deliver nothing. The AWGN table is used because the gain is constant within an RB.

## Results (`results/exercise1_task3/tables/task3_summary_all_methods.csv`, 300 drops)

| P_max | metric | 1 RR | 2 Strongest+WF | 3 MILP | 4 MILP+KKT | 5 DNN | 6 Hybrid |
|---|---|---|---|---|---|---|---|
| 15 dBm | sum rate [Mbps] | 7.8 | 21.4 | 13.1 | 13.2 | 16.4 | 15.4 |
| | users ≥ 1 Mbps [%] | 38.4 | 27.5 | 78.1 | 78.1 | 49.3 | 67.3 |
| | sentences/s | 13 815 | 36 735 | 22 876 | 22 929 | 28 572 | 27 608 |
| 25 dBm | sum rate [Mbps] | 14.9 | 30.9 | 21.2 | 21.2 | 24.1 | 23.2 |
| | users ≥ 1 Mbps [%] | 71.2 | 32.0 | 97.8 | 97.9 | 70.8 | 84.4 |
| | sentences/s | 25 664 | 40 112 | 31 246 | 31 157 | 36 258 | 36 268 |
| 40 dBm | sum rate [Mbps] | 28.2 | 45.2 | 40.2 | 40.2 | 36.2 | 36.2 |
| | users ≥ 1 Mbps [%] | 98.2 | 32.5 | 99.96 | 99.96 | 90.5 | 91.7 |
| | sentences/s | 37 787 | 40 231 | 39 954 | 39 945 | 40 070 | 40 116 |

Other metrics (full tables in the CSV, plots in `figures/`): Jain's index is highest for round-robin (0.58 → 0.89 from 15 to 40 dBm), 0.35–0.55 for the MILP methods, 0.45–0.66 for DNN/hybrid and about 0.21–0.22 for strongest-user;
the 5th-percentile user rate is at most 0.03 Mbps for every method at 15 dBm, and at 40 dBm it is 1.41 Mbps for round-robin, 1.21 for the MILP methods and 0 for strongest-user, DNN and hybrid.
Median computation time per drop: round-robin 0.01 ms, strongest user 0.4 ms, DNN 0.5–0.9 ms, hybrid 5.5–6.6 ms, MILP 5–23 ms, MILP + KKT 11–28 ms.
All 26 x 300 allocations of all six methods were valid (≤ 1 user per RB, p ≥ 0, Σp ≤ P_max).
The all-users-QoS-satisfied fraction is also in the CSV (e.g. 25 dBm: MILP 82.7 %, hybrid 20.0 %, DNN 3.7 %).

## Observations

1. **QoS is often infeasible at low power**: with equal power the MILP can give all 8 users 1 Mbps in only 7.7 % of the drops at 15 dBm, 39 % at 20 dBm, 82.7 % at 25 dBm and 99.7 % from 35 dBm (`task3_milp_feasibility.csv`). The drops where it is infeasible are reported in the flags, never hidden, and the QoS columns therefore stay below 100 % for every method at low power.
2. **Strongest-user assignment maximises sum rate but starves users**: it has the highest sum rate and about 32 % QoS (Jain ≈ 0.22), because strong users take most RBs.
3. **Exact MILP** keeps QoS highest (97.8 % at 25 dBm, 99.96 % at 40 dBm), at the price of a sum rate below method 2 and the lowest fairness index among the QoS-aware methods (method 2 is lower still).
   KKT water-filling after the MILP changes the sum rate by at most 0.07 Mbps and the QoS by at most 0.17 percentage points (method 4 ≥ method 3 at every power); the equal-power MILP already satisfies the constraint wherever it is feasible, so the gain is small in these simulations.
4. **The standalone DNN is clearly weaker than the MILP at QoS**: 49 % / 71 % / 91 % of users satisfied at 15 / 25 / 40 dBm against 78 / 98 / 99.96 %, and fully satisfied drops are rare (3.7 % at 25 dBm).
   Its sum rate is *higher* than the MILP's at low power (24.1 vs 21.2 Mbps at 25 dBm) only because it ignores QoS more often, and lower at 35–40 dBm (36.2 vs 40.2 Mbps). It runs about 8–28 times faster than the MILP.
5. **Hybrid**: replacing the DNN power by KKT water-filling raises QoS by 18.0 points at 15 dBm, 13.6 points at 25 dBm and 1.2 points at 40 dBm, for a sum-rate cost of about 1 Mbps at 15–25 dBm (none at 40 dBm).
   It stays well below the MILP methods in QoS, so the DNN assignment is the limiting factor, not the power allocation. Its time (5.5–6.6 ms) is dominated by the bisection implementation of the water-filling and is below MILP + KKT (11–28 ms) at every power.
6. **Sentences per second**: throughput-driven, so method 2 is best at low power (36.7 k at 15 dBm against 22.9 k for the MILP); all methods saturate at about 40 k at 40 dBm, which is the ceiling of the highest MCS over 16 RBs. This metric does not penalise users left without service, unlike the QoS metrics.

## Task 3 limitations

300 drops per power point (confidence intervals of sum rate and sentences/s are in the CSV and drawn in those two plots); one DNN for all power levels with P_max as input; DNN architecture and λ chosen from a few short trials only,
no further tuning, so a better-trained network may close part of the gap to the MILP; the 5th percentile is 0 whenever more than 5 % of the users receive no RB; QoS-infeasible drops use best-effort fallbacks;
sentences/s rests on the assumed 180 ksym/s per RB and the AWGN table. Task 3 notebooks execute without errors (kernel `exercise1`); the full sweep is not re-run by them.
