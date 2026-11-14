# Handoff prompt: finish Exercise 1 – Task 2 (Autoencoder Physical Layer)

You are an experienced Python / PyTorch developer with a digital-communications background. Continue work in the existing
`Exercise1` project (JupyterLab, kernel "Python (Exercise1)", venv `.venv`). **Exercise 1 Task 1 is complete and must not be
modified** (81 pytest tests pass: `python -m pytest`). Do only **Exercise 1 – Task 2**; do not implement Exercise 1 Task 3 or Exercise 2.
Keep everything simple: small MLP autoencoders, no Transformers, CPU only, reproducible seeds, real simulation results only.

## Task 2 requirements (from the problem statement)
Train an autoencoder (AE) transceiver with (n,k) = (7,4) in AWGN. Compare its BLER/BER with Hamming (7,4) under hard-decision and
ML decoding and with uncoded BPSK. Train a (2,4) AE (16 messages on one complex symbol) and compare the learned constellation with
16-QAM. Repeat the (7,4) comparison in Rayleigh fading with receiver CSI. Explain why the AE approaches but does not exceed ML
decoding of Hamming(7,4), and what limits the approach for longer blocks.

## What already exists (written, NOT yet run – syntax-checked only)
* `task2_autoencoder/autoencoder.py` – `Autoencoder(n_real, hidden=16, rx_extra, power)`: one-hot(16)→Dense16+ReLU→Dense n→codebook power
  normalisation (mean energy per real symbol = `power`; 1 for (7,4), 0.5 for (2,4) so the complex symbol has power 1) → channel →
  Dense16+ReLU→Dense16 (softmax/cross-entropy on the message index). Torch AWGN and Rayleigh channels (one gain per block, ZF `y/h`,
  CSI `|h|²` fed to the decoder via `rx_extra=1`), `train()` (random Eb/N0 per step), `save_model/load_model`.
* `task2_autoencoder/baselines.py` – systematic Hamming(7,4), syndrome hard decoding, ML (correlation) decoding, uncoded BPSK,
  theory (BPSK AWGN, BPSK Rayleigh), and `simulate_point()` Monte-Carlo (same bits/noise/fading for all methods). Reuses
  `task1_conventional.channel` (`complex_gaussian`, `noise_variance`, `rayleigh_gains`, `ebn0_to_esn0_db`) and `equalizer.zf_equalize`.
* `task2_autoencoder/experiments.py` – training configs (`CFG`), `train_model()`, `eval_74()`, `eval_24()` (AE vs Gray 16-QAM from
  `task1_conventional.modulation.get_modem("16QAM")`); outputs go to `models/autoencoder/` and `results/exercise1_task2/{figures,tables}`.
SNR convention: Eb/N0 per information bit; Es/N0 = Eb/N0 · R per symbol (R = 4/7 bit per real symbol for (7,4); 4 bits per complex
symbol for (2,4)); noise std per real dimension = sqrt(N0/2). Keep this consistent with `task1_conventional/channel.py`.

## Remaining work (do in this order, run and fix after each step)
1. **Install PyTorch into `.venv`** (it is not installed). In the build environment the CPU index `download.pytorch.org` was blocked and the
   PyPI wheel (with CUDA deps) downloaded very slowly, so on the user's own machine run
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`. Add `torch` to `requirements.txt` afterwards.
2. **Debug the existing code**: confirm it imports and that shapes, power normalisation and channels behave. Fix any bugs; verify the
   (7,4) AE reaches roughly the Hamming-ML curve (it should approach, not beat it) and that BPSK/Hamming curves match theory.
   Tune training only if needed (SNR range, steps, learning rate); document the final settings.
3. **pytest tests** in `tests/test_ae_*.py` covering: input/output dimensions; codebook power = `power`; encode→decode exact in a noiseless
   channel after training; Hamming encode/hard/ML decoding noiseless exactness and min distance 3; AWGN noise variance; Rayleigh gain
   E|h|²≈1 and ZF exactness without noise; loss backpropagates (non-zero grads); BER/BLER computation; model save/load round trip;
   reproducibility with a fixed seed. Existing 81 tests must still pass.
4. **Notebooks** in `notebooks/Exercise1_Task2/` (generate with nbformat, execute with `nbclient` in kernel `exercise1`, see
   `experiments/run_notebooks.py` for how Task 1 does it): `01_autoencoder_7_4_AWGN.ipynb` (BER vs Eb/N0 for uncoded BPSK, Hamming hard,
   Hamming ML, AE; also BLER), `02_autoencoder_7_4_Rayleigh.ipynb` (same in Rayleigh with CSI + ZF, and an AWGN-vs-Rayleigh comparison),
   `03_autoencoder_2_4_constellation.ipynb` (learned constellation, conventional normalised 16-QAM, BER/SER vs Eb/N0, explanation of the
   differences). Short markdown explanation before each section; import code from the modules; run top to bottom without errors.
   Note: the AE is trained with message-level cross-entropy, so its bit labelling is not Gray – report SER as well as BER for (2,4)
   and say so.
5. **Save**: three trained models (`models/autoencoder/*.pt` + config JSON), training-loss CSVs, simulation CSVs, BER-vs-SNR plots, learned and
   16-QAM constellation figures. Use enough Monte-Carlo blocks that the lowest plotted BER has ≥100 errors, or state the limit.
6. **README**: add a short "Exercise 1 – Task 2" section (how to run, outputs, settings) and these six short explanations, using the
   actual results: how the (7,4) AE works; comparison with uncoded BPSK and Hamming (hard vs ML); effect of Rayleigh fading; how the (2,4) AE
   learns its constellation; why the AE does not beat ML decoding of Hamming(7,4) (ML is optimal for that code; the AE can at best match it
   and learns a code of equal or better distance structure only up to training limits); limits for longer blocks (2^k messages → exponential
   one-hot size and training set, k=4 is tiny).
7. State clearly any requirement not completed. Do not invent numbers.
