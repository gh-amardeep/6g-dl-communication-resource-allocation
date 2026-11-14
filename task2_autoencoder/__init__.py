"""Exercise 1 / Task 2: deep-learning physical layer (small autoencoders, PyTorch, CPU).

autoencoder.py : (7,4) and (2,4) autoencoder, torch channels, training, save/load
baselines.py   : uncoded BPSK, Hamming(7,4) hard-decision and ML, Monte-Carlo evaluator
experiments.py : train / evaluate / plot helpers used by the notebooks
Reuses task1_conventional.channel (noise, Rayleigh gains, SNR conversions), equalizer (ZF)
and modulation (16-QAM)."""
