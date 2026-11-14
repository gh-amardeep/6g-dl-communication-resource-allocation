# LIVE_SESSION_STATE.md (read FIRST) – state at Checkpoint 06 (Task 3 COMPLETE)
## 1. Current task
Exercise 1 – **Task 3: DNN-based multi-user resource allocation** (8 users, 16 RBs). Task 1 and Task 2 are COMPLETE; do not modify them. Exercise 2 is out of scope. NOTE: the assignment PDF was never provided in this chat; the spec is the user's Task 3 prompt (parameters table, 4 benchmarks, DNN, hybrid, 5 metrics, AMC, 5 notebooks).
## 2. Current progress
Task 3 COMPLETE: modules (system, allocation M1-M4, dnn, amc_eval, experiments), 15 tests, trained DNN (12000 steps, lam 5, hidden 512), sweep (300 drops x 26 powers, seed 20241114, identical drops for all methods), 6 metric plots + 3 extra figures, CSVs, 5 notebooks executed without errors (kernel `exercise1`), README Task 3 section with results and limitations. Full `pytest`: 103 passed, 1 skipped (spaCy missing in this sandbox).
## 3. Exact working point
Modules + unit tests finished; about to (a) train the final DNN, (b) write experiments.py (sweep 15–40 dBm, identical drops for all methods), (c) amc_eval.py, (d) notebooks via experiments/build_notebooks_task3.py.
## 4. Files being worked on
Created: task3_resource/{__init__,system,allocation,dnn}.py, tests/test_task3.py.
## 5. Technical decisions
- Units: W; a = g/(N0 B F) with N0BF = −112.45 dBm; rate = B log2(1+p a). Channel: Task 1 `rayleigh_gains` (CN(0,1), independent per user & RB) × path loss 128.1+37.6log10(d_km) × log-normal shadowing 8 dB (one per user per drop); users uniform in cell, d∈[35,500] m (area-uniform). One drop = one channel realisation (8×16).
- M1 round-robin: RB n → user n mod 8, equal power Pmax/16. M2: argmax user per RB + plain water-filling (no QoS enforcement). M3: SciPy `milp` (HiGHS), r_un=B log2(1+Pmax/N a_un) constants, ≤1 user/RB, R_u ≥ 1 Mbps hard. If infeasible: flagged `qos_feasible=False` and best-effort MILP (max #satisfied users, then sum rate). M4: M3 assignment + KKT water-filling.
- KKT WF with QoS (allocation.kkt_waterfilling): p_n=[L_u−1/g_n]⁺, L_u=max(L, λ_u), λ_u = level giving exactly Rmin; scalar L by bisection so Σp=Pmax. If infeasible: Rmin is guaranteed for the cheapest users that fit in the budget, flag False.
- DNN: MLP (129→256→256→144), input 10log10(a) standardised (30 dB, 13 dB) + Pmax dBm (extra scalar input – documented deviation: "H only"); outputs assignment logits (softmax over users per RB, hard argmax with straight-through) and power fractions (softmax) × Pmax. Loss = −SumRate[Mbps] + λ Σ max(0,1−R_u)², λ=20 default; trained on fresh random drops, Pmax ~ U(15,40) dBm, seed 20241114. Hybrid = DNN assignment + KKT WF.
- Planned AMC link: reuse Task 1 `AMC` + full AWGN LUT (`results/tables/full_amc_lut_awgn.csv`); per RB SNR = p_n a_un (instantaneous, AWGN-equivalent because the RB gain is constant, matching Task 1 block fading + ZF); sentences/s per RB = B·η·(1−BLER)/(mean information bits per test sentence); symbol rate per RB assumed = B = 180 ksym/s; outage/undefined BLER → 0 delivered.
## 6. Last successful operation
`pytest tests`: all passing (see PROJECT_STATUS §D). Short DNN trials (800 steps) ran fine; λ=5 gave higher sum rate but lower QoS than I expected – final λ to be chosen on the full training run.
## 7. Current errors / limitations
None open. QoS is often infeasible at low power (e.g. 15 dBm: 37/40 test drops infeasible for MILP equal-power) – this is a real property of the setup, reported not hidden.
## 8. Pending operations
Train final DNN (6000+ steps); experiments.py sweep; amc_eval.py + test; 5 notebooks; plots/CSVs; README; docs; final checkpoint.
## 9. Immediate next action
Run `python experiments/run_task3.py` (300 drops, 15-40 dBm; ~10+ min on 1 core, run detached), then notebooks, README, docs.
## 10. Context
Sandbox: torch installed with --no-deps; disk is tight. No spaCy here (embedding tests skipped).

Key findings (see README Task 3): QoS equal-power feasibility 7.7 % of drops at 15 dBm -> 99.7 % from 35 dBm; DNN QoS 49/71/91 % vs MILP 78/98/99.96 % at 15/25/40 dBm; hybrid +18/+13.6/+1.2 points over DNN; assignment PDF was never supplied; P_max is an extra DNN input; 180 ksym/s per RB and 322.1 bits/sentence are assumptions. Sandbox: torch --no-deps, nbformat/nbclient/ipykernel installed for notebooks; sweep takes ~5 min and must be launched detached (setsid nohup).
