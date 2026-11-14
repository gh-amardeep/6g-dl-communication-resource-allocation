# CHANGELOG
## Milestone 1 – Phase 1 + Task 1 (complete)
Venv/kernel, dataset (Brown, 25,932/2,000), Huffman, conv. code K=7 (133,171) with puncturing, QAM, AWGN/Rayleigh, ZF, max-log LLR, soft Viterbi, BER/BLER sweeps (full mode), BLEU/word-acc/embedding cosine, AMC; 11 notebooks; README; 81 tests. Fixed: notebook 09 `%` formatting bug.
## Milestone 2 – Task 2 (code, tests, models, notebooks, results DONE; README pending) – Checkpoint 02
Added: torch install; `task2_autoencoder/*`; `tests/test_ae_task2.py` (+9 tests, total 90 pass); `experiments/build_notebooks_task2.py`; 3 notebooks (built); trained models `models/autoencoder/ae74_awgn.pt, ae74_rayleigh.pt, ae24_awgn.pt`; training-loss CSVs; handover docs (LIVE_SESSION_STATE, PROJECT_STATUS, CHANGELOG, NEXT_STEPS, HANDOFF_PROMPT_Task2).
Fixes: `Tensor→float` warnings (`.item()`); (2,4) power handled via `power=0.5`; constellation helper uses `Modem.constellation()[0]`.
Executed the 3 notebooks (0 errors) → 5 figures + 5 result CSVs in results/exercise1_task2/. Corrected notebook commentary to match the real numbers (rebuilt + re-executed; results identical). Tests: pytest 90 passed.
Pending: README Task 2 section + six explanations, torch in requirements.txt.

## Milestone 3 – Task 2 documentation (takeover session, 2026-11-14) – Checkpoint 03
Modified: README.md (Task 2 section: files, run steps, settings table, real results, six explanations, limitations; checklist ticked), requirements.txt (+torch==2.15.0), LIVE_SESSION_STATE.md, PROJECT_STATUS.md, NEXT_STEPS.md, CHANGELOG.md. No code, model, notebook or result changed.
Checks: numbers re-derived from results CSVs; corrected two draft statements (1.5 dB Hamming-ML gain; constellation shape). Tests: pytest 88 passed, 1 skipped (spaCy absent in sandbox).

## Milestone 4 – Task 3 modules + tests (2026-11-14) – Checkpoint 04
Created: task3_resource/{__init__,system,allocation,dnn}.py, tests/test_task3.py (13 tests PASS). Decisions: see LIVE_SESSION_STATE §5. Fix during dev: KKT fallback now guarantees Rmin to as many users as the budget allows instead of plain water-filling. No Task 1/Task 2 file changed.

## Milestone 5 – Task 3 AMC module, sweep driver, trained DNN (2026-11-14) – Checkpoint 05 (recovery)
Created: task3_resource/amc_eval.py, task3_resource/experiments.py, models/resource_allocation/dnn_alloc.{pt,json}, results/exercise1_task3/tables/dnn_training_history.csv. Sweep not yet run.

## Milestone 6 – Task 3 complete (2026-11-14) – Checkpoint 06
Created: experiments/run_task3.py, experiments/build_notebooks_task3.py, notebooks/Exercise1_Task3/01-05 (executed), results/exercise1_task3/{figures (9 PNG), tables (CSVs), user_rates_mbps.npz}, tests (+2 AMC/DNN-file tests). Modified: README.md (Task 3 section, checklist), task3_resource/experiments.py (median timing). Bug fixed: kkt_waterfilling return unpacking in evaluate_drop. Tests: 103 passed, 1 skipped. Checkpoint 06.
