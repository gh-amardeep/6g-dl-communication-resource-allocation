# PROJECT_STATUS.md
## A. Overview
M.Tech assignment "Exercise 1: Deep Learning for a Conventional Communication System with Multi-User Resource Allocation" (JupyterLab, Python/NumPy/PyTorch, CPU). Original statement PDF + master prompt supplied by the user. Task 1 = conventional text link; Task 2 = autoencoder physical layer; Task 3 = resource allocation (NOT started, out of current scope).
## B. Progress
| Item | Status |
|---|---|
| Phase 1 environment/.venv/kernel/requirements | DONE |
| Task 1 (a)-(d) conventional system, 11 notebooks, full-mode results, README | DONE (81 tests pass) |
| Task 2 modules (`task2_autoencoder/`) + 9 tests | DONE, PASS |
| Task 2 models trained & saved | DONE |
| Task 2 notebooks executed, figures/CSVs | DONE (3/3 execute, 0 errors) |
| Task 2 README section + six explanations, torch in requirements | DONE (Checkpoint 03) |
| Task 3 modules, tests, trained DNN, sweep, plots, CSVs, 5 notebooks, README | DONE (Checkpoint 06) |
Approx. overall: Task 1 100 %, Task 2 100 %, Task 3 100 %.
## C. Implementation
Created (Task 2): task2_autoencoder/{__init__,autoencoder,baselines,experiments}.py, tests/test_ae_task2.py, experiments/build_notebooks_task2.py, notebooks/Exercise1_Task2/*.ipynb (executed, outputs embedded), models/autoencoder/*.pt + *_config.json, results/exercise1_task2/{figures (5 PNG), tables (7 CSV)}, handover docs. Task 1 files untouched. Libraries: torch 2.15, numpy, scipy, pandas, matplotlib, nbformat/nbclient. Decisions: see LIVE_SESSION_STATE.md §5.
## D. Testing
`pytest tests` at Checkpoint 06: 103 passed, 1 skipped (test_embedding.py: spaCy missing in this sandbox; passed earlier at Checkpoint 02). Task 3 notebooks executed (5/5 OK). Sweep: 300 drops x 26 powers, all allocations valid. DNN validated on 200 unseen drops (notebook 03). Not tested: nothing known.
## E. Current working point
Task 3 finished. No open work.
## F. Next steps
See NEXT_STEPS.md.
