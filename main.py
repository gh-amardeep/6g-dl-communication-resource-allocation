#!/usr/bin/env python
"""Command-line runner for Exercise 1 / Task 1.

    python main.py --experiment dataset
    python main.py --experiment ber           --mode quick     # Task 1(a) uncoded BER validation
    python main.py --experiment mcs           --mode quick     # Task 1(b)+(c) sweep (BER, BLER, text metrics)
    python main.py --experiment text-quality  --mode quick     # (c) figures (runs the sweep if missing)
    python main.py --experiment amc           --mode quick     # (d) LUTs + AMC figures
    python main.py --experiment all           --mode full
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True,
                    choices=["dataset", "ber", "mcs", "text-quality", "amc", "all"])
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--seed", type=int, default=None, help="override the master random seed")
    ap.add_argument("--no-resume", action="store_true", help="recompute the MCS sweep from scratch")
    a = ap.parse_args(argv)

    import os
    os.environ.setdefault("MPLBACKEND", "Agg")      # headless figure export from the terminal
    os.chdir(ROOT)
    from task1_conventional import experiments as ex, plotting
    seed = ex.SEED if a.seed is None else a.seed
    tab = ex.paths(".")["tab"]
    if a.experiment == "dataset":
        from task1_conventional.dataset import prepare_dataset
        print(prepare_dataset("data"))
        return
    if a.experiment in ("ber", "all"):
        ex.run_uncoded_validation(a.mode, ".", seed)
    if a.experiment in ("mcs", "text-quality", "amc", "all"):
        if a.experiment in ("mcs", "all") or not (tab / f"{a.mode}_mcs_sweep.csv").exists():
            ex.run_mcs_sweep(a.mode, ".", seed, resume=not a.no_resume)
    if a.experiment in ("amc", "all"):
        ex.run_amc(a.mode, ".", seed)
    try:
        plotting.make_all_figures(a.mode, ".")
    except FileNotFoundError as e:
        print("figures skipped (missing table):", e)
    print(f"done ({a.experiment}, {a.mode}); tables -> results/tables, figures -> results/figures")


if __name__ == "__main__":
    main()
