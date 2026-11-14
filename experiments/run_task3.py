"""Run the Task 3 sweep (all six methods, identical drops) and write CSVs + figures.  Usage: python experiments/run_task3.py [n_drops]"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)
from task3_resource import experiments as ex

n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
df, ur, bits = ex.run_sweep(n_drops=n)
T = ex.RES / "tables"; T.mkdir(parents=True, exist_ok=True)
df.to_csv(T / "task3_per_drop_results.csv", index=False)
summ = ex.summarise(df, ur)
summ.to_csv(T / "task3_summary_all_methods.csv", index=False)
summ[summ.method.str.startswith(("1", "2", "3", "4"))].to_csv(T / "task3_benchmark_results.csv", index=False)
summ[summ.method.str.startswith(("3", "4"))].to_csv(T / "task3_milp_results.csv", index=False)
summ[summ.method.str.startswith("5")].to_csv(T / "task3_dnn_results.csv", index=False)
summ[summ.method.str.startswith("6")].to_csv(T / "task3_hybrid_results.csv", index=False)
np.savez_compressed(ex.RES / "user_rates_mbps.npz", methods=np.array(ex.METHODS), power_dbm=np.array(list(ur)),
                    rates=np.stack([ur[k] for k in ur]))          # (power, method, drop, user)
# MILP feasibility vs power
f = df[df.method == ex.METHODS[2]].groupby("power_dbm").qos_feasible_flag.mean().mul(100).rename("milp_qos_feasible_pct")
f.to_csv(T / "task3_milp_feasibility.csv")
ex.plot_all(summ)
print("bits/sentence", bits, "done")
