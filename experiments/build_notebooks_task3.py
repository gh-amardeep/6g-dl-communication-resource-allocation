"""Generate the five Task 3 notebooks (edit text here, rebuild, then execute with nbclient)."""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "Exercise1_Task3"
OUT.mkdir(parents=True, exist_ok=True)
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = code('''import os, sys
from pathlib import Path
ROOT = Path.cwd().resolve().parents[1]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
torch.set_num_threads(1)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
from task3_resource.system import P, dbm_to_w, draw_channels, rb_rates, user_rates, POWER_DBM_GRID
from task3_resource import allocation as al, dnn, experiments as ex
T = ex.RES / "tables"; F = ex.RES / "figures"
''')


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "exercise1", "display_name": "Python (Exercise1)", "language": "python"}
    nbf.write(nb, OUT / name)


# ---------------------------------------------------------------- 01
save("01_channel_matrix_and_problem.ipynb", [
    md("# Task 3 – Notebook 1: channel matrix and problem formulation\n"
       "8 users, 16 resource blocks (RB) of 180 kHz, cell radius 500 m, minimum distance 35 m.\n\n"
       "**Channel** (reused from Task 1: Rayleigh gains `rayleigh_gains`): path loss $128.1+37.6\\log_{10}(d/\\mathrm{km})$ dB, 8 dB log-normal shadowing (one value per user), "
       "Rayleigh fading independent per user and RB.  Noise: $N_0=-174$ dBm/Hz, noise figure 9 dB, so the noise power per RB is $\\sigma^2=N_0BF=-112.45$ dBm.\n\n"
       "**Problem**\n\n"
       "$$\\max_{x_{u,n},p_n}\\sum_u\\sum_n x_{u,n}B\\log_2(1+\\gamma_{u,n}),\\qquad \\gamma_{u,n}=\\frac{p_n g_{u,n}}{N_0BF}$$\n\n"
       "s.t. $\\sum_u x_{u,n}\\le1$, $\\sum_n p_n\\le P_{\\max}$, $R_u=\\sum_n x_{u,n}B\\log_2(1+\\gamma_{u,n})\\ge R_{\\min}=1$ Mbps, $x_{u,n}\\in\\{0,1\\}$, $p_n\\ge0$.\n\n"
       "Assumptions: powers in W; $a_{u,n}=g_{u,n}/(N_0BF)$ so $\\gamma=p\\,a$; every RB may be left to one user only; one drop = one channel realisation (user positions, shadowing, fading) assumed constant during the allocation. "
       "The QoS constraint can be **infeasible** (low power, far users) – this is detected and reported, never hidden."),
    SETUP,
    code('''print("noise power per RB: %.2f dBm" % (10*np.log10(P.noise_w*1e3)))
rng = np.random.default_rng(0)
a = draw_channels(1, rng)[0]            # H: 8 users x 16 RBs (noise-normalised gain)
print("H shape:", a.shape)
fig, ax = plt.subplots(figsize=(8, 3.6))
im = ax.imshow(10*np.log10(a), aspect="auto", cmap="viridis"); plt.colorbar(im, label="10log10(g/(N0 B F)) [dB]")
ax.set_xlabel("RB"); ax.set_ylabel("user"); ax.set_title("One channel drop H (8 x 16)")
fig.tight_layout(); fig.savefig(F / "channel_matrix_example.png", dpi=150); plt.show()'''),
    md("### Is QoS (1 Mbps for all 8 users) achievable?  MILP feasibility over 100 drops"),
    code('''A = draw_channels(100, np.random.default_rng(1))
rows = []
for pd_ in [15, 20, 25, 30, 40]:
    feas = np.mean([al.milp_equal_power(x, dbm_to_w(pd_))[2]["qos_feasible"] for x in A])
    rows.append({"P_max [dBm]": pd_, "drops with feasible QoS (equal power) [%]": 100*feas})
pd.DataFrame(rows)'''),
])

# ---------------------------------------------------------------- 02
save("02_benchmark_allocations.ipynb", [
    md("# Task 3 – Notebook 2: the four benchmark allocations\n"
       "1. **Round-robin + equal power** – RB $n\\to$ user $n \\bmod 8$, $p_n=P_{\\max}/16$.\n"
       "2. **Strongest user + water-filling** – each RB to its best user, then $p_n=[L-1/a_n]^+$, $\\sum p_n=P_{\\max}$ (no QoS enforcement).\n"
       "3. **MILP + equal power** – constants $r_{u,n}=B\\log_2(1+\\tfrac{P_{\\max}}{16}a_{u,n})$; maximise $\\sum r_{u,n}x_{u,n}$ s.t. one user per RB and $\\sum_n r_{u,n}x_{u,n}\\ge R_{\\min}$ (SciPy `milp`, HiGHS). "
       "If infeasible it reports `qos_feasible=False` and returns the best effort (most users satisfied, then sum rate).\n"
       "4. **MILP assignment + KKT water-filling** – fixed assignment, Lagrangian $\\sum_u(1+\\mu_u)R_u-\\nu(\\sum p-P_{\\max})$. Stationarity on RB $n$ of user $u$ gives "
       "$p_n=[L_u-1/a_n]^+$ with $L_u=(1+\\mu_u)B/(\\nu\\ln2)$. If user $u$'s QoS is slack, $\\mu_u=0$ and $L_u=L$; if it is active, $L_u=\\lambda_u$, the level that gives exactly $R_{\\min}$. "
       "Hence $L_u=\\max(L,\\lambda_u)$ and the scalar $L$ is found by bisection so that $\\sum p_n=P_{\\max}$. If $\\sum_u P_u^{\\min}>P_{\\max}$ the assignment cannot meet QoS: Rmin is guaranteed to the cheapest users that fit and the case is flagged."),
    SETUP,
    code('''pdbm = 30; pmax = dbm_to_w(pdbm)
a = draw_channels(1, np.random.default_rng(1))[0]
milp_res = al.milp_equal_power(a, pmax)
runs = {"1 Round-robin": al.round_robin(a, pmax), "2 Strongest+WF": al.strongest_user_wf(a, pmax), "3 MILP+equal": milp_res,
        "4 MILP+KKT-WF": al.milp_then_waterfilling(a, pmax, milp_result=milp_res)}
tab = []
for name, (x, pw, info) in runs.items():
    R = user_rates(x, pw, a)/1e6
    tab.append({"method": name, "sum rate [Mbps]": R.sum(), "users >= 1 Mbps": int((R >= 1).sum()), "min user rate": R.min(),
                "power used / budget": pw.sum()/pmax, "QoS feasible flag": info.get("qos_feasible", "n/a"), "RB owner": x.argmax(0).tolist()})
pd.DataFrame(tab).round(3)'''),
    code('''x, pw, _ = runs["4 MILP+KKT-WF"]
print("user rates [Mbps]:", (user_rates(x, pw, a)/1e6).round(2))
print("power per RB [mW]:", (pw*1e3).round(1))'''),
    md("### Benchmark curves (methods 1–4, 300 drops per power) and MILP feasibility\nProduced by `python experiments/run_task3.py` (see notebook 5 for all six methods)."),
    code('''bench = pd.read_csv(T / "task3_benchmark_results.csv")
display(bench[bench.power_dbm.isin([15, 20, 25, 30, 40])][["power_dbm", "method", "sum_rate_mbps", "qos_pct", "all_users_qos_pct", "jain", "p5_user_rate_mbps", "time_ms"]].round(3))
feas = pd.read_csv(T / "task3_milp_feasibility.csv")
fig, ax = plt.subplots(figsize=(6, 3.4)); ax.plot(feas.power_dbm, feas.milp_qos_feasible_pct, "o-")
ax.set_xlabel("P_max [dBm]"); ax.set_ylabel("drops with feasible QoS [%]"); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(F / "milp_qos_feasibility_vs_power.png", dpi=150); plt.show()'''),
])

# ---------------------------------------------------------------- 03
save("03_dnn_resource_allocation.ipynb", [
    md("# Task 3 – Notebook 3: DNN resource allocation (unsupervised)\n"
       "MLP (input 8x16 gains in dB + $P_{\\max}$ in dBm, two hidden layers of 512 ReLU units). Outputs: (a) assignment logits per RB, softmax over users, hard arg-max at test time "
       "(straight-through gradient in training) – each RB gets exactly one user; (b) power fractions via softmax, $p_n=P_{\\max}\\,\\mathrm{softmax}_n$ – non-negative and summing to the budget.\n\n"
       "Loss: $L=-\\text{SumRate}[\\text{Mbps}]+\\lambda\\sum_u\\max(0,1-R_u[\\text{Mbps}])^2$, $\\lambda=5$. Trained on freshly generated channels with $P_{\\max}\\sim U(15,40)$ dBm, **no MILP labels**. "
       "Note: $P_{\\max}$ is given as an additional scalar input (one model for the whole sweep); the assignment text lists only $H$. $\\lambda$ and size were chosen from short trials on held-out drops (λ∈{5,10,20,60}, hidden 256/512)."),
    SETUP,
    code('''net = dnn.load(ex.MODELS / "dnn_alloc.pt")
print(open(ex.MODELS / "dnn_alloc.json").read())
print("parameters:", sum(p.numel() for p in net.parameters()))
h = pd.read_csv(T / "dnn_training_history.csv")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
ax[0].plot(h.step, h.loss); ax[0].set_title("training loss"); ax[1].plot(h.step, h.sum_rate_mbps); ax[1].set_title("sum rate [Mbps] (training batches)")
ax[2].plot(h.step, 100*h.qos_satisfied); ax[2].set_title("users >= 1 Mbps [%] (training batches)")
for x_ in ax: x_.set_xlabel("step"); x_.grid(alpha=.3)
fig.tight_layout(); fig.savefig(F / "dnn_training_curves.png", dpi=150); plt.show()'''),
    md("### Validity and QoS check on 200 fresh test drops (not seen in training)"),
    code('''V = draw_channels(200, np.random.default_rng(424242))
rows = []
for pdbm in [15, 20, 25, 30, 35, 40]:
    pmax = dbm_to_w(pdbm); ok = True; S = []; Q = []; AQ = []; Sm = []; Qm = []
    for a in V:
        x, pw, _ = dnn.predict(net, a, pmax)
        ok &= bool((x.sum(0) == 1).all() and (pw >= 0).all() and pw.sum() <= pmax*(1+1e-9) and x.shape == (8, 16) and pw.shape == (16,))
        R = user_rates(x, pw, a); S.append(R.sum()/1e6); Q.append((R >= 1e6).mean()); AQ.append((R >= 1e6).all())
        xm, pm, _ = al.milp_equal_power(a, pmax); Rm = user_rates(xm, pm, a); Sm.append(Rm.sum()/1e6); Qm.append((Rm >= 1e6).mean())
    rows.append({"P_max [dBm]": pdbm, "all outputs valid": ok, "DNN sum rate": np.mean(S), "MILP sum rate": np.mean(Sm), "DNN users>=1Mbps [%]": 100*np.mean(Q),
                 "MILP users>=1Mbps [%]": 100*np.mean(Qm), "DNN drops with all users OK [%]": 100*np.mean(AQ)})
dv = pd.DataFrame(rows).round(2); dv.to_csv(T / "task3_dnn_validity_check.csv", index=False); dv'''),
    md("QoS violations of the standalone DNN are **reported** above (columns 'users>=1Mbps' and 'drops with all users OK'); nothing is post-processed here. The hybrid method of notebook 4 repairs the power allocation."),
])

# ---------------------------------------------------------------- 04
save("04_hybrid_allocation.ipynb", [
    md("# Task 3 – Notebook 4: hybrid DNN assignment + KKT water-filling\n"
       "Keep the DNN's RB assignment, discard its power output and recompute the powers with the same KKT water-filling used in method 4 (water level $L_u=\\max(L,\\lambda_u)$, total power $=P_{\\max}$, "
       "$R_{\\min}$ guaranteed whenever the assignment admits it).  Inference cost = DNN forward pass + water-filling."),
    SETUP,
    code('''net = dnn.load(ex.MODELS / "dnn_alloc.pt")
V = draw_channels(5, np.random.default_rng(11)); a = V[0]; pmax = dbm_to_w(25)
x, p_dnn, _ = dnn.predict(net, a, pmax)
p_hyb, info = al.kkt_waterfilling(x, a, pmax)
for n, p_ in [("DNN", p_dnn), ("Hybrid", p_hyb)]:
    R = user_rates(x, p_, a)/1e6
    print(n, "sum %.2f Mbps | users>=1Mbps: %d/8 | power %.3f of budget" % (R.sum(), (R >= 1).sum(), p_.sum()/pmax), "| QoS feasible flag:", info["qos_feasible"] if n == "Hybrid" else "n/a")'''),
    code('''S = pd.read_csv(T / "task3_summary_all_methods.csv")
sel = S[S.method.str.startswith(("4", "5", "6")) & S.power_dbm.isin([15, 20, 25, 30, 35, 40])]
display(sel.pivot(index="power_dbm", columns="method", values="sum_rate_mbps").round(2))
display(sel.pivot(index="power_dbm", columns="method", values="qos_pct").round(1))
display(sel.pivot(index="power_dbm", columns="method", values="all_users_qos_pct").round(1))'''),
])

# ---------------------------------------------------------------- 05
import pandas as pd
RES = Path(__file__).resolve().parents[1] / "results" / "exercise1_task3" / "tables"
S = pd.read_csv(RES / "task3_summary_all_methods.csv")


def row(p, m):
    return S[(S.power_dbm == p) & S.method.str.startswith(m)].iloc[0]


lines = []
for p in (15, 25, 40):
    parts = [f"{m}: {row(p, m).sum_rate_mbps:.1f} Mbps, {row(p, m).qos_pct:.0f} % QoS, {row(p, m).sentences_per_s:,.0f} sent/s" for m in "123456"]
    lines.append(f"* **{p} dBm** – " + "; ".join(f"method {s}" for s in parts))
save("05_final_comparison_and_amc.ipynb", [
    md("# Task 3 – Notebook 5: final comparison of the six methods and AMC\n"
       "All six methods are evaluated on the **same 300 channel drops** at every $P_{\\max}$ = 15…40 dBm (1 dB steps) by `experiments/run_task3.py`; this notebook only loads the results.\n\n"
       "Metrics: sum rate, % users with $R_u\\ge1$ Mbps, Jain's index $J=(\\sum R_u)^2/(U\\sum R_u^2)$ per drop (mean), 5th percentile of all user rates (pooled over drops and users), "
       "median computation time per drop (CPU, 1 thread; method 4 = MILP + KKT, method 6 = DNN + KKT; first-call overheads excluded by using the median), and correct sentences per second.\n\n"
       "**AMC link** (Task 1 `AMC` class and AWGN look-up table, unchanged): for every RB the SNR is $p_n a_{u,n}$, the AMC picks the highest-efficiency MCS with BLER ≤ 10 %, "
       "and sentences/s of an RB = $B\\,\\eta\\,(1-\\mathrm{BLER})/\\bar b$ with $B$=180 ksym/s per RB (assumption), $\\eta$ = efficiency of the chosen MCS, $\\bar b$ = 322.1 information bits per sentence (mean Huffman length of the Task 1 test sentences). "
       "RBs with undefined BLER (SNR below the table) or zero power deliver nothing."),
    SETUP,
    code('''S = pd.read_csv(T / "task3_summary_all_methods.csv")
cols = ["power_dbm", "method", "sum_rate_mbps", "qos_pct", "all_users_qos_pct", "jain", "p5_user_rate_mbps", "time_ms", "sentences_per_s"]
display(S[S.power_dbm.isin([15, 20, 25, 30, 35, 40])][cols].round(3).reset_index(drop=True))
print("all allocations valid (<=1 user/RB, p>=0, sum p<=Pmax):", bool(S.all_valid.all()))'''),
    code('''from IPython.display import Image, display
for f in ["sum_rate_vs_power", "qos_satisfaction_vs_power", "jain_fairness_vs_power", "p5_user_rate_vs_power", "computation_time_vs_power", "sentences_per_second_vs_power"]:
    display(Image(filename=str(F / (f + ".png"))))'''),
    md("## Observations (numbers from `task3_summary_all_methods.csv`)\n" + "\n".join(lines)),
    md("Further reading of the figures is left to the README Task 3 section, which lists the conclusions together with the limitations "
       "(300 drops per point, QoS infeasibility at low power, DNN trained with one model for all powers, AMC assumptions above)."),
])
print("notebooks written to", OUT)
