"""Four benchmark allocators. Each returns (assign[U,N] in {0,1}, power[N] in W, info dict).

Formulation (per drop): max sum_{u,n} x_un B log2(1 + p_n a_un)
  s.t. sum_u x_un <= 1, sum_n p_n <= Pmax, R_u >= Rmin, x binary, p >= 0.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, vstack

from .system import P, Params


# ------------------------------------------------------------------ Method 1
def round_robin(a: np.ndarray, pmax_w: float, p: Params = P):
    U, N = a.shape
    x = np.zeros((U, N))
    x[np.arange(N) % U, np.arange(N)] = 1                      # RB n -> user n mod U
    return x, np.full(N, pmax_w / N), {}


# ------------------------------------------------------------------ water-filling tools
def _wf_power(inv_gain: np.ndarray, level) -> np.ndarray:
    return np.maximum(level - inv_gain, 0.0)


def _bisect(f, lo, hi, iters=200):
    """root of an increasing function f on [lo, hi]"""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def plain_waterfilling(g_assigned: np.ndarray, pmax_w: float) -> np.ndarray:
    """p_n = [L - 1/g_n]^+ with sum p_n = Pmax  (g = noise-normalised gain of the assigned user).

    KKT of max sum log2(1+p g) s.t. sum p <= P: g/((1+p g) ln2) = nu  =>  p = 1/(nu ln2) - 1/g.
    """
    inv = 1.0 / g_assigned
    L = _bisect(lambda L: _wf_power(inv, L).sum() - pmax_w, inv.min(), inv.max() + pmax_w)
    return _wf_power(inv, L)


def _min_power_for_rate(g: np.ndarray, r_target: float, B: float):
    """Smallest total power for one user to reach ``r_target`` on its RBs (gains ``g``), and its
    water level.  Level lam: p_n=[lam-1/g_n]^+, rate = B sum_{active} log2(lam g_n)."""
    inv = 1.0 / g
    rate = lambda lam: (B * np.log2(1 + _wf_power(inv, lam) * g)).sum() - r_target
    lam = _bisect(rate, inv.min(), inv.max() * 2 ** (r_target / B + 1) + 1.0)
    return _wf_power(inv, lam).sum(), lam


def kkt_waterfilling(assign: np.ndarray, a: np.ndarray, pmax_w: float, p: Params = P):
    """Water-filling for a FIXED assignment that also guarantees R_u >= Rmin when possible.

    Lagrangian: sum_u (1+mu_u) R_u(p) - nu (sum p - Pmax).  Stationarity on an RB n of user u:
        (1+mu_u) B /ln2 * g/(sigma2+p g) = nu   =>   p_n = [ L_u - 1/g_n ]^+,
        L_u = (1+mu_u) B / (nu ln2)  (per-user water level, L_u >= L = B/(nu ln2)).
    Complementary slackness: mu_u = 0 (L_u = L) if the QoS is slack, else L_u = lam_u, the level at
    which user u gets exactly Rmin.  Hence  L_u = max(L, lam_u), and the single scalar L is found
    by bisection so that sum p = Pmax.  If sum_u Pmin_u > Pmax (or a user has no RB) the QoS is
    infeasible for this assignment: Rmin is guaranteed to as many users as the budget allows and ``qos_feasible`` = False.
    """
    U, N = assign.shape
    owner = assign.argmax(axis=0)                              # user of every RB
    g = a[owner, np.arange(N)]
    lam, pm = np.zeros(U), np.full(U, np.inf)
    for u in range(U):
        idx = np.where(owner == u)[0]
        if len(idx):
            pm[u], lam[u] = _min_power_for_rate(g[idx], p.r_min, p.B)
    feasible = bool(np.isfinite(pm).all() and pm.sum() <= pmax_w)
    if not feasible:
        # QoS infeasible for this assignment: guarantee Rmin to as many users as the budget allows
        # (cheapest first); the others only share the remaining water (reported as QoS-infeasible).
        order = np.argsort(pm)
        keep, tot = [], 0.0
        for u in order:
            if np.isfinite(pm[u]) and tot + pm[u] <= pmax_w:
                keep.append(u); tot += pm[u]
        lam[[u for u in range(U) if u not in keep]] = 0.0
    inv = 1.0 / g
    lev = lambda L: np.maximum(L, lam[owner])
    pw = lambda L: _wf_power(inv, lev(L))
    L = _bisect(lambda L: pw(L).sum() - pmax_w, 0.0, inv.max() + pmax_w)
    return pw(L), {"qos_feasible": feasible}


# ------------------------------------------------------------------ Method 2
def strongest_user_wf(a: np.ndarray, pmax_w: float, p: Params = P):
    U, N = a.shape
    x = np.zeros((U, N))
    owner = a.argmax(axis=0)
    x[owner, np.arange(N)] = 1
    return x, plain_waterfilling(a[owner, np.arange(N)], pmax_w), {}


# ------------------------------------------------------------------ Method 3
def milp_equal_power(a: np.ndarray, pmax_w: float, p: Params = P, time_limit: float = 20.0):
    """Exact MILP with equal power.  r_un = B log2(1 + (Pmax/N) a_un) are constants, so
        max sum r_un x_un  s.t. sum_u x_un <= 1,  sum_n r_un x_un >= Rmin (all u),  x binary.
    If that is infeasible, the infeasibility is reported (info['qos_feasible']=False) and the best-effort
    solution (maximise number of QoS-satisfied users, then sum rate) is returned so metrics remain defined."""
    U, N = a.shape
    r = p.B * np.log2(1 + (pmax_w / N) * a)                    # (U,N)
    rv = r.reshape(-1)                                         # x index = u*N + n
    one_user = csr_matrix((np.ones(U * N), (np.tile(np.arange(N), U), np.arange(U * N))), shape=(N, U * N))
    qos = csr_matrix((r.reshape(-1), (np.repeat(np.arange(U), N), np.arange(U * N))), shape=(U, U * N))
    opts = {"time_limit": time_limit}
    res = milp(-rv, constraints=[LinearConstraint(one_user, -np.inf, 1),
                                 LinearConstraint(qos, p.r_min, np.inf)],
               integrality=np.ones(U * N), bounds=Bounds(0, 1), options=opts)
    if res.status == 0 and res.x is not None:
        return np.round(res.x).reshape(U, N), np.full(N, pmax_w / N), {"qos_feasible": True}
    # infeasible -> best effort: variables [x (U*N), y (U)], maximise W*sum(y) + sum rate
    W = 1e3 * rv.max() * N
    one_user2 = hstack([one_user, csr_matrix((N, U))])
    qos2 = hstack([qos, csr_matrix((-p.r_min * np.eye(U)))])      # sum r x - Rmin y >= 0
    c = np.concatenate([-rv, -W * np.ones(U)])
    res2 = milp(c, constraints=[LinearConstraint(one_user2, -np.inf, 1),
                                LinearConstraint(qos2, 0, np.inf)],
                integrality=np.ones(U * N + U), bounds=Bounds(0, 1), options=opts)
    x = np.round(res2.x[:U * N]).reshape(U, N)
    return x, np.full(N, pmax_w / N), {"qos_feasible": False}


# ------------------------------------------------------------------ Method 4
def milp_then_waterfilling(a: np.ndarray, pmax_w: float, p: Params = P, milp_result=None):
    """MILP assignment (from Method 3) followed by QoS-aware KKT water-filling."""
    x, _, info = milp_result if milp_result is not None else milp_equal_power(a, pmax_w, p)
    pw, info2 = kkt_waterfilling(x, a, pmax_w, p)
    return x, pw, {"qos_feasible": info2["qos_feasible"], "milp_qos_feasible": info["qos_feasible"]}


def timed(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - t0
