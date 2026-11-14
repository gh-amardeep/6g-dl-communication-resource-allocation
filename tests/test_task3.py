import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from task3_resource import allocation as al, dnn
from task3_resource.system import P, dbm_to_w, draw_channels, rb_rates, user_rates


@pytest.fixture(scope="module")
def A():
    return draw_channels(30, np.random.default_rng(7))


def check_valid(x, pw, pmax):
    assert x.shape == (P.U, P.N) and pw.shape == (P.N,)
    assert set(np.unique(x)) <= {0.0, 1.0}
    assert (x.sum(axis=0) <= 1).all()
    assert (pw >= 0).all() and pw.sum() <= pmax * (1 + 1e-9)


def test_channel_shape_and_noise(A):
    assert A.shape == (30, 8, 16) and (A > 0).all()
    assert 10 * np.log10(P.noise_w * 1e3) == pytest.approx(-174 + 10 * np.log10(180e3) + 9, abs=1e-9)


def test_rate_formula():
    a = np.full((8, 16), 1e9)
    x = np.zeros((8, 16)); x[0, :2] = 1
    pw = np.full(16, 0.1)
    assert user_rates(x, pw, a)[0] == pytest.approx(2 * 180e3 * np.log2(1 + 1e8))
    assert user_rates(x, pw, a)[1] == 0


@pytest.mark.parametrize("pdbm", [15, 27, 40])
def test_all_methods_valid(A, pdbm):
    pmax = dbm_to_w(pdbm)
    for a in A[:5]:
        for f in (al.round_robin, al.strongest_user_wf, al.milp_equal_power, al.milp_then_waterfilling):
            x, pw, _ = f(a, pmax)
            check_valid(x, pw, pmax)


def test_round_robin_pattern():
    x, pw, _ = al.round_robin(np.ones((8, 16)), 1.0)
    assert (x.sum(0) == 1).all() and (x.sum(1) == 2).all() and np.allclose(pw, 1 / 16)
    assert x[3, 3] == 1 and x[3, 11] == 1


def test_strongest_user(A):
    x, pw, _ = al.strongest_user_wf(A[0], 1.0)
    assert (x.argmax(0) == A[0].argmax(0)).all()


def test_waterfilling_kkt(A):
    a = A[0][A[0].argmax(0), np.arange(16)]
    pw = al.plain_waterfilling(a, 2.0)
    assert pw.sum() == pytest.approx(2.0, rel=1e-9)
    act = pw > 0
    level = pw[act] + 1 / a[act]
    assert np.allclose(level, level[0])                      # common water level on active RBs
    assert (1 / a[~act] >= level[0] - 1e-12).all()           # inactive RBs lie above the level


def test_milp_optimal_vs_bruteforce():
    rng = np.random.default_rng(3)
    a = draw_channels(1, rng)[0][:3, :5]                   # 3 users, 5 RBs
    pmax = dbm_to_w(30)
    r = P.B * np.log2(1 + pmax / 5 * a)
    best, bx = -1, None
    for own in itertools.product(range(3), repeat=5):
        R = np.array([sum(r[u, n] for n in range(5) if own[n] == u) for u in range(3)])
        if (R >= 5e5).all() and R.sum() > best:
            best = R.sum()
    from dataclasses import replace
    x, pw, info = al.milp_equal_power(a, pmax, p=replace(P, r_min=5e5))
    if best >= 0:
        assert info["qos_feasible"]
        assert (r * x).sum() == pytest.approx(best, rel=1e-6)


def test_milp_feasible_and_infeasible(A):
    a = A[0]
    x, pw, info = al.milp_equal_power(a, dbm_to_w(40))
    assert info["qos_feasible"] and (user_rates(x, pw, a) >= P.r_min - 1e-6).all()
    hopeless = a * 1e-8                                    # all users in deep outage
    _, _, info = al.milp_equal_power(hopeless, dbm_to_w(15))
    assert info["qos_feasible"] is False


def test_kkt_wf_meets_qos_when_feasible(A):
    pmax = dbm_to_w(30)
    n_checked = 0
    for a in A:
        x, pw0, info0 = al.milp_equal_power(a, pmax)
        if not info0["qos_feasible"]:
            continue
        pw, info = al.kkt_waterfilling(x, a, pmax)
        assert info["qos_feasible"]
        assert (user_rates(x, pw, a) >= P.r_min * (1 - 1e-6)).all() and pw.sum() <= pmax * (1 + 1e-9)
        # not worse than equal power
        assert user_rates(x, pw, a).sum() >= user_rates(x, pw0, a).sum() - 1e-3
        n_checked += 1
    assert n_checked > 5


def test_dnn_output_valid(A):
    net = dnn.AllocNet().double()
    pmax = dbm_to_w(30)
    x, pw, _ = dnn.predict(net, A[0], pmax)
    check_valid(x, pw, pmax)
    assert (x.sum(0) == 1).all()


def test_dnn_training_reduces_loss():
    net, hist = dnn.train(steps=300, batch=64, log_every=50)
    assert hist[-1]["loss"] < hist[0]["loss"]
    path = "/tmp/_dnn_test.pt"
    dnn.save(net, path, {"hidden": 256})
    net2 = dnn.load(path)
    a = draw_channels(1, np.random.default_rng(0))[0]
    x1, p1, _ = dnn.predict(net, a, 1.0)
    x2, p2, _ = dnn.predict(net2, a, 1.0)
    assert (x1 == x2).all() and np.allclose(p1, p2)


def test_amc_matches_task1_and_sentences():
    from pathlib import Path
    from task3_resource.amc_eval import AMCTable, load_amc, sentences_per_second
    root = Path(__file__).resolve().parents[1]
    amc = load_amc(root)
    tab = AMCTable(amc)
    for s in np.random.default_rng(0).uniform(-5, 35, 100):
        r = amc.select(float(s))
        m, e, b, o = tab.lookup(np.array([s]))
        assert m[0] == r["mcs"] and bool(o[0]) == r["outage"]
        assert (np.isnan(b[0]) and np.isnan(r["bler"])) or b[0] == pytest.approx(r["bler"])
    # high-SNR RB -> positive sentences/s; zero-power RBs deliver nothing
    a = np.full((P.U, P.N), 1e12)
    x = np.zeros((P.U, P.N)); x[np.arange(P.N) % P.U, np.arange(P.N)] = 1
    total, per_rb, _, _ = sentences_per_second(x, np.full(P.N, 1e-3), a, tab, 322.0)
    assert total > 0
    total0, _, _, _ = sentences_per_second(x, np.zeros(P.N), a, tab, 322.0)
    assert total0 == 0


def test_trained_dnn_file_loads_and_is_valid():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "models/resource_allocation/dnn_alloc.pt"
    if not path.exists():
        pytest.skip("trained DNN not present")
    net = dnn.load(path)
    a = draw_channels(1, np.random.default_rng(5))[0]
    x, pw, _ = dnn.predict(net, a, dbm_to_w(20))
    check_valid(x, pw, dbm_to_w(20))
