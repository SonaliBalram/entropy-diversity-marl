# Self-tests for windowed_diversity.py.

import numpy as np
from windowed_diversity import (windowed_diversity, permutation_null_windowed,
                                critic_context, critic_context_empty,
                                bonus_from_context)

LOG2_3 = np.log2(3)


# Reference value of windowed D, computed as a mutual information.
def mi_reference(tokens, n_windows, n_landmarks=3):
    tokens = np.asarray(tokens)
    n_agents, T = tokens.shape
    L = T // n_windows
    counts = np.zeros((n_agents, n_windows, n_landmarks))
    for i in range(n_agents):
        for t in range(T):
            counts[i, t // L, tokens[i, t]] += 1
    p = counts / counts.sum()
    p_w = p.sum(axis=(0, 2))
    p_zw = p.sum(axis=2)
    p_xw = p.sum(axis=0)
    total = 0.0
    for z in range(n_agents):
        for w in range(n_windows):
            for x in range(n_landmarks):
                if p[z, w, x] > 0:
                    total += p[z, w, x] * np.log2(p[z, w, x] * p_w[w] / (p_zw[z, w] * p_xw[w, x]))
    return total


# True if two numbers are equal within a tolerance.
def close(a, b, tol=1e-9):
    return abs(a - b) < tol


# Run every test case.
def run():
    tokens = np.zeros((3, 25), dtype=int)
    d, dn, dw = windowed_diversity(tokens)
    assert close(d, 0.0) and np.allclose(dw, 0.0), "test 1"
    print("1 all on L0            d_win = %.5f" % d)

    tokens = np.array([[0] * 25, [1] * 25, [2] * 25])
    d, dn, dw = windowed_diversity(tokens)
    assert close(d, LOG2_3) and close(dn, 1.0), "test 2"
    print("2 each owns one        d_win = %.5f  norm = %.5f" % (d, dn))

    swap = np.array([[0] * 15 + [1] * 10,
                     [1] * 15 + [0] * 10,
                     [2] * 25])
    d5, _, _ = windowed_diversity(swap, n_windows=5)
    d1, _, _ = windowed_diversity(swap, n_windows=1)
    assert close(d5, LOG2_3), "test 3 windowed"
    assert close(d1, 0.93766, tol=1e-5), "test 3 whole"
    print("3 swap                 windowed = %.5f  whole = %.5f" % (d5, d1))

    together = np.array([[0] * 15 + [1] * 10,
                         [0] * 15 + [1] * 10,
                         [2] * 25])
    d5, _, _ = windowed_diversity(together, n_windows=5)
    d1, _, _ = windowed_diversity(together, n_windows=1)
    assert close(d5, 0.91830, tol=1e-5) and close(d1, 0.91830, tol=1e-5), "test 4"
    print("4 moving together      windowed = %.5f  whole = %.5f" % (d5, d1))

    rng = np.random.default_rng(0)
    for trial in range(200):
        tokens = rng.integers(0, 3, size=(3, 25))
        d5, _, dw = windowed_diversity(tokens, n_windows=5)
        d1, _, _ = windowed_diversity(tokens, n_windows=1)
        assert close(d5, mi_reference(tokens, 5)), "test 5 windowed vs MI"
        assert close(d1, mi_reference(tokens, 1)), "test 5 whole vs MI"
        assert np.all(dw >= -1e-12) and np.all(dw <= LOG2_3 + 1e-12), "test 6 bounds"
        assert d5 >= d1 - 1e-12, "test 6 windowed >= whole"
    print("5 matches I(X;Z|W) from the joint table on 200 random episodes")
    print("6 bounds and windowed >= whole hold on 200 random episodes")

    try:
        windowed_diversity(np.zeros((3, 25), dtype=int), n_windows=4)
        raise AssertionError("test 7 should have raised")
    except ValueError:
        print("7 n_windows=4 correctly rejected")

    rng = np.random.default_rng(1)
    own = np.array([[0] * 25, [1] * 25, [2] * 25])
    z = permutation_null_windowed(own, rng=rng)
    assert 0.0 < z < 0.4, "test 8"
    print("8 shuffle null on differentiated agents = %.3f (normalised; chance, not 1.0)" % z)

    swap = np.array([[0] * 15 + [1] * 10, [1] * 15 + [0] * 10, [2] * 25]).T
    c5 = critic_context(swap[:5])
    c6 = critic_context(swap[:6])
    assert c5.shape == (11,), "test 9 shape"
    assert close(c5[:9].sum(), 3.0) and close(c5[9], 0.0) and close(c5[10], 0.2), "test 9 t=5"
    assert close(c6[:9].sum(), 0.6) and close(c6[9], 1 / 5) and close(c6[10], 0.4), "test 9 t=6"
    ctx0 = critic_context_empty()
    assert close(ctx0.sum(), 0.2), "test 9 t=0"
    for trial in range(200):
        tok = rng.integers(0, 3, size=(25, 3))
        paid = windowed_diversity(tok.T)[1]
        assert close(bonus_from_context(critic_context(tok)), paid), "test 9 recovery"
    print("9 A2c context: 11 numbers, rolls over at t=6 not t=5, final context "
          "reproduces the bonus on 200 random episodes")

    print("\nALL TESTS PASS")


if __name__ == "__main__":
    run()
