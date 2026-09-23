"""
exp3_behaviour_matched.py

THE BEHAVIOURAL METRICS, COMPARED AT MATCHED COMPETENCE.

Why this file exists
--------------------
exp3_compare_a2_a2c.py section E compared the behavioural metrics
over the LAST 25% OF TRAINING. That is not a fair comparison: at
the end of training A2 sits near -35.7 and A2c near -32.0, so the
arms are at different points on the competence curve -- and the
hysteresis result already established that a MORE COLLAPSED policy
scores as MORE differentiated at the same return. Any apparent
diversity advantage measured at end-of-training may be nothing but
the difference in how far each arm has collapsed.

This script repeats the comparison the way the token measure is
already handled: at MATCHED RETURN, on the ascending branch, over a
grid of targets spanning the arms' common range. Same control, same
logic, applied to the measure family that was left unmatched.
"""
import numpy as np

S = [42, 128, 456, 789]
ARMS = [("A2", "beta_3"), ("A2c", "beta_3_augcritic")]
K = 1000          # smoothing window, episodes
TOL = 0.25        # reward units a match must fall within
W = 72


def smooth(x, k=K):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def load(arm, seed):
    r = smooth(np.load(f"results/{arm}/seed_{seed}/episode_rewards.npy"))
    z = np.load(f"results/{arm}/seed_{seed}/behaviour_history.npz")
    m = {k: smooth(z[k]) for k in
         ["cosine_mean", "euclidean_mean", "correlation_mean",
          "token_diversity_norm", "mean_sigma"]}
    return r, m


def match(r, target):
    """Index on the ASCENDING branch closest to target return."""
    peak = int(np.argmax(r))
    asc = r[:peak + 1]
    if len(asc) == 0:
        return None
    i = int(np.argmin(np.abs(asc - target)))
    return i if abs(asc[i] - target) <= TOL else None


DATA = {lab: {s: load(arm, s) for s in S} for lab, arm in ARMS}

print("=" * W)
print("BEHAVIOURAL METRICS AT MATCHED COMPETENCE  (A2 vs A2c)")
print("=" * W)
print(f"  Smoothing {K} episodes; ascending branch only; a target counts")
print(f"  for a seed only if BOTH arms match it within {TOL} reward units.")
print("  Higher cosine / euclidean = MORE diversity.")
print("  Higher correlation = LESS diversity.")
print()

targets = np.arange(-37.0, -27.5, 0.5)
rows = []
for t in targets:
    per = {k: [] for k in ["cosine_mean", "euclidean_mean",
                           "correlation_mean", "token_diversity_norm",
                           "mean_sigma"]}
    n = 0
    for s in S:
        rA, mA = DATA["A2"][s]
        rC, mC = DATA["A2c"][s]
        iA, iC = match(rA, t), match(rC, t)
        if iA is None or iC is None:
            continue
        n += 1
        for k in per:
            per[k].append(mC[k][iC] - mA[k][iA])
    if n >= 3:
        rows.append((t, n, {k: np.array(v) for k, v in per.items()}))

if not rows:
    raise SystemExit("no targets matched in both arms")

hdr = f"  {'target':>8s} {'n':>2s}   {'cosine':>9s} {'euclid':>9s} {'corr':>9s} {'D_norm':>9s} {'sigma':>9s}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for t, n, p in rows:
    print(f"  {t:8.1f} {n:2d}   " + " ".join(
        f"{p[k].mean():+9.4f}" for k in
        ["cosine_mean", "euclidean_mean", "correlation_mean",
         "token_diversity_norm", "mean_sigma"]))

print()
print("  POOLED over all matched (seed, target) points -- A2c minus A2:")
print("  " + "-" * (W - 4))
for k, name, direction in [
        ("cosine_mean", "pairwise cosine distance", "higher = more diverse"),
        ("euclidean_mean", "pairwise euclidean distance", "higher = more diverse"),
        ("correlation_mean", "temporal correlation", "higher = LESS diverse"),
        ("token_diversity_norm", "token D_norm", "higher = more diverse"),
        ("mean_sigma", "policy sigma", "confound check")]:
    v = np.concatenate([p[k] for _, _, p in rows])
    sd = v.std(ddof=1); se = sd / np.sqrt(len(v))
    print(f"  {name:28s} {v.mean():+8.4f} +/- {sd:6.4f}  "
          f"(n={len(v):3d} points)   {direction}")
print()
print("  NOTE: the pooled points are autocorrelated -- adjacent targets")
print("  are nearby episodes of the same run -- so the spread is a")
print("  descriptive range, not a standard error, and no t is quoted.")
print()
