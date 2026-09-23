"""
token_diversity.py

Experiment 3: entropy at the trajectory level.

The measure
-----------
Each agent's 25-step episode is tokenised into a string over a
3-letter alphabet -- "which landmark am I nearest?" -- and the
diversity of the team is the gap between the entropy of the
team's pooled token distribution and the mean entropy of the
individual agents' token distributions:

    D(tau_1, ..., tau_N) = H(f_bar) - (1/N) sum_i H(f_i)

with f_i agent i's within-episode token frequency vector and
f_bar their mean. Three facts make this the measure to use:

  1. D >= 0 by concavity of H (Jensen). It is the generalised
     Jensen-Shannon divergence between the N token
     distributions.

  2. D = I(X ; Z) exactly, where Z is the agent index (drawn
     uniformly) and X the token. It is the mutual information
     between "which agent is this" and "what is it doing" --
     role differentiation stated as an information quantity.

  3. 0 <= D <= log2 min(N, |A|), a hard known bound, so the
     normalised D lies in [0, 1] and beta is in reward units.

Corner cases (all verified by the self-test below):

    all three agents on one landmark      D = 0
    each agent owns one landmark          D = log2(3) = 1.585
    all three roam over all three         D = 0
    partial differentiation (worked ex.)  D = 0.4553 bits

The third row is why the subtraction is doing real work: three
agents each roaming over everything is NOT role differentiation,
and team coverage entropy H(f_bar) alone would score it maximal.

Estimator note
--------------
The plug-in mutual information estimator is biased UPWARD,
unlike plug-in entropy. Miller-Madow gives

    bias ~ (|A| - 1)(N - 1) / 2n  nats,  n = N*T = 75
         = 4 / 150 = 0.0267 nats = 0.0385 bits

which is 2.4% of the maximum and constant across arms. Rather
than correcting analytically, report the PERMUTATION NULL:
shuffle the agent labels across the pooled token stream and
recompute D. Under the null of exchangeable agents this
estimates the bias directly, at the same T, the same alphabet
and the same empirical frequencies. See permutation_null().

No smoothing. Zero frequencies contribute zero by the 0 log 0
convention; Laplace smoothing would bias every f_i toward
uniform and shrink D artificially.

Lineage
-------
D = I(X;Z) is a discretised, tabular special case of the CDS
objective (Li et al., NeurIPS 2021), which maximises I(tau; id)
as an intrinsic reward in a parameter-shared MARL system. The
tokenise-then-entropy construction follows the Shannon Entropy
Framework of Guha & Velegol (J. Cheminformatics, 2023). Both
must be cited; neither is presented here as new.
"""

import numpy as np


# =========================================================
# OBSERVATION LAYOUT
# =========================================================
#
# simple_spread_v3, N = 3, observation_dim = 18:
#
#     [0:2]   self velocity
#     [2:4]   self position
#     [4:10]  landmark positions RELATIVE to self  (3 x 2)
#     [10:14] other agents' positions relative to self
#     [14:18] communication (unused)
#
# The landmark block is ordered by world.landmarks, which is
# the SAME ordering in every agent's observation, so token k
# means the same landmark for every agent. That is what makes
# the frequency vectors comparable across agents.
#
# Tokenising from the relative block means the token is a
# function of the agent's own observation alone -- no reach
# into env.unwrapped.world is needed.

LANDMARK_BLOCK = slice(4, 10)


def tokenise(observations, n_landmarks=3):
    """
    Nearest-landmark token for each agent at one timestep.

        x_i = argmin_{l in A} || p_i - l ||

    observations : [n_agents, observation_dim]
    returns      : [n_agents] int array of tokens in 0..K-1
    """

    observations = np.asarray(observations)

    relative = observations[:, LANDMARK_BLOCK].reshape(
        observations.shape[0], n_landmarks, 2
    )

    squared_distance = (relative ** 2).sum(axis=-1)

    return squared_distance.argmin(axis=-1).astype(np.int64)


# =========================================================
# FREQUENCY VECTORS
# =========================================================

def frequency_vectors(tokens, n_landmarks=3):
    """
    tokens : [T, n_agents] int
    returns: f [n_agents, K], each row summing to 1

        f_i(x) = n_i(x) / T
    """

    tokens = np.asarray(tokens)
    T, n_agents = tokens.shape

    f = np.zeros((n_agents, n_landmarks), dtype=np.float64)

    for i in range(n_agents):
        f[i] = np.bincount(
            tokens[:, i], minlength=n_landmarks
        ) / T

    return f


# =========================================================
# ENTROPY AND THE DIVERGENCE
# =========================================================

def shannon_entropy(p):
    """Shannon entropy in BITS, with 0 log 0 = 0."""

    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0.0]

    return float(-(p * np.log2(p)).sum())


def token_diversity(f):
    """
    D = H(f_bar) - (1/N) sum_i H(f_i),  in bits.

    f : [n_agents, K] row-stochastic
    """

    f = np.asarray(f, dtype=np.float64)

    h_team = shannon_entropy(f.mean(axis=0))
    h_individual = np.mean([shannon_entropy(row) for row in f])

    return float(h_team - h_individual)


def max_diversity(n_agents, n_landmarks):
    """log2 min(N, |A|) -- the hard upper bound on D."""

    return float(np.log2(min(n_agents, n_landmarks)))


def episode_diversity(tokens, n_landmarks=3):
    """
    One call from a finished episode.

    tokens : [T, n_agents] int

    returns (D_bits, D_normalised, f)
        D_normalised = D / log2 min(N, |A|)  in [0, 1]
    """

    tokens = np.asarray(tokens)
    n_agents = tokens.shape[1]

    f = frequency_vectors(tokens, n_landmarks)

    d_bits = token_diversity(f)
    d_norm = d_bits / max_diversity(n_agents, n_landmarks)

    return d_bits, d_norm, f


# =========================================================
# PERMUTATION NULL
# =========================================================

def permutation_null(tokens, n_landmarks=3, n_permutations=200,
                     rng=None):
    """
    The empirical zero for D.

    Pool all N*T tokens, permute them, and deal them back out
    in blocks of T. This destroys the association between
    agent identity and token while preserving EXACTLY:
    T, the alphabet, and the pooled empirical frequencies
    f_bar. What is left is the plug-in estimator's upward
    bias, measured rather than assumed.

    returns (mean, std) of normalised D under the null.
    """

    tokens = np.asarray(tokens)
    T, n_agents = tokens.shape

    if rng is None:
        rng = np.random.default_rng(0)

    pooled = tokens.reshape(-1)
    scale = max_diversity(n_agents, n_landmarks)

    values = np.empty(n_permutations, dtype=np.float64)

    for k in range(n_permutations):
        shuffled = rng.permutation(pooled).reshape(T, n_agents)
        values[k] = token_diversity(
            frequency_vectors(shuffled, n_landmarks)
        ) / scale

    return float(values.mean()), float(values.std())


# =========================================================
# SELF-TEST -- the corner cases and the worked example
# =========================================================

if __name__ == "__main__":

    def check(label, got, want, tolerance=1e-3):
        ok = abs(got - want) < tolerance
        print(f"  {'PASS' if ok else 'FAIL'}  {label:38s}"
              f"  got {got:.4f}  want {want:.4f}")
        return ok

    print("-" * 72)
    print("token_diversity self-test")
    print("-" * 72)

    passed = True

    # --- corner cases, stated directly as frequency vectors

    passed &= check(
        "all three on landmark 1",
        token_diversity(np.array([[1, 0, 0]] * 3)),
        0.0,
    )

    passed &= check(
        "each agent owns one landmark",
        token_diversity(np.eye(3)),
        np.log2(3),
    )

    passed &= check(
        "all three roam over all three",
        token_diversity(np.full((3, 3), 1 / 3)),
        0.0,
    )

    # --- the worked example from the design note

    f = np.array([
        [18, 5, 2],
        [6, 15, 4],
        [3, 2, 20],
    ], dtype=np.float64) / 25.0

    passed &= check("worked example, bits", token_diversity(f), 0.4553)
    passed &= check(
        "worked example, normalised",
        token_diversity(f) / max_diversity(3, 3),
        0.2873,
    )

    # --- the identity D = I(X ; Z), computed independently

    joint = f / 3.0                       # p(Z=i) = 1/3
    marginal_x = joint.sum(axis=0)
    marginal_z = joint.sum(axis=1)

    mutual_information = 0.0
    for i in range(3):
        for x in range(3):
            if joint[i, x] > 0:
                mutual_information += joint[i, x] * np.log2(
                    joint[i, x] / (marginal_z[i] * marginal_x[x])
                )

    passed &= check(
        "D == I(X;Z) computed from the joint",
        token_diversity(f), mutual_information,
    )

    # --- tokenisation from a synthetic observation block

    landmarks = np.array([[0.5, 0.5], [-0.5, 0.5], [0.0, -0.6]])
    positions = np.array([[0.45, 0.55], [-0.52, 0.40], [0.10, -0.55]])

    observations = np.zeros((3, 18), dtype=np.float32)
    observations[:, 2:4] = positions
    for i in range(3):
        observations[i, 4:10] = (landmarks - positions[i]).reshape(-1)

    tokens = tokenise(observations)
    passed &= check(
        "tokenise picks the nearest landmark",
        float(np.array_equal(tokens, np.array([0, 1, 2]))), 1.0,
    )

    # --- permutation null against the Miller-Madow prediction

    rng = np.random.default_rng(0)
    identical = np.tile(
        rng.integers(0, 3, size=25)[:, None], (1, 3)
    )
    null_mean, null_std = permutation_null(
        identical, n_permutations=2000, rng=rng
    )
    predicted = 0.0385 / max_diversity(3, 3)

    print(f"\n  permutation null: {null_mean:.4f} +/- {null_std:.4f}"
          f"   (Miller-Madow predicts {predicted:.4f})")

    print("-" * 72)
    print("ALL CHECKS PASSED" if passed else "SOMETHING FAILED")
    print("-" * 72)
