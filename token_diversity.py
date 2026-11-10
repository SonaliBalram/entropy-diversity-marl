import numpy as np


LANDMARK_BLOCK = slice(4, 10)


def tokenise(observations, n_landmarks=3):
    observations = np.asarray(observations)

    relative = observations[:, LANDMARK_BLOCK].reshape(
        observations.shape[0], n_landmarks, 2
    )

    squared_distance = (relative ** 2).sum(axis=-1)

    return squared_distance.argmin(axis=-1).astype(np.int64)


def frequency_vectors(tokens, n_landmarks=3):
    tokens = np.asarray(tokens)
    T, n_agents = tokens.shape

    f = np.zeros((n_agents, n_landmarks), dtype=np.float64)

    for i in range(n_agents):
        f[i] = np.bincount(
            tokens[:, i], minlength=n_landmarks
        ) / T

    return f


def shannon_entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0.0]

    return float(-(p * np.log2(p)).sum())


def token_diversity(f):
    f = np.asarray(f, dtype=np.float64)

    h_team = shannon_entropy(f.mean(axis=0))
    h_individual = np.mean([shannon_entropy(row) for row in f])

    return float(h_team - h_individual)


def max_diversity(n_agents, n_landmarks):
    return float(np.log2(min(n_agents, n_landmarks)))


def episode_diversity(tokens, n_landmarks=3):
    tokens = np.asarray(tokens)
    n_agents = tokens.shape[1]

    f = frequency_vectors(tokens, n_landmarks)

    d_bits = token_diversity(f)
    d_norm = d_bits / max_diversity(n_agents, n_landmarks)

    return d_bits, d_norm, f


def permutation_null(tokens, n_landmarks=3, n_permutations=200,
                     rng=None):
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

    joint = f / 3.0
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
