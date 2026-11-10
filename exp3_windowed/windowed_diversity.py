# Windowed token diversity: D measured within each of several windows of an episode.

import numpy as np


# Shannon entropy in bits, treating 0 log 0 as 0.
def entropy_bits(p, axis=-1):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log2(p), 0.0)
    return terms.sum(axis=axis)


# Fraction of each window each agent spent nearest each landmark.
def window_frequencies(tokens, n_windows=5, n_landmarks=3):
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be 2-D (n_agents, T), got shape {tokens.shape}")
    n_agents, T = tokens.shape
    if n_windows < 1 or T % n_windows != 0:
        raise ValueError(f"n_windows={n_windows} must divide T={T} exactly")
    if tokens.min() < 0 or tokens.max() >= n_landmarks:
        raise ValueError(f"tokens must lie in 0..{n_landmarks - 1}")

    L = T // n_windows
    windows = tokens.reshape(n_agents, n_windows, L)
    one_hot = windows[..., None] == np.arange(n_landmarks)
    return one_hot.mean(axis=2)


# Average D over windows, its normalised value, and D per window.
def windowed_diversity(tokens, n_windows=5, n_landmarks=3):
    f = window_frequencies(tokens, n_windows, n_landmarks)

    f_bar = f.mean(axis=0)
    h_group = entropy_bits(f_bar, axis=-1)
    h_indiv = entropy_bits(f, axis=-1).mean(axis=0)

    d_per_window = h_group - h_indiv
    d_win = float(d_per_window.mean())
    d_win_norm = d_win / np.log2(n_landmarks)
    return d_win, d_win_norm, d_per_window


# Chance level of windowed D, shuffling agents within each window.
def permutation_null_windowed(tokens, n_windows=5, n_landmarks=3,
                              n_permutations=200, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    tokens = np.asarray(tokens)
    n_agents, T = tokens.shape
    L = T // n_windows
    vals = []
    for _ in range(n_permutations):
        s = tokens.copy()
        for w in range(n_windows):
            block = s[:, w * L:(w + 1) * L].ravel()
            s[:, w * L:(w + 1) * L] = rng.permutation(block).reshape(n_agents, L)
        vals.append(windowed_diversity(s, n_windows, n_landmarks)[1])
    return float(np.mean(vals))


# The augmented critic's context after t steps.
def critic_context(tokens_so_far, n_windows=5, n_landmarks=3, T=25):
    tokens_so_far = np.asarray(tokens_so_far, dtype=int)
    if tokens_so_far.ndim != 2 or tokens_so_far.shape[0] == 0:
        raise ValueError("tokens_so_far must be (t, n_agents) with t >= 1; "
                         "use critic_context_empty() at t = 0")
    t, n_agents = tokens_so_far.shape
    L = T // n_windows
    k = (t - 1) // L
    counts = np.zeros((n_agents, n_landmarks))
    for s in range(k * L, t):
        counts[np.arange(n_agents), tokens_so_far[s]] += 1
    finished = 0.0
    for w in range(k):
        block = tokens_so_far[w * L:(w + 1) * L].T
        finished += windowed_diversity(block, 1, n_landmarks)[0]
    return np.concatenate([counts.ravel() / L,
                           [finished / (n_windows * np.log2(n_landmarks))],
                           [(k + 1) / n_windows]])


# The augmented critic's context at the start of an episode.
def critic_context_empty(n_agents=3, n_windows=5, n_landmarks=3):
    return np.concatenate([np.zeros(n_agents * n_landmarks), [0.0], [1.0 / n_windows]])


# Recompute the bonus from the final context, as a consistency check.
def bonus_from_context(ctx, n_agents=3, n_windows=5, n_landmarks=3):
    f = ctx[:n_agents * n_landmarks].reshape(n_agents, n_landmarks)
    h_group = entropy_bits(f.mean(axis=0))
    h_indiv = entropy_bits(f, axis=-1).mean()
    last_norm = (h_group - h_indiv) / (n_windows * np.log2(n_landmarks))
    return float(ctx[n_agents * n_landmarks] + last_norm)
