"""
Windowed token diversity for the rebuilt Experiment 3 (A2 and A2c).

For each window w (equal-length blocks of the episode):
    f_{i,w}(x) = fraction of window w that agent i spent nearest landmark x
    D_w        = H( mean_i f_{i,w} ) - mean_i H( f_{i,w} )        (bits)

Then:
    d_win      = mean_w D_w                                       (bits)
    d_win_norm = d_win / log2(n_landmarks)                         (in [0, 1])

With n_windows = 1 this is exactly the original whole-episode D.
Equivalently, d_win = I(X; Z | W): how much landmark + window tells you
about which agent it was.
"""
import numpy as np


def entropy_bits(p, axis=-1):
    """Shannon entropy in bits along `axis`, treating 0 * log(0) as 0."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log2(p), 0.0)
    return terms.sum(axis=axis)


def window_frequencies(tokens, n_windows=5, n_landmarks=3):
    """
    tokens : int array, shape (n_agents, T), values in {0, ..., n_landmarks-1}
    returns: float array, shape (n_agents, n_windows, n_landmarks)
             f[i, w, x] = fraction of window w that agent i spent at landmark x
    """
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be 2-D (n_agents, T), got shape {tokens.shape}")
    n_agents, T = tokens.shape
    if n_windows < 1 or T % n_windows != 0:
        raise ValueError(f"n_windows={n_windows} must divide T={T} exactly")
    if tokens.min() < 0 or tokens.max() >= n_landmarks:
        raise ValueError(f"tokens must lie in 0..{n_landmarks - 1}")

    L = T // n_windows                                   # steps per window
    windows = tokens.reshape(n_agents, n_windows, L)     # window w = steps w*L .. w*L+L-1
    one_hot = windows[..., None] == np.arange(n_landmarks)   # (n_agents, n_windows, L, n_landmarks)
    return one_hot.mean(axis=2)                          # average over the L steps


def windowed_diversity(tokens, n_windows=5, n_landmarks=3):
    """
    returns: d_win        - average window D, in bits
             d_win_norm   - d_win / log2(n_landmarks), in [0, 1]
             d_per_window - array (n_windows,), each D_w in bits
    """
    f = window_frequencies(tokens, n_windows, n_landmarks)   # (agents, windows, landmarks)

    f_bar = f.mean(axis=0)                           # group average per window: (windows, landmarks)
    h_group = entropy_bits(f_bar, axis=-1)           # H(f_bar_w): (windows,)
    h_indiv = entropy_bits(f, axis=-1).mean(axis=0)  # mean_i H(f_{i,w}): (windows,)

    d_per_window = h_group - h_indiv
    d_win = float(d_per_window.mean())
    d_win_norm = d_win / np.log2(n_landmarks)
    return d_win, d_win_norm, d_per_window


def permutation_null_windowed(tokens, n_windows=5, n_landmarks=3,
                              n_permutations=200, rng=None):
    """
    The zero level: shuffle which agent each token belongs to, WITHIN each
    window (each window keeps its own pool of tokens), and recompute.
    tokens: (n_agents, T).  Returns the mean d_win_norm over the shuffles.
    """
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


def critic_context(tokens_so_far, n_windows=5, n_landmarks=3, T=25):
    """
    The A2c critic context after t steps (t = number of tokens so far, 0..T).

    tokens_so_far: (t, n_agents) array of the tokens seen so far this episode.
    Returns 1-D float array of length n_agents * n_landmarks + 2:
        [ current window's counts / L   (n_agents x n_landmarks, row-major),
          sum of FINISHED windows' D / (n_windows * log2(n_landmarks)),
          window index (k + 1) / n_windows ]

    Boundary rule: at the last step of a window the counts still show that
    window in full; they roll into the finished sum on the NEXT step. So at
    t = T the context holds the last window's full counts, and the bonus can
    be recomputed exactly from it (bonus_from_context).
    """
    tokens_so_far = np.asarray(tokens_so_far, dtype=int)
    if tokens_so_far.ndim != 2 or tokens_so_far.shape[0] == 0:
        raise ValueError("tokens_so_far must be (t, n_agents) with t >= 1; "
                         "use critic_context_empty() at t = 0")
    t, n_agents = tokens_so_far.shape
    L = T // n_windows
    k = (t - 1) // L                                       # current window, 0-indexed
    counts = np.zeros((n_agents, n_landmarks))
    for s in range(k * L, t):
        counts[np.arange(n_agents), tokens_so_far[s]] += 1
    finished = 0.0
    for w in range(k):
        block = tokens_so_far[w * L:(w + 1) * L].T         # (n_agents, L)
        finished += windowed_diversity(block, 1, n_landmarks)[0]
    return np.concatenate([counts.ravel() / L,
                           [finished / (n_windows * np.log2(n_landmarks))],
                           [(k + 1) / n_windows]])


def critic_context_empty(n_agents=3, n_windows=5, n_landmarks=3):
    """Context at t = 0 (episode start): no counts, nothing finished, window 1."""
    return np.concatenate([np.zeros(n_agents * n_landmarks), [0.0], [1.0 / n_windows]])


def bonus_from_context(ctx, n_agents=3, n_windows=5, n_landmarks=3):
    """
    Recompute d_win_norm from the FINAL context (t = T). Used by the per-episode
    assertion in A2c: if this ever disagrees with the bonus actually paid, the
    context is not what the design says it is.
    """
    f = ctx[:n_agents * n_landmarks].reshape(n_agents, n_landmarks)   # last window's frequencies
    h_group = entropy_bits(f.mean(axis=0))
    h_indiv = entropy_bits(f, axis=-1).mean()
    last_norm = (h_group - h_indiv) / (n_windows * np.log2(n_landmarks))
    return float(ctx[n_agents * n_landmarks] + last_norm)