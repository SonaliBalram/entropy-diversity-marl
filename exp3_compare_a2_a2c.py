# Paired comparison of arms A2 and A2c across seeds.

import numpy as np, os

S = [42, 128, 456, 789]
ARMS = [("A2", "beta_3"), ("A2c", "beta_3_augcritic")]
CACHE = "results/exp3_ladder_cache"
W = 72


# Load one saved array for an arm and seed.
def arr(arm, seed, name):
    return np.load(f"results/{arm}/seed_{seed}/{name}")


# Load one behaviour metric for an arm and seed.
def beh(arm, seed, key):
    return np.load(f"results/{arm}/seed_{seed}/behaviour_history.npz")[key]


# Mean of the last fraction of a series.
def tail(x, frac=0.25):
    x = np.asarray(x, float)
    return np.nanmean(x[int((1 - frac) * len(x)):])


# Mean, spread and t-statistic of the paired difference b - a across seeds.
def paired(a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    sd = d.std(ddof=1)
    se = sd / np.sqrt(len(d))
    return d.mean(), sd, (d.mean() / se if se > 0 else np.nan)


# Print one row of the comparison table.
def line(label, a, b, fmt="{:9.4f}", note=""):
    m, sd, t = paired(a, b)
    print(f"  {label:32s} " + fmt.format(np.mean(a)) + " " +
          fmt.format(np.mean(b)) + "  " +
          f"{m:+8.4f} +/- {sd:6.4f}  t={t:+6.2f}  {note}")


# Print a table header.
def header(title):
    print()
    print("-" * W)
    print(title)
    print("-" * W)
    print(f"  {'quantity':32s} {'A2':>9s} {'A2c':>9s}  {'A2c - A2 (paired)':>28s}")


# Moving average over k episodes.
def smooth(x, k=500):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k


print("=" * W)
print("A2 vs A2c -- THOROUGH COMPARISON")
print("=" * W)
print("  Both arms: beta = 3, alpha = 0, terminal schedule, 50,000")
print("  episodes, batch 256, --update-every 2, seeds 42/128/456/789.")
print("  Sole difference: A2c's centralised critic additionally sees")
print("  the 3x3 running token-count matrix n_i(x)/T (69 -> 78 inputs).")
print("  Same seed => same actor initialisation and same episode layouts.")

header("A. TRAINING DYNAMICS  (from the 50,000-episode training curves)")
pk, pke, fin, deg = {}, {}, {}, {}
for lab, arm in ARMS:
    pk[lab], pke[lab], fin[lab], deg[lab] = [], [], [], []
    for s in S:
        r = smooth(arr(arm, s, "episode_rewards.npy"), 500)
        i = int(np.argmax(r))
        pk[lab].append(r[i]); pke[lab].append(i + 500)
        fin[lab].append(np.mean(r[-2000:]))
        deg[lab].append(r[-1] - r[i])
line("peak smoothed return", pk["A2"], pk["A2c"])
line("episode of peak", pke["A2"], pke["A2c"], fmt="{:9.0f}")
line("final return (last 2k)", fin["A2"], fin["A2c"])
line("degradation, peak to end", deg["A2"], deg["A2c"])

header("B. CRITIC HEALTH  (the augmentation's own diagnostics)")
for nm, f in [("twin-critic disagreement", "critic_disagreements.npy"),
              ("critic loss", "critic_losses.npy"),
              ("actor loss", "actor_losses.npy"),
              ("mean Q", "mean_q_values.npy")]:
    a = [tail(arr("beta_3", s, f)) for s in S]
    b = [tail(arr("beta_3_augcritic", s, f)) for s in S]
    line(nm + " (last 25%)", a, b)

a = [abs(tail(arr("beta_3", s, "mean_q_values.npy"))) /
     abs(tail(arr("beta_3", s, "episode_discounted.npy"))) for s in S]
b = [abs(tail(arr("beta_3_augcritic", s, "mean_q_values.npy"))) /
     abs(tail(arr("beta_3_augcritic", s, "episode_discounted.npy"))) for s in S]
line("Q vs G pessimism ratio", a, b)

print()
print("  Disagreement by training quarter (mean over seeds):")
for lab, arm in ARMS:
    q = []
    for k in range(4):
        v = []
        for s in S:
            x = arr(arm, s, "critic_disagreements.npy")
            n = len(x)
            v.append(np.nanmean(x[k * n // 4:(k + 1) * n // 4]))
        q.append(np.mean(v))
    print(f"    {lab:4s} " + "  ".join(f"Q{k+1} {v:6.3f}" for k, v in enumerate(q)))

header("C. DIVERSITY DURING TRAINING  (D_norm on the training rollouts)")
a = [tail(arr("beta_3", s, "episode_diversity_norm.npy"), 0.02) for s in S]
b = [tail(arr("beta_3_augcritic", s, "episode_diversity_norm.npy"), 0.02) for s in S]
line("D_norm, last 1,000 episodes", a, b)
a = [np.nanmax(smooth(arr("beta_3", s, "episode_diversity_norm.npy"), 1000)) for s in S]
b = [np.nanmax(smooth(arr("beta_3_augcritic", s, "episode_diversity_norm.npy"), 1000)) for s in S]
line("peak smoothed D_norm", a, b)

header("D. DIVERSITY, HELD OUT  (200 fixed layouts, deterministic policy)")
# Load one cached checkpoint evaluation.
def cget(arm, s, ep):
    return np.load(os.path.join(CACHE, f"{arm}__seed{s}__ep{ep}.npz"))
# Cached (episode, return, D) for every checkpoint of one seed.
def ladder(arm, s):
    out = []
    for f in os.listdir(CACHE):
        if f.startswith(f"{arm}__seed{s}__") and "ep-1" not in f:
            z = cget(arm, s, int(f.split("ep")[-1].split(".")[0]))
            out.append((int(z["episode"]), float(z["ret"]), float(z["D"])))
    return sorted(out)

a = [float(cget("beta_3", s, -1)["D"]) for s in S]
b = [float(cget("beta_3_augcritic", s, -1)["D"]) for s in S]
line("D at own best.pt (bits)", a, b)
a = [float(cget("beta_3", s, -1)["ret"]) for s in S]
b = [float(cget("beta_3_augcritic", s, -1)["ret"]) for s in S]
line("return at own best.pt", a, b)
a = [max(x[2] for x in ladder("beta_3", s)) for s in S]
b = [max(x[2] for x in ladder("beta_3_augcritic", s)) for s in S]
line("max D anywhere on ladder", a, b)
a = [max(x[1] for x in ladder("beta_3", s)) for s in S]
b = [max(x[1] for x in ladder("beta_3_augcritic", s)) for s in S]
line("max return on ladder", a, b)

print()
print("  Hysteresis (descending minus ascending D at matched return), per arm:")
for lab, arm in ARMS:
    per = []
    for s in S:
        L = ladder(arm, s)
        pk_i = int(np.argmax([x[1] for x in L]))
        asc, desc = L[:pk_i + 1], L[pk_i + 1:]
        d = [dd[2] - min(asc, key=lambda p: abs(p[1] - dd[1]))[2]
             for dd in desc
             if abs(min(asc, key=lambda p: abs(p[1] - dd[1]))[1] - dd[1]) <= 1.0]
        if d:
            per.append(np.mean(d))
    per = np.array(per)
    print(f"    {lab:4s} {per.mean():+.4f} +/- {per.std(ddof=1):.4f} bits "
          f"(n={len(per)} seeds)")

header("E. BEHAVIOURAL METRICS  (independent measure family, sigma-invariant)")
for nm, key in [("pairwise cosine distance", "cosine_mean"),
                ("pairwise euclidean distance", "euclidean_mean"),
                ("temporal correlation", "correlation_mean"),
                ("squashed W2", "w2_mean"),
                ("degenerate fraction", "degenerate_fraction")]:
    try:
        a = [tail(beh("beta_3", s, key)) for s in S]
        b = [tail(beh("beta_3_augcritic", s, key)) for s in S]
        line(nm, a, b)
    except KeyError:
        pass

header("F. STABILITY ACROSS SEEDS  (spread, not a paired test)")
print(f"  {'quantity':32s} {'A2 sd':>9s} {'A2c sd':>9s}")
for nm, vals in [("peak return", (pk["A2"], pk["A2c"])),
                 ("final return", (fin["A2"], fin["A2c"])),
                 ("return at best.pt",
                  ([float(cget("beta_3", s, -1)["ret"]) for s in S],
                   [float(cget("beta_3_augcritic", s, -1)["ret"]) for s in S])),
                 ("D at best.pt",
                  ([float(cget("beta_3", s, -1)["D"]) for s in S],
                   [float(cget("beta_3_augcritic", s, -1)["D"]) for s in S]))]:
    print(f"  {nm:32s} {np.std(vals[0], ddof=1):9.4f} {np.std(vals[1], ddof=1):9.4f}")
print()
