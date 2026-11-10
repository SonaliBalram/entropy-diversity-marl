# Quick look at one training run: how each behaviour metric correlates with reward.

import argparse
import os

import numpy as np


# Load one run's logs and print the correlations.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str,
                        default=os.path.join("results", "baseline", "seed_42"))
    parser.add_argument("--burn-in", type=int, default=0,
                        help="Skip the first N episodes (warmup uses random "
                             "actions, so those episodes are not policy behaviour).")
    args = parser.parse_args()

    rewards = np.load(os.path.join(args.results, "episode_rewards.npy"))
    beh = np.load(os.path.join(args.results, "behaviour_history.npz"))

    n = len(rewards)
    lo = args.burn_in

    r = rewards[lo:]
    cos = beh["cosine_mean"][lo:]
    cos_raw = beh["cosine_raw_mean"][lo:]
    w2 = beh["w2_mean"][lo:]
    js = beh["js_mean"][lo:]
    sig = beh["mean_sigma"][lo:]
    corr = beh["correlation_mean"][lo:]

    print("=" * 68)
    print(f"{n} episodes ({len(r)} after burn-in of {lo})")
    print("=" * 68)

    # Print the correlation of one metric with reward.
    def rep(name, x):
        ok = np.isfinite(x) & np.isfinite(r)
        if ok.sum() < 3:
            print(f"  {name:24s}   (insufficient data)")
            return
        c = np.corrcoef(r[ok], x[ok])[0, 1]
        print(f"  {name:24s} mean {np.nanmean(x):8.4f} | "
              f"corr with episode reward {c:+.3f}")

    print("\nCORRELATION WITH EPISODE REWARD")
    print("  (negative corr = more of this metric -> WORSE reward)\n")
    rep("cosine (force)", cos)
    rep("cosine (raw 5-D)", cos_raw)
    rep("W2 behavioural sep.", w2)
    rep("JS divergence", js)
    rep("temporal correlation", corr)
    rep("policy sigma", sig)

    k = max(5, len(r) // 20)
    worst = np.argsort(r)[:k]
    best = np.argsort(r)[-k:]

    print(f"\nWORST {k} vs BEST {k} EPISODES")
    print(f"  {'metric':24s} {'worst':>10s} {'best':>10s} {'delta':>10s}")
    for name, x in [
        ("episode reward", r),
        ("cosine (force)", cos),
        ("cosine (raw 5-D)", cos_raw),
        ("W2 behavioural sep.", w2),
        ("policy sigma", sig),
    ]:
        a, b = np.nanmean(x[worst]), np.nanmean(x[best])
        print(f"  {name:24s} {a:10.4f} {b:10.4f} {b - a:+10.4f}")

    ok = np.isfinite(cos) & np.isfinite(r)
    thresh = np.nanpercentile(cos[ok], 90)
    homogenised = cos[ok] >= thresh
    print(f"\nHOMOGENISED EPISODES (top decile of force-cosine, >= {thresh:.3f})")
    print(f"  mean reward when homogenised : {r[ok][homogenised].mean():9.3f}")
    print(f"  mean reward otherwise        : {r[ok][~homogenised].mean():9.3f}")
    print(f"  difference                   : "
          f"{r[ok][homogenised].mean() - r[ok][~homogenised].mean():+9.3f}")
    print("\n  A large negative difference means homogenisation CAUSES the")
    print("  bad episodes -- i.e. the 'noise' is the phenomenon, not a defect.")


if __name__ == "__main__":
    main()
