import numpy as np, os
R = "results/baseline/seed_42"
rew = np.load(f"{R}/episode_rewards.npy")
beh = np.load(f"{R}/behaviour_history.npz")
n = len(rew); BI = 200

def w(x, k=2000):
    m = (len(x)//k)*k
    return x[:m].reshape(-1, k)

print("="*72); print(f"SEED 42 — {n:,} episodes"); print("="*72)

rw = w(rew); wm = rw.mean(1)
best = wm.argmax()
print(f"  best 2k-window   : episodes {best*2}k-{(best+1)*2}k   mean {wm[best]:8.3f}")
print(f"  final 2k-window  : episodes {(len(wm)-1)*2}k-{len(wm)*2}k   mean {wm[-1]:8.3f}")
print(f"  degradation from peak                        {wm[-1]-wm[best]:+8.3f}")
print(f"  last 5,000 episodes (headline number)        {rew[-5000:].mean():8.3f} "
      f"+/- {rew[-5000:].std():.3f}")
print(f"  previous Experiment 1 result                  -40.384 +/- 1.678")

print("\n" + "="*72); print("HOMOGENISATION vs PERFORMANCE"); print("="*72)
cos  = beh["cosine_mean"][BI:];  cosr = beh["cosine_raw_mean"][BI:]
r    = rew[BI:]
sig  = beh["mean_sigma"][BI:];   w2 = beh["w2_mean"][BI:]
corr = beh["correlation_mean"][BI:]

def cc(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[ok], b[ok])[0,1]

print(f"  EPISODE level (n={len(r):,}):")
print(f"    cos (physical force) vs reward : {cc(cos, r):+.3f}")
print(f"    cos (raw 5-D, original metric) : {cc(cosr, r):+.3f}")
print(f"    policy sigma        vs reward  : {cc(sig, r):+.3f}")
print(f"    W2 separation       vs reward  : {cc(w2, r):+.3f}")
print(f"    temporal corr       vs reward  : {cc(corr, r):+.3f}")

cw, rww = w(cos).mean(1), w(r).mean(1)
print(f"\n  WINDOW level (2k windows, n={len(cw)}):")
print(f"    cos vs reward                  : {cc(cw, rww):+.3f}")

th = np.nanpercentile(cos, 90)
h = cos >= th
print(f"\n  Top decile of homogenisation (cos >= {th:.3f}):")
print(f"    mean reward homogenised        : {r[h].mean():8.3f}")
print(f"    mean reward otherwise          : {r[~h].mean():8.3f}")
print(f"    difference                     : {r[h].mean()-r[~h].mean():+8.3f}")

print("\n" + "="*72); print("TRAJECTORIES (10k-episode blocks)"); print("="*72)
print(f"  {'episodes':>12} {'reward':>9} {'cos':>8} {'sigma':>8} {'W2':>8} {'tcorr':>8}")
for k in range(0, n, 10000):
    s = slice(k, min(k+10000, n))
    print(f"  {k//1000:4d}k-{min(k+10000,n)//1000:3d}k "
          f"{rew[s].mean():9.2f} {np.nanmean(beh['cosine_mean'][s]):8.3f} "
          f"{np.nanmean(beh['mean_sigma'][s]):8.4f} {np.nanmean(beh['w2_mean'][s]):8.3f} "
          f"{np.nanmean(beh['correlation_mean'][s]):8.3f}")
