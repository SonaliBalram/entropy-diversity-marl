"""
block4_checks.py  --  pre-training checks for the windowed diversity measure.

Run from the repo root (baseline_fixed/), in your macOS .venv:

    python exp3_windowed/block4_checks.py

No training. It rolls out the EXISTING baseline checkpoints on the same 200
held-out layouts as exp3_outcome.py / token_eval.py, saves the raw per-step
tokens, and reports three numbers that must be fixed BEFORE the new A2 runs:

  1. ZERO LEVEL       windowed D when agent identity carries no information
                      (tokens shuffled across agents within each window)
  2. DETECTION FLOOR  the up/down (hysteresis) gap at matched return on the
                      baseline ladder, measured with windowed D
  3. NOISE CHECK      episode-to-episode spread of D_win_norm at best.pt,
                      and whether beta = 3 keeps the old safety margin

Every number is also computed for the OLD whole-episode D, so the old
published values (0.0385-0.039 null, 0.0566 floor) act as a built-in check
that this script's procedure matches the one used before.

The rollout mirrors exp3_outcome.evaluate_checkpoint exactly (same
environment, checkpoint loading, deterministic actions, tokenise, layout
seeds). It skips the sigma pass, which Block 4 does not need.

Tokens are cached per checkpoint in results/exp3_windowed/block4_cache/,
so the run is resumable and later analyses need no new rollouts.
"""

import os
import re
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)                 # baseline_fixed/
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)

from windowed_diversity import windowed_diversity    # noqa: E402

LOG2_3 = np.log2(3)
N_WINDOWS = 5

# Pre-registered decision rules (written before seeing any output)
MATCH_TOL = 1.0          # reward units, same as exp3_outcome.py
STEP_REWARD = 1.12       # typical per-step base reward magnitude
BETA = 3.0
GATE = dict(seed=42, episode=2000, ret=-35.580, D=0.9306)   # known values


# =========================================================
# ROLLOUT  (heavy imports live here so the analysis can be tested alone)
# =========================================================

def rollout_tokens(path, layout_seeds, n_agents=3, max_cycles=25):
    """
    Returns tokens (L, T, n_agents) int8, returns (L,), old_bits (L,).
    Same procedure as exp3_outcome.evaluate_checkpoint, minus the sigma pass.
    """
    import torch
    from environment import MPEEnvironment
    from maddpg_system import MADDPGSystem
    from token_diversity import tokenise, episode_diversity

    ck = torch.load(path, map_location="cpu", weights_only=False)
    env = MPEEnvironment(n_agents=n_agents, max_cycles=max_cycles)
    maddpg = MADDPGSystem(
        n_agents=ck["n_agents"], observation_dim=ck["observation_dim"],
        action_dim=ck["action_dim"], hidden_dim=ck["hidden_dim"],
        gamma=ck["gamma"], tau=ck["tau"],
        twin_critics=ck.get("twin_critics", True),
        context_dim=ck.get("context_dim", 0), device="cpu",
    )
    maddpg.load(path)
    maddpg.actor.eval()
    maddpg.critic.eval()

    all_tokens, returns, old_bits = [], [], []
    for layout_seed in layout_seeds:
        obs = env.reset(seed=layout_seed)
        toks, total = [], 0.0
        while True:
            actions = maddpg.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(actions)
            toks.append(tokenise(obs))            # next_observations convention
            total += reward
            if terminated or truncated:
                break
        toks = np.asarray(toks)                   # [T, n_agents]
        if toks.shape != (max_cycles, n_agents):
            raise RuntimeError(f"unexpected token shape {toks.shape} in {path}")
        all_tokens.append(toks.astype(np.int8))
        returns.append(total)
        old_bits.append(episode_diversity(toks)[0])
    env.close()
    return np.stack(all_tokens), np.asarray(returns), np.asarray(old_bits)


def cached(cache_dir, key, path, layout_seeds):
    f = os.path.join(cache_dir, f"{key}.npz")
    if os.path.exists(f):
        z = np.load(f)
        return z["tokens"], z["returns"], z["old_bits"]
    tokens, returns, old_bits = rollout_tokens(path, layout_seeds)
    np.savez(f, tokens=tokens, returns=returns, old_bits=old_bits)
    return tokens, returns, old_bits


def ladder(root, seed):
    d = os.path.join(root, f"seed_{seed}")
    items = []
    for name in os.listdir(d):
        m = re.fullmatch(r"ckpt_ep(\d+)\.pt", name)
        if m:
            items.append((int(m.group(1)), os.path.join(d, name)))
    return sorted(items)


# =========================================================
# MEASURES ON CACHED TOKENS
# =========================================================

def per_layout_measures(tokens):
    """tokens (L, T, N) -> windowed bits, windowed norm, one-window bits (L,)."""
    win, win_n, one = [], [], []
    for tok in tokens:
        a = tok.T                                            # (N, T)
        d, dn, _ = windowed_diversity(a, n_windows=N_WINDOWS)
        win.append(d)
        win_n.append(dn)
        one.append(windowed_diversity(a, n_windows=1)[0])
    return np.asarray(win), np.asarray(win_n), np.asarray(one)


def shuffled_windowed(tok_nt, rng, n_windows=N_WINDOWS):
    """Shuffle tokens across agents WITHIN each window, keep the window's pool."""
    n, T = tok_nt.shape
    L = T // n_windows
    out = tok_nt.copy()
    for w in range(n_windows):
        block = out[:, w * L:(w + 1) * L].ravel()
        out[:, w * L:(w + 1) * L] = rng.permutation(block).reshape(n, L)
    return out


def zero_level(tokens, rng, n_perm):
    """Mean windowed D and mean old D over label-shuffled copies of every layout."""
    win, old = [], []
    for tok in tokens:
        a = tok.T
        for _ in range(n_perm):
            win.append(windowed_diversity(shuffled_windowed(a, rng), N_WINDOWS)[0])
            # old-D null: shuffle across agents over the WHOLE episode
            flat = rng.permutation(a.ravel()).reshape(a.shape)
            old.append(windowed_diversity(flat, n_windows=1)[0])
    return float(np.mean(win)), float(np.mean(old))


def hysteresis(points, key, tol=MATCH_TOL):
    """
    points: ladder entries (dicts with 'episode', 'ret', key), episode order.
    Peak = the rung with the highest held-out return. For every rung AFTER the
    peak (descending branch), find the ascending-branch rung (at or before the
    peak) with the closest return; keep the pair if within tol.
    Returns (mean descending-minus-ascending gap, number of pairs).
    """
    rets = np.array([p["ret"] for p in points])
    k = int(np.argmax(rets))
    asc, desc = points[:k + 1], points[k + 1:]
    diffs = []
    for d in desc:
        a = min(asc, key=lambda p: abs(p["ret"] - d["ret"]))
        if abs(a["ret"] - d["ret"]) <= tol:
            diffs.append(d[key] - a[key])
    return (float(np.mean(diffs)) if diffs else float("nan")), len(diffs)


def paired(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return float(np.mean(v)) if len(v) else float("nan"), float("nan"), float("nan")
    sd = v.std(ddof=1)
    return float(v.mean()), float(sd), float(v.mean() / (sd / np.sqrt(len(v))))


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(PROJECT_ROOT, "checkpoints", "baseline"))
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 128, 456, 789])
    ap.add_argument("--layouts", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=20)
    args = ap.parse_args()

    layout_seeds = [900_000_000 + i for i in range(args.layouts)]
    out_dir = os.path.join(PROJECT_ROOT, "results", "exp3_windowed")
    cache_dir = os.path.join(out_dir, "block4_cache")
    os.makedirs(cache_dir, exist_ok=True)
    rng = np.random.default_rng(2026)

    print("=" * 72)
    print(f"BLOCK 4 CHECKS  --  {args.label}  ({args.root})")
    print(f"windows = {N_WINDOWS} x {25 // N_WINDOWS} steps, layouts = {args.layouts}")
    print("=" * 72)

    per_seed, recon_err = {}, 0.0
    for seed in args.seeds:
        pts = []
        for ep, path in ladder(args.root, seed):
            tokens, rets, old_bits = cached(cache_dir, f"{args.label}_s{seed}_ep{ep}",
                                            path, layout_seeds)
            win, win_n, one = per_layout_measures(tokens)
            recon_err = max(recon_err, float(np.max(np.abs(one - old_bits))))
            pts.append(dict(episode=ep, ret=rets.mean(), Dwin=win.mean(), Dold=old_bits.mean()))
            print(f"  seed {seed:>3}  ep {ep:>6}  return {rets.mean():8.3f}   "
                  f"old D {old_bits.mean():.4f}   windowed D {win.mean():.4f}")
            if seed == GATE["seed"] and ep == GATE["episode"] and args.label == "baseline":
                ok = abs(rets.mean() - GATE["ret"]) < 5e-3 and abs(old_bits.mean() - GATE["D"]) < 5e-4
                print(f"    VALIDATION GATE (known: return {GATE['ret']}, D {GATE['D']}): "
                      f"{'PASS' if ok else 'FAIL -- stop and check before trusting anything below'}")

        best_path = os.path.join(args.root, f"seed_{seed}", "best.pt")
        tokens, rets, old_bits = cached(cache_dir, f"{args.label}_s{seed}_best",
                                        best_path, layout_seeds)
        win, win_n, _ = per_layout_measures(tokens)
        z_win, z_old = zero_level(tokens, rng, args.n_perm)
        per_seed[seed] = dict(
            ladder=pts, best_ret=rets.mean(), best_Dwin=win.mean(), best_Dold=old_bits.mean(),
            sd_new=float(win_n.std(ddof=1)), sd_old=float((old_bits / LOG2_3).std(ddof=1)),
            zero_win=z_win, zero_old=z_old)
        print(f"  seed {seed:>3}  best.pt   return {rets.mean():8.3f}   "
              f"old D {old_bits.mean():.4f}   windowed D {win.mean():.4f}\n")

    S = args.seeds
    print("=" * 72)
    print("REDUCTION CHECK  (windowed code with 1 window vs your token_diversity.py)")
    print("=" * 72)
    print(f"  max |difference| over every layout and checkpoint = {recon_err:.2e}   "
          f"{'PASS' if recon_err < 1e-9 else 'FAIL'}\n")

    print("=" * 72)
    print("1. ZERO LEVEL at best.pt  (agents' labels shuffled)")
    print("=" * 72)
    for s in S:
        p = per_seed[s]
        print(f"  seed {s:>3}: windowed {p['zero_win']:.4f} bits   old {p['zero_old']:.4f} bits")
    zw = np.mean([per_seed[s]["zero_win"] for s in S])
    zo = np.mean([per_seed[s]["zero_old"] for s in S])
    print(f"  MEAN      : windowed {zw:.4f} bits   old {zo:.4f} bits "
          f"(old published ~0.039)")
    print(f"  best.pt windowed D above its zero level: "
          f"{np.mean([per_seed[s]['best_Dwin'] for s in S]) - zw:+.4f} bits\n")

    print("=" * 72)
    print(f"2. DETECTION FLOOR  (descending minus ascending, matched within {MATCH_TOL})")
    print("=" * 72)
    fw, fo = [], []
    for s in S:
        gw, nw = hysteresis(per_seed[s]["ladder"], "Dwin")
        go, no = hysteresis(per_seed[s]["ladder"], "Dold")
        fw.append(gw)
        fo.append(go)
        print(f"  seed {s:>3}: windowed {gw:+.4f} ({nw} pairs)   old {go:+.4f} ({no} pairs)")
    mw, sw, tw = paired(fw)
    mo, so, to = paired(fo)
    print(f"  WINDOWED floor = {mw:+.4f} +/- {sw:.4f} bits   t(3) = {tw:+.2f}")
    print(f"  OLD floor      = {mo:+.4f} +/- {so:.4f} bits   t(3) = {to:+.2f}   "
          f"(old published +0.0566 -- should be close)\n")

    print("=" * 72)
    print("3. NOISE CHECK at best.pt  (spread of the normalised measure across layouts)")
    print("=" * 72)
    for s in S:
        p = per_seed[s]
        print(f"  seed {s:>3}: windowed sd {p['sd_new']:.4f}   old sd {p['sd_old']:.4f}   "
              f"ratio {p['sd_new'] / p['sd_old']:.2f}")
    sn = np.mean([per_seed[s]["sd_new"] for s in S])
    so_ = np.mean([per_seed[s]["sd_old"] for s in S])
    ratio = sn / so_
    print(f"  MEAN      : windowed sd {sn:.4f}   old sd {so_:.4f}   ratio {ratio:.2f}")
    print(f"  bonus noise at beta = {BETA:g}:  windowed {BETA * sn:.3f}   old {BETA * so_:.3f}   "
          f"(per-step reward ~{STEP_REWARD})")
    if ratio <= 1.0:
        verdict = "PASS -- windowed D is no noisier than old D, so beta = 3 keeps the old A2's margin"
    elif ratio <= 2.0:
        verdict = (f"CAUTION -- windowed D is {ratio:.2f}x noisier; beta = 3 is still under the old "
                   f"limit scaled by this ratio (~{6 / ratio:.1f}), but with less margin than old A2")
    else:
        verdict = (f"FAIL -- windowed D is {ratio:.2f}x noisier; the old limit of ~6 scales to "
                   f"~{6 / ratio:.1f}, so beta = 3 injects more noise than the old limit allowed")
    print(f"  {verdict}\n")

    np.savez(os.path.join(out_dir, f"block4_{args.label}.npz"),
             zero_windowed=zw, zero_old=zo, floor_windowed=mw, floor_old=mo,
             floor_windowed_per_seed=np.asarray(fw), floor_old_per_seed=np.asarray(fo),
             sd_windowed=sn, sd_old=so_, sd_ratio=ratio)
    print(f"Saved -> results/exp3_windowed/block4_{args.label}.npz "
          f"(tokens cached in results/exp3_windowed/block4_cache/)")


if __name__ == "__main__":
    main()