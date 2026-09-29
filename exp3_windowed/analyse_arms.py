"""
analyse_arms.py  --  the pre-registered comparison for the windowed Exp 3 arms.

Run from baseline_fixed/ in your .venv, e.g.

    # validation today: old A2 vs baseline (must reproduce the published old-D result)
    python exp3_windowed/analyse_arms.py --control baseline:checkpoints/baseline \
                                         --treatment A2old:checkpoints/beta_3

    # the real primary result, once new A2 has trained
    python exp3_windowed/analyse_arms.py --control baseline:checkpoints/baseline \
                                         --treatment A2w:checkpoints/beta_3_w5

    # A2c vs new A2
    python exp3_windowed/analyse_arms.py --control A2w:checkpoints/beta_3_w5 \
                                         --treatment A2cw:checkpoints/beta_3_w5_augcritic

Rollouts reuse block4_checks.py (same environment, loading, deterministic
actions, tokenise, 200 held-out layouts) and its token cache, so anything
already evaluated -- e.g. the baseline -- is not rolled out again.

PRE-REGISTERED RULES (fixed 29 Sept 2026, before any windowed A2 result)
------------------------------------------------------------------------
PRIMARY  matched-return Delta D on the WINDOWED measure.
         Per seed: control best.pt gives (r0, D0). Among the treatment's
         checkpoints on the ASCENDING branch (episode <= the episode of that
         run's highest held-out return), take the one whose held-out return
         is closest to r0. Delta D = D_treatment - D0. Chosen on RETURN ALONE.
         Match tolerance 1.0 reward units; a seed outside it stays in the
         mean and is flagged (dropping it after seeing its number is worse).
SECONDARY the same with no branch restriction, and the same on the OLD D.
CREDITED  an induced-diversity effect is claimed ONLY if the primary
         mean Delta D exceeds the windowed detection floor (0.041 bits) AND
         t(3) >= 3.18 (two-sided 5%, df = 3). Anything else is reported as
         not distinguishable from zero.
CHECKS   competence cost (best.pt return, paired); gaming (training-time
         sigma from episode 2,000 on -- the checkpoint ladder's range -- must
         stay under 0.02; the alpha = 0 floor is 0.0067).
         CORRECTION 29 Sept 2026, made on the OLD A2 validation run, before any
         windowed A2 result existed: the first version took the max over the
         whole run, which includes the initial sigma ~1 of an untrained network
         and trips in every arm. Restricted to episode 2,000 onwards.
"""

import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)

import block4_checks as b4                         # noqa: E402

FLOOR_WINDOWED = 0.041
FLOOR_OLD = 0.0566
ZERO_WINDOWED = 0.207
T_CRIT = 3.18
MATCH_TOL = 1.0
SIGMA_TRIPWIRE = 0.02
SIGMA_FROM_EPISODE = 2000       # same range as the checkpoint ladder


# =========================================================
# EVALUATION
# =========================================================

def best_episode(path):
    """Episode at which best.pt was saved (from its metadata), or None."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ep = ck.get("best_episode")
    return int(ep) if ep is not None else None


def evaluate_arm(label, root, seeds, layout_seeds, cache_dir):
    arm = {}
    for s in seeds:
        pts = []
        for ep, path in b4.ladder(root, s):
            tok, rets, old_bits = b4.cached(cache_dir, f"{label}_s{s}_ep{ep}", path, layout_seeds)
            win, _, _ = b4.per_layout_measures(tok)
            pts.append(dict(name=f"ep{ep}", episode=ep, ret=float(rets.mean()),
                            Dwin=float(win.mean()), Dold=float(old_bits.mean())))
        bpath = os.path.join(root, f"seed_{s}", "best.pt")
        tok, rets, old_bits = b4.cached(cache_dir, f"{label}_s{s}_best", bpath, layout_seeds)
        win, _, _ = b4.per_layout_measures(tok)
        best = dict(name="best.pt", episode=best_episode(bpath), ret=float(rets.mean()),
                    Dwin=float(win.mean()), Dold=float(old_bits.mean()))
        arm[s] = dict(ladder=pts, best=best)
        print(f"  {label:>10} seed {s:>3}: {len(pts)} rungs evaluated, best.pt "
              f"(ep {best['episode']}) return {best['ret']:.3f}", flush=True)
    return arm


# =========================================================
# THE STATISTIC  (pure function: tested on its own)
# =========================================================

def match(control_best, treatment, branch, key, tol=MATCH_TOL):
    """
    control_best: dict with ret and key.  treatment: dict(ladder=[...], best={...}).
    branch 'ascending' keeps candidates at or before the episode of the ladder's
    highest held-out return; 'any' keeps all. best.pt is a candidate if its
    episode qualifies. Returns (matched point, return gap, Delta D).
    """
    lad = treatment["ladder"]
    peak_ep = max(lad, key=lambda p: p["ret"])["episode"]
    cands = lad + [treatment["best"]]
    if branch == "ascending":
        cands = [p for p in cands if p["episode"] is not None and p["episode"] <= peak_ep]
    m = min(cands, key=lambda p: abs(p["ret"] - control_best["ret"]))
    return m, m["ret"] - control_best["ret"], m[key] - control_best[key]


def paired_t(x):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean()), float(sd), float(x.mean() / (sd / np.sqrt(len(x))))


def report(C, T, seeds, branch, key, clabel, tlabel, floor=None):
    print(f"\n  [{key}, branch = {branch}]")
    dd, gaps = [], []
    for s in seeds:
        m, gap, d = match(C[s]["best"], T[s], branch, key)
        dd.append(d)
        gaps.append(gap)
        flag = "OK" if abs(gap) <= MATCH_TOL else "OUT OF TOLERANCE"
        print(f"    seed {s:>3}: {clabel} best.pt {C[s]['best']['ret']:8.3f} / {C[s]['best'][key]:.4f}"
              f"  vs {tlabel} {m['name']:>8} {m['ret']:8.3f} / {m[key]:.4f}"
              f"  | gap {gap:+.3f} {flag} | dD {d:+.4f}")
    mean, sd, t = paired_t(dd)
    line = f"    Delta D = {mean:+.4f} +/- {sd:.4f} bits   t(3) = {t:+.2f}   mean |gap| = {np.mean(np.abs(gaps)):.3f}"
    if floor is not None:
        line += f"   (floor {floor})"
    print(line)
    return mean, sd, t, dd


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, help="label:checkpoint_root")
    ap.add_argument("--treatment", required=True, help="label:checkpoint_root")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 128, 456, 789])
    ap.add_argument("--layouts", type=int, default=200)
    args = ap.parse_args()

    layout_seeds = [900_000_000 + i for i in range(args.layouts)]
    out_dir = os.path.join(PROJECT_ROOT, "results", "exp3_windowed")
    cache_dir = os.path.join(out_dir, "block4_cache")
    os.makedirs(cache_dir, exist_ok=True)

    (cl, cr), (tl, tr) = [s.split(":", 1) for s in (args.control, args.treatment)]
    cr, tr = [p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p) for p in (cr, tr)]

    print("=" * 76)
    print(f"ANALYSIS  {tl}  vs  {cl}")
    print("=" * 76)
    C = evaluate_arm(cl, cr, args.seeds, layout_seeds, cache_dir)
    T = evaluate_arm(tl, tr, args.seeds, layout_seeds, cache_dir)
    S = args.seeds

    print("\n" + "=" * 76)
    print("PRIMARY  matched-return Delta D, WINDOWED measure, ascending branch")
    print("=" * 76)
    pm, psd, pt, pdd = report(C, T, S, "ascending", "Dwin", cl, tl, FLOOR_WINDOWED)
    credited = pm > FLOOR_WINDOWED and pt >= T_CRIT
    print(f"\n    PRE-REGISTERED VERDICT: "
          + ("INDUCED DIVERSITY CREDITED (above the 0.041 floor and t(3) >= 3.18)"
             if credited else
             "NOT CREDITED -- not distinguishable from zero under the registered rule "
             f"(needs mean > {FLOOR_WINDOWED} AND t(3) >= {T_CRIT})"))

    print("\n" + "=" * 76)
    print("SECONDARY")
    print("=" * 76)
    report(C, T, S, "any", "Dwin", cl, tl, FLOOR_WINDOWED)
    om, osd, ot, _ = report(C, T, S, "ascending", "Dold", cl, tl, FLOOR_OLD)
    report(C, T, S, "any", "Dold", cl, tl, FLOOR_OLD)

    print("\n" + "=" * 76)
    print("CHECKS")
    print("=" * 76)
    cost = [T[s]["best"]["ret"] - C[s]["best"]["ret"] for s in S]
    m, sd, t = paired_t(cost)
    print(f"  competence cost (best.pt return, {tl} - {cl}): {m:+.3f} +/- {sd:.3f}   t(3) = {t:+.2f}")

    exp_name = os.path.basename(os.path.normpath(tr))
    hits, missing = [], []
    for s in S:
        rdir = os.path.join(PROJECT_ROOT, "results", exp_name, f"seed_{s}")
        f, fs = os.path.join(rdir, "policy_stds.npy"), os.path.join(rdir, "update_steps.npy")
        if not (os.path.exists(f) and os.path.exists(fs)):
            missing.append(s)
            continue
        # Same range as the checkpoint ladder: from episode 2,000 on. The first
        # updates come from a freshly initialised network whose sigma starts near 1
        # and collapses to the 0.0067 floor within a few hundred episodes; that
        # transient happens in every arm and is not gaming.
        sig, steps = np.load(f), np.load(fs)
        settled = sig[steps > SIGMA_FROM_EPISODE * 25]
        mx = float(np.max(settled))
        print(f"  seed {s:>3}: max training sigma from episode {SIGMA_FROM_EPISODE:,} = {mx:.4f}   "
              f"(whole run incl. initial transient: {float(np.max(sig)):.4f})")
        if mx > SIGMA_TRIPWIRE:
            hits.append(s)
    if missing:
        print(f"  (no training sigma log found for seeds {missing})")
    print("  gaming tripwire: " + (f"HIT on seeds {hits} -- investigate before reporting" if hits
                                   else f"not hit (all under {SIGMA_TRIPWIRE})"))

    np.savez(os.path.join(out_dir, f"analysis_{tl}_vs_{cl}.npz"),
             primary_dD=np.asarray(pdd), primary_mean=pm, primary_t=pt,
             old_measure_mean=om, old_measure_t=ot, competence_cost=np.asarray(cost))
    print(f"\nSaved -> results/exp3_windowed/analysis_{tl}_vs_{cl}.npz")


if __name__ == "__main__":
    main()