# Pre-registered comparison of the windowed Experiment 3 arms at matched return.

import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)

import block4_checks as b4

FLOOR_WINDOWED = 0.041
FLOOR_OLD = 0.0566
ZERO_WINDOWED = 0.207
T_CRIT = 3.18
MATCH_TOL = 1.0
SIGMA_TRIPWIRE = 0.02
SIGMA_FROM_EPISODE = 2000


# Episode at which best.pt was saved, or None.
def best_episode(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ep = ck.get("best_episode")
    return int(ep) if ep is not None else None


# Evaluate every checkpoint of every seed of one arm.
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


# Treatment checkpoint whose return is closest to the control's best.
def match(control_best, treatment, branch, key, tol=MATCH_TOL):
    lad = treatment["ladder"]
    peak_ep = max(lad, key=lambda p: p["ret"])["episode"]
    cands = lad + [treatment["best"]]
    if branch == "ascending":
        cands = [p for p in cands if p["episode"] is not None and p["episode"] <= peak_ep]
    m = min(cands, key=lambda p: abs(p["ret"] - control_best["ret"]))
    return m, m["ret"] - control_best["ret"], m[key] - control_best[key]


# Mean, spread and t-statistic of paired values.
def paired_t(x):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean()), float(sd), float(x.mean() / (sd / np.sqrt(len(x))))


# Print the matched-return difference for one setting.
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


# Evaluate both arms, print the comparison and save it.
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
