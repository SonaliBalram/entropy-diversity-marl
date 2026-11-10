# Exploratory: how much of the arena the team covers, per arm, at matched return.

import os
import sys
import argparse

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "exp3_windowed"))

import propagation_matrix as pm
from windowed_diversity import window_frequencies, entropy_bits, windowed_diversity

N_CELLS = pm.N_CELLS
N_WINDOWS = pm.N_WINDOWS
MEASURES = ["coverage_entropy", "cells_visited", "coverage_windowed"]
UNITS = {"coverage_entropy": "bits", "cells_visited": "cells", "coverage_windowed": "bits"}


# Coverage entropy, cells visited and windowed coverage for each layout.
def coverage_measures(cells):
    ent, vis, win, resid = [], [], [], []
    for ep in cells:
        flat = ep.ravel().astype(np.int64)
        fbar = np.bincount(flat, minlength=N_CELLS) / flat.size
        ent.append(float(entropy_bits(fbar)))
        vis.append(int((fbar > 0).sum()))

        a = ep.T.astype(np.int64)
        f = window_frequencies(a, N_WINDOWS, N_CELLS)
        team_w = entropy_bits(f.mean(axis=0), axis=-1)
        win.append(float(team_w.mean()))

        d_arena = windowed_diversity(a, N_WINDOWS, N_CELLS)[0]
        indiv = float(entropy_bits(f, axis=-1).mean())
        resid.append(abs(team_w.mean() - (d_arena + indiv)))
    return (dict(coverage_entropy=np.asarray(ent), cells_visited=np.asarray(vis, dtype=float),
                 coverage_windowed=np.asarray(win)),
            float(np.max(resid)))


# Compare coverage between each arm and its control and save the results.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 128, 456, 789])
    ap.add_argument("--layouts", type=int, default=200)
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results", "team_coverage"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(PROJECT_ROOT, "results", "propagation_matrix", "cache")
    log = pm.Tee(os.path.join(args.out_dir, "report.txt"))
    layout_seeds = [900_000_000 + i for i in range(args.layouts)]
    arms = {k: os.path.join(PROJECT_ROOT, v) for k, v in pm.ARMS.items()}
    S = args.seeds

    # Deterministic rollout of one checkpoint, from the cache if available.
    def det(arm, s, name, path):
        return pm.cached_rollout(cache_dir, arm, s, name, path, layout_seeds, False)

    treat_arms = {t for t, _ in pm.ROWS}
    ev = {}
    for arm, root in arms.items():
        ev[arm] = {}
        for s in S:
            pts = []
            if arm in treat_arms:
                for ep, path in pm.ladder(root, s):
                    d = det(arm, s, f"ep{ep}", path)
                    pts.append(dict(name=f"ep{ep}", episode=ep, path=path,
                                    ret=float(d["returns"].mean())))
            bpath = os.path.join(root, f"seed_{s}", "best.pt")
            d = det(arm, s, "best", bpath)
            be = int(d["best_episode"])
            ev[arm][s] = dict(ladder=pts, best=dict(name="best", episode=(None if be < 0 else be),
                                                    path=bpath, ret=float(d["returns"].mean())))

    published = None
    mfile = os.path.join(PROJECT_ROOT, "results", "propagation_matrix", "matrix.npz")
    if os.path.exists(mfile):
        z = np.load(mfile, allow_pickle=True)
        published = {str(r): [str(x) for x in m] for r, m in zip(z["rows"], z["matched"])}

    log("=" * 78)
    log("TEAM COVERAGE  (deterministic pass, 3x3 grid, matched return)  --  EXPLORATORY")
    log("=" * 78)

    worst_resid, all_match = 0.0, True
    table, absolute = {}, {}
    for t, c in pm.ROWS:
        row = f"{t} vs {c}"
        diffs = {k: [] for k in MEASURES}
        avals = {k: ([], []) for k in MEASURES}
        names, gaps = [], []
        for s in S:
            cb = ev[c][s]["best"]
            m, gap = pm.match(cb["ret"], ev[t][s])
            cm, r1 = coverage_measures(det(c, s, cb["name"], cb["path"])["cells"])
            tm, r2 = coverage_measures(det(t, s, m["name"], m["path"])["cells"])
            worst_resid = max(worst_resid, r1, r2)
            for k in MEASURES:
                diffs[k].append(float(tm[k].mean() - cm[k].mean()))
                avals[k][0].append(float(cm[k].mean()))
                avals[k][1].append(float(tm[k].mean()))
            names.append(m["name"])
            gaps.append(gap)
        if published is not None and row in published and published[row] != names:
            all_match = False
            log(f"  WARNING {row}: matched {names}, propagation matrix used {published[row]}")
        table[row] = diffs
        absolute[row] = avals
        log(f"\n  {row}   matched treatment checkpoints: {names}   "
            f"gaps: {', '.join(f'{g:+.3f}' for g in gaps)}")
        for k in MEASURES:
            mean, sd, tt = pm.paired_t(diffs[k])
            star = "*" if abs(tt) >= pm.T_CRIT else " "
            log(f"    {k:<18} {np.mean(avals[k][0]):7.3f} -> {np.mean(avals[k][1]):7.3f} {UNITS[k]:<5}"
                f"  diff {mean:+.4f} +/- {sd:.4f}   t(3) = {tt:+6.2f}{star}"
                f"   seeds positive: {sum(d > 0 for d in diffs[k])}/4")

    log("\n" + "=" * 78)
    log("CHECKS")
    log("=" * 78)
    log(f"  identity  mean_w H(fbar_w) = D_arena + mean H(f_iw):  max |residual| = {worst_resid:.2e}   "
        f"{'PASS' if worst_resid < 1e-9 else 'FAIL'}")
    if published is None:
        log("  matched checkpoints: results/propagation_matrix/matrix.npz not found, not compared")
    else:
        log(f"  matched checkpoints identical to the propagation matrix: {'PASS' if all_match else 'FAIL'}")
    log("  status: exploratory diagnostic, no pre-registered verdict; 12 comparisons, so about one")
    log("  |t| >= 3.18 ('*') is expected by chance. Read as a pattern alongside the propagation matrix.")

    rows = [f"{t} vs {c}" for t, c in pm.ROWS]
    D = np.array([[table[r][k] for k in MEASURES] for r in rows])
    A = np.array([[absolute[r][k] for k in MEASURES] for r in rows])
    np.savez(os.path.join(args.out_dir, "coverage.npz"), rows=np.array(rows),
             measures=np.array(MEASURES), seeds=np.array(S), diffs=D, absolute=A)
    with open(os.path.join(args.out_dir, "coverage.csv"), "w") as f:
        f.write("row,measure,control_mean,treatment_mean,diff_mean,diff_sd,t3\n")
        for r in rows:
            for k in MEASURES:
                mean, sd, tt = pm.paired_t(table[r][k])
                f.write(f"{r},{k},{np.mean(absolute[r][k][0]):.6f},{np.mean(absolute[r][k][1]):.6f},"
                        f"{mean:.6f},{sd:.6f},{tt:.4f}\n")
    log(f"\nSaved -> {args.out_dir}/ (report.txt, coverage.npz, coverage.csv)")


if __name__ == "__main__":
    main()
