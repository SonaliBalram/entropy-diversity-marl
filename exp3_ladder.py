import argparse
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from exp3_outcome import evaluate_checkpoint


LAYOUT_BASE = 900_000_000


def rungs(root, seed):
    d = os.path.join(root, f"seed_{seed}")
    out = []
    for name in sorted(os.listdir(d)):
        m = re.fullmatch(r"ckpt_ep(\d+)\.pt", name)
        if m:
            out.append((int(m.group(1)), os.path.join(d, name)))
    out.sort()
    best = os.path.join(d, "best.pt")
    if os.path.exists(best):
        out.append((-1, best))
    return out


def fill(args):
    layout_seeds = [LAYOUT_BASE + i for i in range(args.layouts)]
    os.makedirs(args.cache_dir, exist_ok=True)

    jobs = []
    for arm in args.arms:
        root = os.path.join("checkpoints", arm)
        for seed in args.seeds:
            for episode, path in rungs(root, seed):
                key = f"{arm}__seed{seed}__ep{episode}.npz"
                jobs.append((arm, seed, episode, path,
                             os.path.join(args.cache_dir, key)))

    todo = [j for j in jobs if not os.path.exists(j[-1])]
    print(f"cache: {len(jobs) - len(todo)} / {len(jobs)} present, "
          f"{len(todo)} to do")

    start = time.time()
    done = 0
    for arm, seed, episode, path, out in todo:
        if time.time() - start > args.budget:
            print(f"  budget reached, stopping cleanly")
            break

        t0 = time.time()
        r = evaluate_checkpoint(
            path, layout_seeds, args.n_agents, args.max_cycles,
            args.n_perm, seed,
        )
        np.savez(
            out,
            arm=arm, seed=seed, episode=episode,
            alpha=r["alpha"], ret=r["ret"], D=r["D"],
            null=r["null"], sigma=r["sigma"],
            per_layout_return=r["per_layout_return"],
            per_layout_D=r["per_layout_D"],
        )
        done += 1
        print(f"  {arm:12s} seed {seed:>3} ep {episode:>6}  "
              f"return {r['ret']:8.3f}  D {r['D']:6.4f}  "
              f"sigma {r['sigma']:.4f}   [{time.time()-t0:4.1f}s]",
              flush=True)

    print(f"filled {done} this session; "
          f"{len(todo) - done} still outstanding")


def load_arm(cache_dir, arm, seeds):
    out = {}
    for seed in seeds:
        pts = []
        for f in os.listdir(cache_dir):
            if not f.startswith(f"{arm}__seed{seed}__"):
                continue
            z = np.load(os.path.join(cache_dir, f), allow_pickle=True)
            pts.append(dict(
                episode=int(z["episode"]), ret=float(z["ret"]),
                D=float(z["D"]), null=float(z["null"]),
                sigma=float(z["sigma"]),
            ))
        ladder = sorted([p for p in pts if p["episode"] >= 0],
                        key=lambda p: p["episode"])
        rep = [p for p in pts if p["episode"] < 0]
        out[seed] = dict(ladder=ladder,
                         reporting=rep[0] if rep else None)
    return out


def paired(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    sd = v.std(ddof=1) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    t = v.mean() / se if se and np.isfinite(se) and se > 0 else float("nan")
    return v.mean(), sd, t


def hysteresis_floor(arm, seeds, tol):
    per_seed, counts = [], []
    for seed in seeds:
        lad = arm[seed]["ladder"]
        if len(lad) < 4:
            continue
        peak = int(np.argmax([p["ret"] for p in lad]))
        asc, desc = lad[:peak + 1], lad[peak + 1:]
        if not asc or not desc:
            continue
        diffs = []
        for d in desc:
            a = min(asc, key=lambda p: abs(p["ret"] - d["ret"]))
            if abs(a["ret"] - d["ret"]) <= tol:
                diffs.append(d["D"] - a["D"])
        if diffs:
            per_seed.append(float(np.mean(diffs)))
            counts.append(len(diffs))
    return per_seed, counts


def report(args):
    ctrl = load_arm(args.cache_dir, args.control, args.seeds)
    trt = load_arm(args.cache_dir, args.treatment, args.seeds)

    print("=" * 72)
    print(f"LADDER COVERAGE")
    print("=" * 72)
    for lab, arm in [(args.control, ctrl), (args.treatment, trt)]:
        n = [len(arm[s]["ladder"]) for s in args.seeds]
        rep = sum(arm[s]["reporting"] is not None for s in args.seeds)
        print(f"  {lab:14s} rungs per seed {n}   best.pt present: {rep}/"
              f"{len(args.seeds)}")

    print()
    print("=" * 72)
    print("HYSTERESIS FLOOR -- measured inside the control arm alone")
    print("=" * 72)
    per_seed, counts = hysteresis_floor(ctrl, args.seeds, args.match_tol)
    if per_seed:
        m, sd, t = paired(per_seed)
        print(f"  descending minus ascending D at matched return")
        print(f"  per seed: " +
              "  ".join(f"{v:+.4f}" for v in per_seed))
        print(f"  matched pairs per seed: {counts}")
        print(f"  mean {m:+.4f} +/- {sd:.4f} bits   t({len(per_seed)-1}) "
              f"= {t:+.2f}")
        print(f"\n  This is the bias that matching on return alone leaves")
        print(f"  uncontrolled. Compare it with the effect below.")
    else:
        print("  not enough rungs cached yet")

    for mode in ["ascending", "any"]:
        print()
        print("=" * 72)
        print(f"MATCHED-RETURN DELTA D   [branch = {mode}]")
        print("=" * 72)
        dd, gaps, eps, unmatched = [], [], [], []
        for seed in args.seeds:
            c = ctrl[seed]["reporting"]
            if c is None:
                continue
            pts = list(trt[seed]["ladder"])
            if mode == "ascending" and pts:
                peak_ep = max(pts, key=lambda p: p["ret"])["episode"]
                pts = [p for p in pts if p["episode"] <= peak_ep]
            if trt[seed]["reporting"] is not None:
                pts = pts + [trt[seed]["reporting"]]
            if not pts:
                continue
            b = min(pts, key=lambda p: abs(p["ret"] - c["ret"]))
            gap = b["ret"] - c["ret"]
            if abs(gap) > args.match_tol:
                unmatched.append(seed)
            dd.append(b["D"] - c["D"])
            gaps.append(gap)
            eps.append(b["episode"])
            print(f"  seed {seed:>3}: control {c['ret']:8.3f} / D {c['D']:.4f}"
                  f"   vs treatment ep {b['episode']:>7} "
                  f"{b['ret']:8.3f} / D {b['D']:.4f}"
                  f"   gap {gap:+.3f}   dD {b['D']-c['D']:+.4f}")
        if dd:
            m, sd, t = paired(dd)
            print(f"\n  Delta D matched = {m:+.4f} +/- {sd:.4f} bits   "
                  f"t({len(dd)-1}) = {t:+.2f}")
            print(f"  mean |return gap| = {np.mean(np.abs(gaps)):.3f}")
            if unmatched:
                print(f"  WARNING seeds outside tolerance: {unmatched}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--arms", nargs="+",
                    default=["baseline", "alpha_0.01"])
    ap.add_argument("--control", default="baseline")
    ap.add_argument("--treatment", default="alpha_0.01")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 128, 456, 789])
    ap.add_argument("--layouts", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=25)
    ap.add_argument("--match-tol", type=float, default=1.0)
    ap.add_argument("--budget", type=float, default=150.0,
                    help="seconds; fill stops cleanly after this")
    ap.add_argument("--cache-dir",
                    default=os.path.join("results", "exp3_ladder_cache"))
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--max-cycles", type=int, default=25)
    args = ap.parse_args()

    if args.fill:
        fill(args)
    if args.report:
        report(args)
    if not args.fill and not args.report:
        ap.error("pass --fill, --report, or both")


if __name__ == "__main__":
    main()
