# Does entropy and diversity spread from actions to trajectories to the arena? Compares each arm with its control.

import os
import re
import sys
import argparse

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "exp3_windowed"))

from token_diversity import tokenise
from action_utils import to_force
from windowed_diversity import (
    windowed_diversity, window_frequencies, entropy_bits, permutation_null_windowed
)


ARMS = {
    "A0":   "checkpoints/baseline",
    "A1":   "checkpoints/alpha_0.01",
    "A1r":  "checkpoints/a1r_both",
    "A2w":  "checkpoints/beta_3_w5",
    "A2cw": "checkpoints/beta_3_w5_augcritic",
}
ROWS = [("A1", "A0"), ("A1r", "A1"), ("A2w", "A0"), ("A2cw", "A2w")]

LEVELS = ["action", "trajectory", "arena"]
UNITS = {"ent_action": "nats", "ent_trajectory": "bits", "ent_arena": "bits",
         "div_action": "force", "div_trajectory": "bits", "div_arena": "bits"}

N_AGENTS, T_STEPS, N_LANDMARKS = 3, 25, 3
N_WINDOWS = 5
FORCE_DIMS = [1, 2, 3, 4]
GRID_EDGES = [-1 / 3, 1 / 3]
N_CELLS = 9
MATCH_TOL = 1.0
T_CRIT = 3.18
N_PERM = 20

GATE = dict(arm="A0", seed=42, name="ep2000", ret=-35.580, D_old=0.9306)

PUBLISHED = {
    ("A1", "A0"):    dict(old=(+0.0127, +1.72)),
    ("A1r", "A1"):   dict(old=(+0.0043, +1.31)),
    ("A2w", "A0"):   dict(win=(+0.0065, +0.43)),
    ("A2cw", "A2w"): dict(win=(+0.0375, +2.37)),
}


# Which 3x3 arena grid cell each agent is in.
def grid_cell(observations):
    pos = np.asarray(observations)[:, 2:4]
    ix = np.digitize(pos[:, 0], GRID_EDGES)
    iy = np.digitize(pos[:, 1], GRID_EDGES)
    return (3 * ix + iy).astype(np.int64)


# Load a trained learner from a checkpoint.
def load_policy(path):
    import torch
    from maddpg_system import MADDPGSystem
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = MADDPGSystem(
        n_agents=ck["n_agents"], observation_dim=ck["observation_dim"],
        action_dim=ck["action_dim"], hidden_dim=ck["hidden_dim"],
        gamma=ck["gamma"], tau=ck["tau"],
        twin_critics=ck.get("twin_critics", True),
        context_dim=ck.get("context_dim", 0), device="cpu",
    )
    m.load(path)
    m.actor.eval()
    ep = ck.get("best_episode")
    return m, (int(ep) if ep is not None else None)


# Run one checkpoint on the held-out layouts and record tokens, cells, forces and entropy.
def rollout(path, layout_seeds, stochastic, torch_seed):
    import torch
    import torch.nn.functional as Fn
    from environment import MPEEnvironment

    maddpg, best_ep = load_policy(path)
    env = MPEEnvironment(n_agents=N_AGENTS, max_cycles=T_STEPS)
    torch.manual_seed(torch_seed)

    R, TOK, CELL, FORCE, ENT, SIG = [], [], [], [], [], []
    for layout_seed in layout_seeds:
        obs = env.reset(seed=layout_seed)
        tok, cell, force, ent, sig, total = [], [], [], [], [], 0.0
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist = maddpg.actor.get_distribution(obs_t)
                if stochastic:
                    z = dist.sample()
                    log_jac = -Fn.softplus(-z) - Fn.softplus(z)
                    log_pi = dist.log_prob(z) - log_jac
                    ent.append((-log_pi[:, FORCE_DIMS]).sum(-1).numpy())
                    sig.append(dist.stddev[:, FORCE_DIMS].mean(-1).numpy())
                else:
                    z = dist.mean
                actions = torch.sigmoid(z).numpy()
            obs, reward, terminated, truncated, _ = env.step(actions)
            tok.append(tokenise(obs))
            cell.append(grid_cell(obs))
            force.append(to_force(actions))
            total += reward
            if terminated or truncated:
                break
        if len(tok) != T_STEPS:
            raise RuntimeError(f"episode length {len(tok)} != {T_STEPS} in {path}")
        R.append(total)
        TOK.append(tok)
        CELL.append(cell)
        FORCE.append(force)
        if stochastic:
            ENT.append(ent)
            SIG.append(sig)
    env.close()

    out = dict(returns=np.asarray(R),
               tokens=np.asarray(TOK, dtype=np.int8),
               cells=np.asarray(CELL, dtype=np.int8),
               forces=np.asarray(FORCE, dtype=np.float32),
               best_episode=np.array(-1 if best_ep is None else best_ep))
    if stochastic:
        out["entropy"] = np.asarray(ENT, dtype=np.float32)
        out["sigma"] = np.asarray(SIG, dtype=np.float32)
    return out


# Rollout results, from the cache if available.
def cached_rollout(cache_dir, arm, seed, name, path, layout_seeds, stochastic):
    kind = "stoch" if stochastic else "det"
    f = os.path.join(cache_dir, f"{arm}_s{seed}_{name}_{kind}.npz")
    if os.path.exists(f):
        return dict(np.load(f))
    out = rollout(path, layout_seeds, stochastic, torch_seed=10_000 + seed)
    np.savez_compressed(f, **out)
    return out


# (episode, path) for every checkpoint of one seed.
def ladder(root, seed):
    d = os.path.join(root, f"seed_{seed}")
    items = []
    for name in os.listdir(d):
        m = re.fullmatch(r"ckpt_ep(\d+)\.pt", name)
        if m:
            items.append((int(m.group(1)), os.path.join(d, name)))
    return sorted(items)


# Windowed D, average entropy and whole-episode D for each layout.
def token_measures(tokens, n_symbols):
    div, ent, old = [], [], []
    for tok in tokens:
        a = tok.T.astype(np.int64)
        div.append(windowed_diversity(a, N_WINDOWS, n_symbols)[0])
        f = window_frequencies(a, N_WINDOWS, n_symbols)
        ent.append(float(entropy_bits(f, axis=-1).mean()))
        old.append(windowed_diversity(a, 1, n_symbols)[0])
    return np.asarray(div), np.asarray(ent), np.asarray(old)


# Mean distance between agents' forces.
def pairwise_force_distance(forces):
    n = forces.shape[2]
    d = [np.linalg.norm(forces[:, :, i] - forces[:, :, j], axis=-1)
         for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(d))


# Diversity at the action, trajectory and arena levels.
def diversity_cells(det):
    tdiv, _, told = token_measures(det["tokens"], N_LANDMARKS)
    adiv, _, _ = token_measures(det["cells"], N_CELLS)
    return dict(div_action=pairwise_force_distance(det["forces"]),
                div_trajectory=float(tdiv.mean()),
                div_arena=float(adiv.mean()),
                D_old=float(told.mean()))


# Entropy at the action, trajectory and arena levels.
def entropy_cells(sto):
    _, tent, _ = token_measures(sto["tokens"], N_LANDMARKS)
    _, aent, _ = token_measures(sto["cells"], N_CELLS)
    return dict(ent_action=float(sto["entropy"].mean()),
                ent_trajectory=float(tent.mean()),
                ent_arena=float(aent.mean()),
                sigma=float(sto["sigma"].mean()))


# Chance levels of the trajectory and arena diversity, by shuffling.
def null_levels(det, rng):
    # Chance level for one kind of token.
    def one(tokens, k):
        v = [permutation_null_windowed(t.T.astype(np.int64), N_WINDOWS, k, N_PERM, rng)
             for t in tokens]
        return float(np.mean(v)) * np.log2(k)
    return dict(null_trajectory=one(det["tokens"], N_LANDMARKS),
                null_arena=one(det["cells"], N_CELLS))


# Treatment checkpoint whose return is closest to the control's.
def match(control_ret, treatment):
    lad = treatment["ladder"]
    peak_ep = max(lad, key=lambda p: p["ret"])["episode"]
    cands = lad + [treatment["best"]]
    cands = [p for p in cands if p["episode"] is not None and p["episode"] <= peak_ep]
    m = min(cands, key=lambda p: abs(p["ret"] - control_ret))
    return m, m["ret"] - control_ret


# Mean, spread and t-statistic of paired values.
def paired_t(x):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean()), float(sd), float(x.mean() / (sd / np.sqrt(len(x))))


# Print to the screen and to a report file at the same time.
class Tee:
    # Open the report file.
    def __init__(self, path):
        self.f = open(path, "w")

    # Print a line and write it to the file.
    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


# Build the matrix of differences, save it and plot it.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 128, 456, 789])
    ap.add_argument("--layouts", type=int, default=200)
    ap.add_argument("--extra-arm", action="append", default=[],
                    help="LABEL:checkpoint_root, e.g. ID:checkpoints/id_arm")
    ap.add_argument("--extra-row", action="append", default=[],
                    help="TREATMENT:CONTROL, e.g. ID:A0")
    ap.add_argument("--rungs", type=int, nargs="*", default=None,
                    help="(testing only) restrict ladders to these episodes")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "results", "propagation_matrix"))
    args = ap.parse_args()

    arms = dict(ARMS)
    for spec in args.extra_arm:
        k, v = spec.split(":", 1)
        arms[k] = v
    rows = list(ROWS) + [tuple(s.split(":", 1)) for s in args.extra_row]
    arms = {k: (v if os.path.isabs(v) else os.path.join(PROJECT_ROOT, v)) for k, v in arms.items()}

    cache_dir = os.path.join(args.out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    log = Tee(os.path.join(args.out_dir, "report.txt"))
    layout_seeds = [900_000_000 + i for i in range(args.layouts)]
    S = args.seeds

    log("=" * 78)
    log("VALIDATION GATE")
    log("=" * 78)
    g = GATE
    gpath = os.path.join(arms[g["arm"]], f"seed_{g['seed']}", "ckpt_ep2000.pt")
    det = cached_rollout(cache_dir, g["arm"], g["seed"], g["name"], gpath, layout_seeds, False)
    ret, dold = float(det["returns"].mean()), diversity_cells(det)["D_old"]
    ok = abs(ret - g["ret"]) < 5e-3 and abs(dold - g["D_old"]) < 5e-4
    log(f"  {g['arm']} seed {g['seed']} {g['name']}: return {ret:.3f} (want {g['ret']}), "
        f"D {dold:.4f} bits (want {g['D_old']})  ->  {'PASS' if ok else 'FAIL'}")
    if not ok and args.layouts == 200:
        log("  Stopping: the rollout does not reproduce the known values.")
        return

    treat_arms = {t for t, _ in rows}
    ev = {}
    for arm, root in arms.items():
        if not any(arm in r for r in rows):
            continue
        ev[arm] = {}
        for s in S:
            pts = []
            if arm in treat_arms:
                for ep, path in ladder(root, s):
                    if args.rungs is not None and ep not in args.rungs:
                        continue
                    d = cached_rollout(cache_dir, arm, s, f"ep{ep}", path, layout_seeds, False)
                    pts.append(dict(name=f"ep{ep}", episode=ep, path=path, ret=float(d["returns"].mean())))
            bpath = os.path.join(root, f"seed_{s}", "best.pt")
            d = cached_rollout(cache_dir, arm, s, "best", bpath, layout_seeds, False)
            be = int(d["best_episode"])
            best = dict(name="best", episode=(None if be < 0 else be), path=bpath,
                        ret=float(d["returns"].mean()))
            ev[arm][s] = dict(ladder=pts, best=best)
            log(f"  {arm:>5} seed {s:>3}: {len(pts):>2} rungs, best.pt (ep {best['episode']}) "
                f"return {best['ret']:.3f}")

    COLS = [f"ent_{l}" for l in LEVELS] + [f"div_{l}" for l in LEVELS]
    rng = np.random.default_rng(2026)
    table = {}
    extras = {}
    abs_vals = {}

    # All measures for one checkpoint.
    def cells_for(arm, s, point):
        det = cached_rollout(cache_dir, arm, s, point["name"], point["path"], layout_seeds, False)
        sto = cached_rollout(cache_dir, arm, s, point["name"], point["path"], layout_seeds, True)
        c = diversity_cells(det)
        c.update(entropy_cells(sto))
        c.update(null_levels(det, rng))
        return c

    log("\n" + "=" * 78)
    log("MATCHED PAIRS  (treatment checkpoint chosen on held-out return alone, ascending branch)")
    log("=" * 78)
    for t, c in rows:
        diffs = {k: [] for k in COLS + ["D_old", "sigma"]}
        avals = {k: ([], []) for k in COLS + ["sigma", "null_trajectory", "null_arena"]}
        gaps, names = [], []
        log(f"\n  {t} vs {c}")
        for s in S:
            cb = ev[c][s]["best"]
            m, gap = match(cb["ret"], ev[t][s])
            cc, tc = cells_for(c, s, cb), cells_for(t, s, m)
            for k in diffs:
                diffs[k].append(tc[k] - cc[k])
            for k in avals:
                avals[k][0].append(cc[k])
                avals[k][1].append(tc[k])
            gaps.append(gap)
            names.append(m["name"])
            flag = "OK" if abs(gap) <= MATCH_TOL else "OUT OF TOLERANCE (kept in the mean)"
            log(f"    seed {s:>3}: {c} best.pt {cb['ret']:8.3f}  vs  {t} {m['name']:>8} "
                f"{m['ret']:8.3f}  | gap {gap:+.3f} {flag}")
        table[(t, c)] = {k: diffs[k] for k in COLS}
        extras[(t, c)] = dict(D_old=diffs["D_old"], sigma=diffs["sigma"], gaps=gaps, names=names)
        abs_vals[(t, c)] = {k: (float(np.mean(a)), float(np.mean(b))) for k, (a, b) in avals.items()}

    # Format a paired difference, starring significant ones.
    def fmt(v):
        m, sd, tt = paired_t(v)
        star = "*" if abs(tt) >= T_CRIT else " "
        return f"{m:+9.4f} ({tt:+6.2f}){star}"

    for title, prefix in [("ENTROPY MATRIX  (stochastic pass)  -- how random each agent is", "ent"),
                          ("DIVERSITY MATRIX  (deterministic pass)  -- how differently agents behave", "div")]:
        log("\n" + "=" * 78)
        log(title)
        log("  cell = mean paired difference, treatment - control, over seeds   (t(3))   * = |t| >= 3.18")
        log("=" * 78)
        head = "  " + f"{'row':<14}" + "".join(
            f"{l.upper() + ' [' + UNITS[prefix + '_' + l] + ']':>22}" for l in LEVELS)
        log(head)
        for r in rows:
            log("  " + f"{r[0] + ' vs ' + r[1]:<14}" +
                "".join(f"{fmt(table[r][prefix + '_' + l]):>22}" for l in LEVELS))

    log("\n  Seeds with a positive difference (out of %d):" % len(S))
    for r in rows:
        log("    " + f"{r[0] + ' vs ' + r[1]:<14}" + "  ".join(
            f"{k}: {sum(np.asarray(table[r][k]) > 0)}" for k in COLS))

    log("\n" + "=" * 78)
    log("ABSOLUTE LEVELS  (control -> matched treatment, mean over seeds)")
    log("=" * 78)
    for r in rows:
        a = abs_vals[r]
        log(f"  {r[0]} vs {r[1]}")
        for k in COLS + ["sigma"]:
            log(f"    {k:<16} {a[k][0]:10.4f} -> {a[k][1]:10.4f}")
        log(f"    {'null traj (bits)':<16} {a['null_trajectory'][0]:10.4f} -> {a['null_trajectory'][1]:10.4f}"
            f"      {'null arena (bits)'} {a['null_arena'][0]:.4f} -> {a['null_arena'][1]:.4f}")

    log("\n" + "=" * 78)
    log("CHECK AGAINST PUBLISHED MATCHED-RETURN RESULTS")
    log("=" * 78)
    for r in rows:
        pub = PUBLISHED.get(r, {})
        m1, _, t1 = paired_t(table[r]["div_trajectory"])
        m2, _, t2 = paired_t(extras[r]["D_old"])
        line = f"  {r[0] + ' vs ' + r[1]:<14} windowed D {m1:+.4f} (t {t1:+.2f})"
        if "win" in pub:
            line += f"  [published {pub['win'][0]:+.4f}, t {pub['win'][1]:+.2f}]"
        line += f"   old D {m2:+.4f} (t {t2:+.2f})"
        if "old" in pub:
            line += f"  [published {pub['old'][0]:+.4f}, t {pub['old'][1]:+.2f}]"
        log(line)
    log("  Small differences mean the match picked a different checkpoint for some seed;")
    log("  the MATCHED PAIRS table above shows which.")

    np.savez(os.path.join(args.out_dir, "matrix.npz"),
             rows=np.array([f"{t} vs {c}" for t, c in rows]),
             cols=np.array(COLS), seeds=np.array(S),
             diffs=np.array([[table[r][k] for k in COLS] for r in rows]),
             D_old=np.array([extras[r]["D_old"] for r in rows]),
             gaps=np.array([extras[r]["gaps"] for r in rows]),
             matched=np.array([extras[r]["names"] for r in rows]))
    with open(os.path.join(args.out_dir, "matrix.csv"), "w") as f:
        f.write("row,column,unit,mean,sd,t3,n_positive\n")
        for r in rows:
            for k in COLS:
                m, sd, tt = paired_t(table[r][k])
                f.write(f"{r[0]} vs {r[1]},{k},{UNITS[k]},{m:.6f},{sd:.6f},{tt:.3f},"
                        f"{int(sum(np.asarray(table[r][k]) > 0))}\n")
    log(f"\nSaved -> {args.out_dir}/ (report.txt, matrix.npz, matrix.csv)")

    plot(rows, table, COLS, args.out_dir, log)


# Draw the two heatmaps of the matrix.
def plot(rows, table, COLS, out_dir, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed in this environment: skipped heatmap.png "
            "(pip install matplotlib, then re-run; the cache makes it instant).")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.9 * len(rows) + 1.8))
    for ax, prefix, title in [(axes[0], "ent", "Entropy (how random)"),
                              (axes[1], "div", "Diversity (how different)")]:
        T = np.array([[paired_t(table[r][f"{prefix}_{l}"])[2] for l in LEVELS] for r in rows])
        M = np.array([[paired_t(table[r][f"{prefix}_{l}"])[0] for l in LEVELS] for r in rows])
        im = ax.imshow(np.clip(T, -8, 8), cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
        for i in range(len(rows)):
            for j in range(len(LEVELS)):
                star = "*" if abs(T[i, j]) >= T_CRIT else ""
                ax.text(j, i, f"{M[i, j]:+.3g}{star}\nt={T[i, j]:+.1f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(T[i, j]) > 5 else "black")
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels([f"{l}\n[{UNITS[prefix + '_' + l]}]" for l in LEVELS])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{t} vs {c}" for t, c in rows] if prefix == "ent" else [])
        ax.set_title(title)
    fig.colorbar(im, ax=axes, shrink=0.8, label="paired t(3), clipped at ±8")
    path = os.path.join(out_dir, "heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    log(f"Saved -> {path}")


if __name__ == "__main__":
    main()
