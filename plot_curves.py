import argparse
import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib is not installed in this environment.\n"
                     "  source .venv/bin/activate && pip install matplotlib")

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_CYCLES = 25

LABELS = {
    "baseline":         "A0  (alpha = 0)",
    "alpha_0.01":       "A1  (alpha = 0.01)",
    "alpha_0.05":       "A1  (alpha = 0.05)",
    "a1r_both":         "A1r (gated, alpha = 0.01)",
    "beta_3":           "A2  (beta = 3)",
    "beta_3_augcritic": "A2c (beta = 3, aug. critic)",
    "beta_0_augcritic": "A2z (beta = 0, aug. critic)",
}
COLOURS = {
    "baseline": "#9a3b3b", "alpha_0.01": "#6f7888", "alpha_0.05": "#b0741f",
    "a1r_both": "#0f6e7a", "beta_3": "#5b4b9a", "beta_3_augcritic": "#3f7d3a",
    "beta_0_augcritic": "#9a7b2e",
}

METRICS = [
    ("episode_rewards",       "Task return (unshaped)",  "reward per episode", "ep"),
    ("policy_stds",           "Policy sigma",            "mean sigma (replay batch, 5 dims)", "up"),
    ("critic_disagreements",  "Twin-critic disagreement", "|Q1 - Q2|", "up"),
    ("critic_losses",         "Critic loss",             "loss", "up"),
    ("mean_q_values",         "Mean Q",                  "Q", "up"),
    ("gate_geomeans",         "A1r realised budget",     "geometric mean alpha_eff / alpha", "up"),
    ("gate_spreads",          "A1r allocation spread",   "interdecile log10 alpha_eff (decades)", "up"),
    ("episode_diversity_norm", "Training D (Exp 3 arms)", "D / log2(3)", "ep"),
]


def load_arm(arm, seeds, fname, kind, n_bins, bin_eps, alpha):
    rows = []
    for s in seeds:
        p = os.path.join(HERE, "results", arm, f"seed_{s}", fname + ".npy")
        if not os.path.exists(p):
            continue
        x = np.load(p).astype(np.float64)
        if kind == "ep":
            x = x[: n_bins * bin_eps]
            pad = n_bins * bin_eps - len(x)
            if pad > 0:
                x = np.concatenate([x, np.full(pad, np.nan)])
            rows.append(np.nanmean(x.reshape(n_bins, bin_eps), axis=1))
        else:
            steps = np.load(os.path.join(HERE, "results", arm, f"seed_{s}",
                                         "update_steps.npy"))
            b = np.minimum((steps // MAX_CYCLES) // bin_eps, n_bins - 1)
            b = b.astype(int)[: len(x)]
            if fname == "gate_geomeans":
                tot = np.bincount(b, np.log(x[: len(b)]), n_bins)
                cnt = np.bincount(b, None, n_bins)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rows.append(np.exp(tot / cnt) / alpha)
                continue
            tot = np.bincount(b, x[: len(b)], n_bins)
            cnt = np.bincount(b, None, n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                rows.append(tot / cnt)
    return np.array(rows) if rows else None


def draw(ax, centres, data, arms, title, ylabel, ref=None):
    for arm, y in data.items():
        m = np.nanmean(y, axis=0)
        c = COLOURS.get(arm)
        ax.plot(centres, m, color=c, lw=1.8,
                label=f"{LABELS.get(arm, arm)}  (n={len(y)})")
        if len(y) > 1:
            sd = np.nanstd(y, axis=0, ddof=1)
            ax.fill_between(centres, m - sd, m + sd, color=c, alpha=0.18,
                            lw=0)
    if ref is not None:
        ax.axhline(ref, color="0.5", lw=0.8, ls="--")
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("episode", fontsize=9)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["alpha_0.01", "a1r_both"])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 128, 456, 789])
    ap.add_argument("--episodes", type=int, default=50000)
    ap.add_argument("--bin", type=int, default=500,
                    help="episodes per plotted point")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n_bins = args.episodes // args.bin
    centres = (np.arange(n_bins) + 0.5) * args.bin
    out = args.out or os.path.join(HERE, "results", "plots",
                                   "__".join(args.arms))
    os.makedirs(out, exist_ok=True)

    alpha = 0.01
    cpath = os.path.join(HERE, "results", "a1r_pilot", "a1r_constants.json")
    if os.path.exists(cpath):
        alpha = json.load(open(cpath))["alpha"]

    panels = []
    for fname, title, ylabel, kind in METRICS:
        data = {}
        for arm in args.arms:
            y = load_arm(arm, args.seeds, fname, kind, n_bins, args.bin,
                         alpha)
            if y is not None:
                data[arm] = y
        if not data:
            continue
        ref = 1.0 if fname == "gate_geomeans" else None
        panels.append((fname, title, ylabel, data, ref))

        fig, ax = plt.subplots(figsize=(7, 4))
        draw(ax, centres, data, args.arms, title, ylabel, ref)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{fname}.png"), dpi=160)
        plt.close(fig)
        print(f"  {fname:24s} arms: {', '.join(data)}")

    if not panels:
        raise SystemExit("no arrays found -- check --arms against results/")

    ncol = 3 if len(panels) > 4 else 2
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow),
                             squeeze=False)
    for ax, (fname, title, ylabel, data, ref) in zip(axes.flat, panels):
        draw(ax, centres, data, args.arms, title, ylabel, ref)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=9, frameon=False)
    fig.text(0.5, 0.005, f"mean across seeds {args.seeds}; band = +/- 1 SD "
             f"across seeds; {args.bin}-episode bins", ha="center",
             fontsize=8, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(os.path.join(out, "overview.png"), dpi=160)
    plt.close(fig)
    print(f"\nwritten -> {out}/overview.png (+ one PNG per metric)")


if __name__ == "__main__":
    main()
