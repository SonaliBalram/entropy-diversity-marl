# Make every figure in Chapter 4 of the report.

import os
import re
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "exp3_windowed"))
from windowed_diversity import windowed_diversity

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "report_figures")
SEEDS = [42, 128, 456, 789]
STEPS_PER_EPISODE = 25
SIGMA_FLOOR = 0.0067

COLOURS = {"baseline": "#2a78d6", "alpha_0.01": "#eb6834", "beta_3_w5": "#1baf7a"}
LABELS = {"baseline": r"A0 ($\alpha=0$)", "alpha_0.01": r"A1 ($\alpha=0.01$)", "beta_3_w5": "A2w"}
DETECTION_FLOOR = 0.041

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888780", "axes.labelcolor": "#2C2C2A",
    "xtick.color": "#5F5E5A", "ytick.color": "#5F5E5A",
    "grid.color": "#D3D1C7", "grid.linewidth": 0.5,
})


# Save a figure as PDF and PNG.
def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name + ".pdf"))
    fig.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote results/report_figures/{name}.pdf and .png")


# Figure 4.0 (compact): the experimental logic as a flow diagram.
def figure_4_0_compact():
    from matplotlib.patches import FancyBboxPatch

    INK, GREY, GREEN = "#2C2C2A", "#B4B2A9", "#97C459"
    ROLE = {
        "baseline":  ("#E6F1FB", "#185FA5", "#0C447C"),
        "entropy":   ("#FAECE7", "#993C1D", "#712B13"),
        "reward":    ("#E1F5EE", "#0F6E56", "#085041"),
        "decision":  ("#EEEDFE", "#534AB7", "#3C3489"),
        "yes":       ("#EAF3DE", "#97C459", "#3B6D11"),
        "result":    ("#F7F6F2", "#D3D1C7", "#444441"),
        "outcome":   ("#FAEEDA", "#854F0B", "#633806"),
    }
    W, H = 6.6, 5.2
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # Draw a labelled box.
    def box(x0, x1, cy, h, text, role, size=7.5, ls="-", lw=1.0, r=0.06):
        fill, edge, color = ROLE[role]
        ax.add_patch(FancyBboxPatch((x0, cy - h / 2), x1 - x0, h,
                                    boxstyle=f"round,pad=0.0,rounding_size={r}",
                                    fc=fill, ec=edge, lw=lw, ls=ls, zorder=2))
        ax.text((x0 + x1) / 2, cy, text, ha="center", va="center", fontsize=size,
                color=color, linespacing=1.25, zorder=3)

    # Draw an arrow, solid if the path was taken.
    def arrow(x1, y1, x2, y2, taken=True):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", lw=1.5 if taken else 0.9,
                                    color=INK if taken else GREY,
                                    ls="-" if taken else (0, (3, 2)),
                                    shrinkA=0, shrinkB=0, mutation_scale=8))

    SX = 3.3
    AX0, AX1 = 2.05, 4.55
    LX0, LX1 = 0.05, 1.8
    NX0, NX1 = 4.7, 6.55

    box(AX0, AX1, 4.97, 0.36, r"A0  Baseline:  $\alpha=0,\ \beta=0$", "baseline", size=8)
    box(NX0, NX1, 4.97, 0.36, "$\\sigma$ collapses; agents already\ndifferentiate ($D^{\\mathrm{lm}}=1.22$ bits)",
        "result", size=6.8)
    arrow(SX, 4.79, SX, 4.67)

    stages = [
        (r"A1  Uniform entropy:  $\alpha=0.01$", "entropy",
         "Does entropy at L1 raise\ndiversity at L2 / L3?",
         "Entropy promotes\nbehavioural diversity",
         "Action entropy +22 nats;\n$\\Delta D^{\\mathrm{lm}}=+0.008$; same at $\\alpha=0.05$"),
        ("A1r  Same entropy, placed at contests", "entropy",
         "Was A1's null caused by\nwhere entropy was placed?",
         "Placement matters\nfor propagation",
         "Partly delivered ($\\tau=0.32$);\n$\\Delta D=+0.004$, below prediction"),
        (r"A2w  Diversity reward:  $\alpha=0,\ \beta=3$", "reward",
         "Does L2 diversity emerge\nif rewarded directly?",
         "L2 inducible; entropy\nfails to propagate",
         "$\\Delta D^{\\mathrm{lm}}=+0.007$;\ncompetence $-1.23$ ($t_3=-6.62$)"),
        ("A2cw  Critic sees the context", "reward",
         "Was A2w's null caused by the\ncritic lacking behavioural history?",
         None, None),
    ]
    ya = 4.5
    STEP = 1.0
    for k, (arm, role, q, yes, note) in enumerate(stages):
        yp = ya - 0.44
        box(AX0, AX1, ya, 0.34, arm, role, size=8)
        ax.plot([SX, SX], [ya - 0.17, yp + 0.2], color=INK, lw=1.5, zorder=1)
        box(AX0, AX1, yp, 0.4, q, "decision", size=7.2, r=0.12)
        if yes is not None:
            ax.plot([LX1, AX0], [yp, yp], color=GREY, lw=0.9, ls=(0, (3, 2)), zorder=1)
            ax.text((LX1 + AX0) / 2, yp + 0.06, "Yes", ha="center", va="bottom", fontsize=7, color=GREEN)
            box(LX0, LX1, yp, 0.4, yes, "yes", size=6.8, ls=(0, (3, 2)))
            box(NX0, NX1, yp, 0.4, note, "result", size=6.6)
            arrow(SX, yp - 0.2, SX, ya - STEP + 0.17)
            ax.text(SX + 0.07, (yp - 0.2 + ya - STEP + 0.17) / 2, "No", fontsize=7.5, color=INK,
                    fontweight="bold", va="center")
        ya -= STEP

    yp = 4.5 - 3 * STEP - 0.44
    yb = yp - 0.38
    ax.plot([SX, SX], [yp - 0.2, yb], color=INK, lw=1.5, zorder=1)
    ax.plot([0.9, SX], [yb, yb], color=GREY, lw=0.9, ls=(0, (3, 2)), zorder=1)
    top = 0.5
    for cx, lab in [(0.9, "Yes"), (2.25, "No")]:
        arrow(cx, yb, cx, top, taken=False)
        ax.text(cx + 0.06, yb + 0.07, lab, fontsize=7, va="bottom",
                color=GREEN if lab == "Yes" else GREY)
    box(0.05, 1.75, 0.27, 0.46, "Credit assignment\nwas the key issue", "yes", size=6.8, ls=(0, (3, 2)))
    box(1.85, 2.65, 0.27, 0.46, "No explanation\nsupported", "result", size=6.8, ls=(0, (3, 2)))
    arrow(SX, yb, SX, top)
    ax.text(SX + 0.07, yb + 0.07, "Inconclusive", fontsize=7.5, color=INK, fontweight="bold", va="bottom")
    box(2.8, 6.55, 0.27, 0.46,
        "$\\Delta D^{\\mathrm{lm}}=+0.038$ ($t_3=2.37$): positive in 4 of 4 seeds\n"
        "but below the floor; a $\\beta=0$ critic control is needed",
        "outcome", size=7, lw=1.3)

    save(fig, "fig4_0_experimental_logic_compact")


# Smoothed return and policy spread for each seed of one arm.
def load_curves(arm, every=100, sigma_bin=250):
    out = []
    for s in SEEDS:
        d = os.path.join(PROJECT_ROOT, "results", arm, f"seed_{s}")
        rm = np.load(os.path.join(d, "running_mean_rewards.npy"))
        sig = np.load(os.path.join(d, "policy_stds.npy"))
        steps = np.load(os.path.join(d, "update_steps.npy"))

        idx = np.unique(np.r_[np.arange(0, len(rm), every), len(rm) - 1])
        episodes = np.arange(1, len(rm) + 1)[idx]

        sig_episode = steps / STEPS_PER_EPISODE
        edges = np.arange(0, len(rm) + 1, sigma_bin)
        which = np.digitize(sig_episode, edges)
        sig_binned = np.array([sig[which == k].mean() if np.any(which == k) else np.nan
                               for k in range(1, len(edges))])
        out.append(dict(ep=episodes, rm=rm[idx], sig_ep=edges[1:] - sigma_bin / 2, sig=sig_binned))
    return out


# Figure 4.1: learning curves for A0 and A1.
def figure_4_1():
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for arm in ["baseline", "alpha_0.01"]:
        runs = load_curves(arm)
        ep = runs[0]["ep"] / 1000
        R = np.array([r["rm"] for r in runs])
        ax[0].fill_between(ep, R.min(0), R.max(0), color=COLOURS[arm], alpha=0.15, lw=0)
        ax[0].plot(ep, R.mean(0), color=COLOURS[arm], lw=1.6, label=LABELS[arm])

        se = runs[0]["sig_ep"] / 1000
        G = np.array([r["sig"] for r in runs])
        ax[1].fill_between(se, np.nanmin(G, 0), np.nanmax(G, 0), color=COLOURS[arm], alpha=0.15, lw=0)
        ax[1].plot(se, np.nanmean(G, 0), color=COLOURS[arm], lw=1.6, label=LABELS[arm])

    ax[0].set_xlabel("training episode (thousands)")
    ax[0].set_ylabel("training return (100-ep. mean)")
    ax[0].set_title("(a) return", fontsize=9, loc="left")
    ax[1].set_yscale("log")
    ax[1].axhline(SIGMA_FLOOR, color="#888780", lw=0.8, ls="--")
    ax[1].text(49.5, 0.0075, "floor 0.0067", ha="right", va="bottom", fontsize=8, color="#5F5E5A")
    ax[1].set_xlabel("training episode (thousands)")
    ax[1].set_ylabel(r"policy std. dev. $\sigma$")
    ax[1].set_title("(b) policy standard deviation", fontsize=9, loc="left")
    for a in ax:
        a.grid(True, axis="y")
        a.set_xlim(0, 50)
    ax[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig4_1_learning_curves")


# (episode, return, windowed D) for every cached baseline checkpoint.
def baseline_ladder(seed):
    cache = os.path.join(PROJECT_ROOT, "results", "exp3_windowed", "block4_cache")
    pts = []
    for f in glob.glob(os.path.join(cache, f"baseline_s{seed}_ep*.npz")):
        ep = int(re.search(r"_ep(\d+)\.npz$", f).group(1))
        z = np.load(f)
        D = np.mean([windowed_diversity(tok.T.astype(np.int64), 5)[0] for tok in z["tokens"]])
        pts.append((ep, float(z["returns"].mean()), float(D)))
    if not pts:
        raise SystemExit(f"No cached baseline rollouts for seed {seed} in {cache}. "
                         "Run exp3_windowed/block4_checks.py first.")
    return np.array(sorted(pts))


# Figure 4.2: baseline return against D for each seed.
def figure_4_2():
    fig, ax = plt.subplots(1, 4, figsize=(6.4, 2.2), sharey=True)
    colour = COLOURS["baseline"]
    for k, s in enumerate(SEEDS):
        P = baseline_ladder(s)
        R, D = P[:, 1], P[:, 2]
        peak = int(np.argmax(R))
        a = ax[k]
        a.plot(R, D, color=colour, lw=0.8, alpha=0.6)
        a.scatter(R[:peak + 1], D[:peak + 1], s=16, color=colour, zorder=3,
                  label="ascending (to peak)")
        a.scatter(R[peak + 1:], D[peak + 1:], s=16, facecolor="white", edgecolor=colour,
                  lw=1, zorder=3, label="descending (after peak)")
        a.set_title(f"seed {s}", fontsize=9, loc="left")
        a.grid(True, axis="y")
        a.set_xlabel(r"held-out return $\hat J$")
    ax[0].set_ylabel(r"$D^{\mathrm{lm}}$ (bits)")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig4_2_baseline_hysteresis")


# (episode, return, windowed D) for every cached A1 checkpoint.
def a1_ladder(seed):
    cache = os.path.join(PROJECT_ROOT, "results", "propagation_matrix", "cache")
    pts = []
    for f in glob.glob(os.path.join(cache, f"A1_s{seed}_ep*_det.npz")):
        ep = int(re.search(r"_ep(\d+)_det\.npz$", f).group(1))
        z = np.load(f)
        D = np.mean([windowed_diversity(tok.T.astype(np.int64), 5)[0] for tok in z["tokens"]])
        pts.append((ep, float(z["returns"].mean()), float(D)))
    if not pts:
        raise SystemExit(f"No cached A1 rollouts for seed {seed} in {cache}. "
                         "Run propagation_matrix.py first.")
    return np.array(sorted(pts))


# Figure 4.3: return against diversity for A0 and A1.
def figure_4_3():
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    first = True
    for s in SEEDS:
        P = baseline_ladder(s)
        R, D = P[:, 1], P[:, 2]
        peak = int(np.argmax(R))
        ax.scatter(R[:peak + 1], D[:peak + 1], s=14, color=COLOURS["baseline"], zorder=3,
                   label=LABELS["baseline"] + ", ascending" if first else None)
        ax.scatter(R[peak + 1:], D[peak + 1:], s=14, facecolor="white", edgecolor=COLOURS["baseline"],
                   lw=0.9, zorder=3, label=LABELS["baseline"] + ", descending" if first else None)
        Q = a1_ladder(s)
        ax.scatter(Q[:, 1], Q[:, 2], s=14, color=COLOURS["alpha_0.01"], zorder=3,
                   label=LABELS["alpha_0.01"] if first else None)
        first = False
    ax.set_xlabel(r"held-out return $\hat J$")
    ax.set_ylabel(r"$D^{\mathrm{lm}}$ (bits)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig4_3_return_diversity_A0_A1")


# (episode, return, windowed D) for every cached checkpoint of one arm.
def block4_ladder(label, seed):
    cache = os.path.join(PROJECT_ROOT, "results", "exp3_windowed", "block4_cache")
    pts = []
    for f in glob.glob(os.path.join(cache, f"{label}_s{seed}_ep*.npz")):
        ep = int(re.search(r"_ep(\d+)\.npz$", f).group(1))
        z = np.load(f)
        D = np.mean([windowed_diversity(tok.T.astype(np.int64), 5)[0] for tok in z["tokens"]])
        pts.append((ep, float(z["returns"].mean()), float(D)))
    if not pts:
        raise SystemExit(f"No cached rollouts for {label} seed {seed} in {cache}. "
                         "Run exp3_windowed/analyse_arms.py first.")
    return np.array(sorted(pts))


# Figure 4.4: A2w against the baseline.
def figure_4_4():
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.4), gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.7]))

    for arm in ["baseline", "beta_3_w5"]:
        runs = load_curves(arm)
        ep = runs[0]["ep"] / 1000
        R = np.array([r["rm"] for r in runs])
        ax[0].fill_between(ep, R.min(0), R.max(0), color=COLOURS[arm], alpha=0.15, lw=0)
        ax[0].plot(ep, R.mean(0), color=COLOURS[arm], lw=1.4, label=LABELS[arm])
    ax[0].set_xlabel("training episode (thousands)")
    ax[0].set_ylabel("training return (100-ep. mean)")
    ax[0].set_title("(a) task return", fontsize=9, loc="left")
    ax[0].set_xlim(0, 50)
    ax[0].grid(True, axis="y")
    ax[0].legend(frameon=False, loc="lower right", fontsize=8)

    for arm, label in [("baseline", "baseline"), ("beta_3_w5", "A2w")]:
        for k, s in enumerate(SEEDS):
            P = block4_ladder(label, s)
            R, D = P[:, 1], P[:, 2]
            peak = int(np.argmax(R))
            ax[1].scatter(R[:peak + 1], D[:peak + 1], s=10, color=COLOURS[arm], zorder=3)
            ax[1].scatter(R[peak + 1:], D[peak + 1:], s=10, facecolor="white",
                          edgecolor=COLOURS[arm], lw=0.8, zorder=3)
    ax[1].set_xlabel(r"held-out return $\hat J$")
    ax[1].set_ylabel(r"$D^{\mathrm{lm}}$ (bits)")
    ax[1].set_title("(b) checkpoints", fontsize=9, loc="left")
    ax[1].grid(True, axis="y")

    z = np.load(os.path.join(PROJECT_ROOT, "results", "exp3_windowed", "analysis_A2w_vs_baseline.npz"))
    d = np.asarray(z["primary_dD"], dtype=float)
    mean, sd = d.mean(), d.std(ddof=1)
    half = 3.182 * sd / np.sqrt(len(d))
    x = np.linspace(-0.12, 0.12, len(d))
    ax[2].axhspan(-DETECTION_FLOOR, DETECTION_FLOOR, color="#D3D1C7", alpha=0.35, lw=0)
    ax[2].axhline(0, color="#888780", lw=0.8)
    ax[2].scatter(x, d, s=16, color=COLOURS["beta_3_w5"], zorder=3)
    ax[2].errorbar([0.35], [mean], yerr=[[half], [half]], fmt="o", color="#2C2C2A", ms=4, capsize=3, lw=1)
    ax[2].set_xlim(-0.3, 0.55)
    ax[2].set_xticks([0, 0.35])
    ax[2].set_xticklabels(["seeds", "mean"])
    ax[2].set_ylabel(r"matched $\Delta D^{\mathrm{lm}}$ (bits)")
    ax[2].set_title("(c) A2w $-$ A0", fontsize=9, loc="left")
    ax[2].text(-0.28, DETECTION_FLOOR, "floor", ha="left", va="bottom", fontsize=7, color="#5F5E5A")

    fig.tight_layout()
    save(fig, "fig4_4_a2w")


C_A2CW, C_A2CWT = "#8f6bd0", "#c2477e"


# Mean of a per-update series in bins of training episodes.
def binned_by_episode(values, update_steps, n_episodes, width=500):
    ep = np.asarray(update_steps) / STEPS_PER_EPISODE
    edges = np.arange(0, n_episodes + 1, width)
    which = np.digitize(ep, edges)
    vals = np.asarray(values, float)
    out = np.array([vals[which == k].mean() if np.any(which == k) else np.nan
                    for k in range(1, len(edges))])
    return edges[1:] - width / 2, out


# Critic's estimate minus the actual return, binned over training.
def critic_error(arm, width=500):
    runs = []
    for s in SEEDS:
        d = os.path.join(PROJECT_ROOT, "results", arm, f"seed_{s}")
        q = np.load(os.path.join(d, "mean_q_values.npy"))
        steps = np.load(os.path.join(d, "update_steps.npy"))
        g = np.load(os.path.join(d, "episode_value_avg.npy"))
        n = len(g)
        x, qb = binned_by_episode(q, steps, n, width)
        gb = np.array([g[int(c - width / 2):int(c + width / 2)].mean() for c in x])
        runs.append(qb - gb)
    return x, np.array(runs)


# Figure 4.5: A2cw against A2w.
def figure_4_5():
    arms = [("beta_3_w5", "A2w", "A2w", COLOURS["beta_3_w5"]),
            ("beta_3_w5_augcritic_term", "A2cwT", "A2cw", C_A2CWT)]
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.4), gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.7]))

    for arm, _, lab, col in arms:
        runs = load_curves(arm)
        ep = runs[0]["ep"] / 1000
        R = np.array([r["rm"] for r in runs])
        ax[0].fill_between(ep, R.min(0), R.max(0), color=col, alpha=0.15, lw=0)
        ax[0].plot(ep, R.mean(0), color=col, lw=1.4, label=lab)
    ax[0].set_xlabel("training episode (thousands)")
    ax[0].set_ylabel("training return (100-ep. mean)")
    ax[0].set_title("(a) task return", fontsize=9, loc="left")
    ax[0].set_xlim(0, 50)
    ax[0].grid(True, axis="y")
    ax[0].legend(frameon=False, loc="lower right", fontsize=8)

    for s in SEEDS:
        P = block4_ladder("baseline", s)
        ax[1].scatter(P[:, 1], P[:, 2], s=7, color="#C9C7BD", zorder=2, lw=0,
                      label="A0, A1" if s == SEEDS[0] else None)
        Q = a1_ladder(s)
        ax[1].scatter(Q[:, 1], Q[:, 2], s=7, color="#C9C7BD", zorder=2, lw=0)
    for arm, label, lab, col in arms:
        for k, s in enumerate(SEEDS):
            P = block4_ladder(label, s)
            ax[1].scatter(P[:, 1], P[:, 2], s=10, color=col, zorder=3, lw=0,
                          label=lab if k == 0 else None)
    ax[1].set_xlabel(r"held-out return $\hat J$")
    ax[1].set_ylabel(r"$D^{\mathrm{lm}}$ (bits)")
    ax[1].set_title("(b) checkpoints", fontsize=9, loc="left")
    ax[1].grid(True, axis="y")
    ax[1].legend(frameon=False, loc="lower right", fontsize=7)

    z = np.load(os.path.join(PROJECT_ROOT, "results", "exp3_windowed", "analysis_A2cwT_vs_A2w.npz"))
    d = np.asarray(z["primary_dD"], dtype=float)
    mean, sd = d.mean(), d.std(ddof=1)
    half = 3.182 * sd / np.sqrt(len(d))
    x = np.linspace(-0.12, 0.12, len(d))
    ax[2].axhspan(-DETECTION_FLOOR, DETECTION_FLOOR, color="#D3D1C7", alpha=0.35, lw=0)
    ax[2].axhline(0, color="#888780", lw=0.8)
    ax[2].scatter(x, d, s=16, color=C_A2CWT, zorder=3)
    ax[2].errorbar([0.35], [mean], yerr=[[half], [half]], fmt="o", color="#2C2C2A", ms=4, capsize=3, lw=1)
    ax[2].set_xlim(-0.3, 0.55)
    ax[2].set_ylim(-0.06, 0.06)
    ax[2].set_xticks([0, 0.35])
    ax[2].set_xticklabels(["seeds", "mean"])
    ax[2].set_ylabel(r"matched $\Delta D^{\mathrm{lm}}$ (bits)")
    ax[2].set_title("(c) A2cw $-$ A2w", fontsize=9, loc="left")
    ax[2].text(-0.28, DETECTION_FLOOR, "floor", ha="left", va="bottom", fontsize=7, color="#5F5E5A")

    fig.tight_layout()
    save(fig, "fig4_5_a2cw")


# Figure 4.0: the experimental logic as a flow diagram.
def figure_4_0():
    from matplotlib.patches import FancyBboxPatch, Patch
    from matplotlib.lines import Line2D

    INK, MUTED, GREY = "#2C2C2A", "#5F5E5A", "#B4B2A9"
    ROLE = {
        "question":  ("#F1EFE8", "#888780", "#2C2C2A"),
        "baseline":  ("#E6F1FB", "#185FA5", "#0C447C"),
        "entropy":   ("#FAECE7", "#993C1D", "#712B13"),
        "reward":    ("#E1F5EE", "#0F6E56", "#085041"),
        "decision":  ("#EEEDFE", "#534AB7", "#3C3489"),
        "yes":       ("#EAF3DE", "#97C459", "#3B6D11"),
        "result":    ("#F7F6F2", "#D3D1C7", "#444441"),
        "outcome":   ("#FAEEDA", "#854F0B", "#633806"),
    }

    fig, ax = plt.subplots(figsize=(6.6, 9.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(-4.7, 19.1)
    ax.axis("off")

    # Draw a labelled box.
    def box(cx, cy, w, h, text, role, size=8.5, ls="-", lw=1.1, round_=0.15):
        fill, edge, color = ROLE[role]
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle=f"round,pad=0.02,rounding_size={round_}",
                                    fc=fill, ec=edge, lw=lw, ls=ls, zorder=2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=size, color=color,
                linespacing=1.35, zorder=3)

    # Draw an arrow, solid if the path was taken.
    def arrow(x1, y1, x2, y2, taken=True, color=None):
        c = color or (INK if taken else GREY)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", lw=1.7 if taken else 1.0, color=c,
                                    ls="-" if taken else (0, (3, 2)),
                                    shrinkA=0, shrinkB=0, mutation_scale=10))

    # Draw a line, solid if the path was taken.
    def line(xs, ys, taken=True):
        ax.plot(xs, ys, color=INK if taken else GREY, lw=1.7 if taken else 1.0,
                ls="-" if taken else (0, (3, 2)), solid_capstyle="butt", zorder=1)

    # Write a label next to a path.
    def label(x, y, text, taken):
        ax.text(x, y, text, fontsize=8, va="center", zorder=3,
                color=INK if taken else "#97C459", fontweight="bold" if taken else "normal")

    SPINE, LEFT = 6.3, 1.85
    AW, AH = 4.6, 1.05

    box(5.0, 18.35, 9.6, 1.05,
        "Research question\nIs diversity hierarchical? Can entropy drive diversity from one level to another?",
        "question", size=8.8)
    arrow(SPINE, 17.82, SPINE, 17.25)
    box(SPINE, 16.7, AW, AH, "A0  Baseline\n" + r"$\alpha=0,\ \beta=0$:  what happens naturally?", "baseline")
    box(LEFT + 0.05, 16.7, 3.6, 1.05,
        "$\\sigma$ collapses to its floor; agents already\ndifferentiate through their observations\n"
        "($D^{\\mathrm{lm}}=1.22$ bits)", "result", size=7.5)
    line([LEFT + 1.85, SPINE - AW / 2], [16.7, 16.7], taken=False)
    arrow(SPINE, 16.17, SPINE, 15.62)

    STEP = 4.8

    # Draw one stage: an arm, its question and the yes / no paths.
    def stage(y_box, arm_text, role, q_text, yes_text, no_text):
        box(SPINE, y_box, AW, AH, arm_text, role)
        line([SPINE, SPINE], [y_box - AH / 2, y_box - 0.9])
        box(SPINE, y_box - 1.3, 4.2, 0.8, q_text, "decision", size=8, round_=0.3)
        yb = y_box - 2.15
        line([SPINE, SPINE], [y_box - 1.7, yb])
        line([LEFT, SPINE], [yb, yb], taken=False)
        arrow(LEFT, yb, LEFT, yb - 0.75, taken=False)
        label(LEFT + 0.15, yb - 0.37, "Yes", taken=False)
        box(LEFT, yb - 1.3, 3.3, 1.05, yes_text, "yes", size=7.8, ls=(0, (3, 2)), lw=1.0)
        arrow(SPINE, yb, SPINE, y_box - STEP + AH / 2, taken=True)
        label(SPINE + 0.15, yb - 0.32, "No", taken=True)
        box(8.2, yb - 1.3, 3.4, 1.0, no_text, "result", size=7.3)
        return y_box - STEP

    y = 15.1
    y = stage(y, "A1  Uniform entropy\n" + r"$\alpha=0.01,\ \beta=0$", "entropy",
              "Does entropy at L1 (action)\nraise diversity at L2 / L3?",
              "Entropy promotes\nbehavioural diversity",
              "Action entropy +22 nats; matched\n$\\Delta D^{\\mathrm{lm}}=+0.008$ ($t_3=0.66$);\nsame at $\\alpha=0.05$")
    y = stage(y, "A1r  Role-conditioned entropy\nsame budget, placed where agents compete", "entropy",
              "Was A1's null caused by\nwhere entropy was placed?",
              "Placement matters\nfor propagation",
              "Partly delivered ($\\tau=0.32$);\n$\\Delta D=+0.004$, interval excludes\nthe predicted 0.03 to 0.06 bits")
    y = stage(y, "A2w  Windowed diversity reward\n" + r"$\alpha=0,\ \beta=3$", "reward",
              "Does L2 diversity emerge\nif it is rewarded directly?",
              "L2 is inducible; entropy\nsimply fails to propagate",
              "$\\Delta D^{\\mathrm{lm}}=+0.007$ ($t_3=0.43$);\ncompetence $-1.23$ ($t_3=-6.62$)")

    box(SPINE, y, AW, AH, "A2cw  Critic sees the windowed context\n" + r"$\alpha=0,\ \beta=3$", "reward")
    line([SPINE, SPINE], [y - AH / 2, y - 0.9])
    box(SPINE, y - 1.3, 4.6, 0.8, "Was A2w's null caused by the\ncritic lacking behavioural history?",
        "decision", size=8, round_=0.3)
    yb = y - 2.15
    line([SPINE, SPINE], [y - 1.7, yb])
    YES_X, NO_X, INC_X = 1.45, 4.05, 7.55
    line([YES_X, SPINE], [yb, yb], taken=False)
    for cx, lab in [(YES_X, "Yes"), (NO_X, "No")]:
        arrow(cx, yb, cx, yb - 0.75, taken=False)
        ax.text(cx + 0.15, yb - 0.37, lab, fontsize=8, va="center", color="#97C459" if lab == "Yes" else GREY)
    box(YES_X, yb - 1.3, 2.7, 1.05, "Credit assignment\nwas the key issue", "yes", size=7.8, ls=(0, (3, 2)), lw=1.0)
    box(NO_X, yb - 1.3, 2.2, 1.05, "No explanation\nsupported", "result", size=7.8, ls=(0, (3, 2)), lw=1.0)
    arrow(SPINE, yb, SPINE, yb - 0.75, taken=True)
    label(SPINE + 0.15, yb - 0.37, "Inconclusive", taken=True)
    box(INC_X, yb - 1.3, 4.65, 1.05,
        "$\\Delta D^{\\mathrm{lm}}=+0.038$ ($t_3=2.37$), positive in 4 of 4\n"
        "seeds but below the floor; a $\\beta=0$ critic\ncontrol is needed to decide", "outcome", size=7.6, lw=1.4)

    handles = [Patch(fc=ROLE[r][0], ec=ROLE[r][1], label=t) for r, t in [
        ("baseline", "baseline"), ("entropy", "policy-entropy arm"), ("reward", "diversity-reward arm"),
        ("decision", "question tested"), ("yes", "conclusion if yes"), ("outcome", "where the chain ends")]]
    handles += [Line2D([0], [0], color=INK, lw=1.7, label="path observed"),
                Line2D([0], [0], color=GREY, lw=1.0, ls=(0, (3, 2)), label="not taken")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
              frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.2)

    fig.tight_layout(pad=0.2)
    save(fig, "fig4_0_experimental_logic")


# Make all the figures.
def main():
    print("Chapter 4 figures")
    figure_4_0()
    figure_4_0_compact()
    figure_4_1()
    figure_4_2()
    figure_4_3()
    figure_4_4()
    figure_4_5()


if __name__ == "__main__":
    main()
