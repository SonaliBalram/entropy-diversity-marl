"""
exp3_outcome.py

THE OUTCOME MEASURE FOR EXPERIMENT 3, DEFINED BEFORE THE RUNS.

The problem this exists to solve
--------------------------------
D = I(X ; Z) is not independent of task performance on
Cooperative Navigation, because the task reward IS landmark
coverage. A policy whose agents split across three distinct
landmarks scores well AND has high D; one whose agents pile up
scores badly and has low D. Measured on the finished
Experiment 2 policies, per-layout D regresses on per-layout
return, WITHIN A FIXED POLICY, with a slope of 0.015 to 0.025
bits per reward unit (r = 0.44 to 0.65).

So a raw comparison of D between an Experiment 3 arm and its
control cannot separate

    "the bonus induced role differentiation"        (the claim)
    "the bonus changed the return, and D followed"  (an artefact)

and the artefact is large: the entire +0.163 bit rise of D
between the alpha = 0 and alpha = 0.01 policies is accounted
for by their return difference, leaving a residual of
+0.013 +/- 0.015 bits once return is matched.

Reporting raw D would therefore have produced a false positive
for Experiment 2. The measure has to be fixed before the
Experiment 3 runs, not after seeing them.

THE PRIMARY STATISTIC: MATCHED-RETURN Delta D
---------------------------------------------
Both arms are evaluated across their CHECKPOINT LADDER on the
same fixed held-out layouts, giving each arm a trajectory
through (return, D) space. Then, per seed:

  1. take the control arm's reporting checkpoint, best.pt,
     at held-out return r0 with diversity D0;
  2. find the treatment arm's checkpoint whose held-out return
     is CLOSEST to r0;
  3. require the match to be within --match-tol reward units;
  4. Delta D_matched = D_treatment(matched) - D0.

Paired across seeds, so the comparison inherits the same
pairing that carries the power at n = 4.

This needs no regression, no slope, no functional form, and no
extrapolation. It asks one question: at the same level of task
performance, do the two arms differ in role differentiation?

THE SECONDARY STATISTIC: TREND-LINE RESIDUAL
--------------------------------------------
As a cross-check that fails differently, D is also predicted
from return using the within-policy slope, and the residual
reported. Because that slope is itself policy-dependent, it is
computed three ways -- control arm's slope, pooled slope,
treatment arm's slope -- and the RANGE is reported. A residual
that changes sign across those three is not evidence of
anything, and saying so is the point of reporting all three.

Two measures that fail in different directions agreeing on the
same answer is the same argument already used for D against
H(phi): it closes the objection from both sides.

THE GAMING MONITOR
------------------
The pre-registered failure mode for Experiment 3 is that the
bonus is earned through action noise rather than behaviour. At
alpha = 0 the policy sigma sits at the LOG_STD_MIN floor of
0.0067, orders of magnitude below the distance scale that moves
an agent across a Voronoi boundary between landmarks, so
quantisation into tokens discards the sampling noise outright.
If sigma RISES along the treatment arm's ladder, that argument
has stopped holding. Sigma is therefore recorded at every
checkpoint and flagged automatically against --sigma-tripwire.
The tripwire is a tripwire, not a physical threshold: the
physical requirement is that sigma stays far below the
inter-landmark separation, which is of order 1.

Usage
-----
    # Experiment 3, once the runs exist
    python3 exp3_outcome.py \
        --control  "A0:checkpoints/baseline" \
        --treatment "A2:checkpoints/beta_3" \
        --seeds 42 128 456 789

    # dry run on data where the answer is already known
    python3 exp3_outcome.py \
        --control  "alpha0:checkpoints/baseline" \
        --treatment "alpha0.01:checkpoints/alpha_0.01" \
        --seeds 42 128 456 789
"""

import os
import re
import argparse

import numpy as np
import torch

from environment import MPEEnvironment
from maddpg_system import MADDPGSystem
from token_diversity import (
    tokenise, episode_diversity, permutation_null, max_diversity
)


# =========================================================
# ROLLOUT
# =========================================================
#
# Deterministic only. The deterministic pass is the
# sigma-invariant one: every agent acts at its mean mu, so no
# sampling-noise term enters any behaviour statistic and arms
# with different sigma stay comparable. Sigma itself is still
# recorded, because it is the gaming monitor.

def evaluate_checkpoint(path, layout_seeds, n_agents, max_cycles,
                        n_perm, torch_seed):

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    env = MPEEnvironment(n_agents=n_agents, max_cycles=max_cycles)
    maddpg = MADDPGSystem(
        n_agents=checkpoint["n_agents"],
        observation_dim=checkpoint["observation_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        gamma=checkpoint["gamma"],
        tau=checkpoint["tau"],
        twin_critics=checkpoint.get("twin_critics", True),
        # Rebuilt from the file so an augmented-critic
        # checkpoint (arm A2c, context_dim = 9) loads with the
        # right critic shape. Absent in every pre-Sept-2026
        # checkpoint, where 0 is the correct answer.
        context_dim=checkpoint.get("context_dim", 0),
        device="cpu",
    )
    alpha = float(checkpoint.get("alpha", 0.0))

    maddpg.load(path)
    maddpg.actor.eval()
    maddpg.critic.eval()

    torch.manual_seed(torch_seed)
    rng = np.random.default_rng(torch_seed)

    returns, d_bits, nulls, sigmas = [], [], [], []

    for layout_seed in layout_seeds:

        observations = env.reset(seed=layout_seed)
        tokens, total, sigma_acc = [], 0.0, []

        step = 0
        while True:
            # Sigma is state-conditioned but changes little within an
            # episode, and this extra forward pass is the dominant cost
            # of the sweep, so it is sampled every 5th step.
            if step % 5 == 0:
                _, std = maddpg.policy_params(observations)
                sigma_acc.append(float(np.mean(np.asarray(std))))
            step += 1

            actions = maddpg.act(observations, deterministic=True)
            observations, reward, terminated, truncated, _ = \
                env.step(actions)

            tokens.append(tokenise(observations))
            total += reward

            if terminated or truncated:
                break

        tokens = np.asarray(tokens)
        bits, _, _ = episode_diversity(tokens)

        returns.append(total)
        d_bits.append(bits)
        sigmas.append(float(np.mean(sigma_acc)))

        if n_perm > 0:
            m, _ = permutation_null(tokens, n_permutations=n_perm, rng=rng)
            nulls.append(m * max_diversity(n_agents, 3))

    env.close()

    return dict(
        alpha=alpha,
        ret=float(np.mean(returns)),
        D=float(np.mean(d_bits)),
        null=float(np.mean(nulls)) if nulls else float("nan"),
        sigma=float(np.mean(sigmas)),
        per_layout_return=np.asarray(returns),
        per_layout_D=np.asarray(d_bits),
    )


def ladder(root, seed):
    """Every ckpt_ep*.pt in a seed folder, in episode order."""
    d = os.path.join(root, f"seed_{seed}")
    items = []
    for name in os.listdir(d):
        m = re.fullmatch(r"ckpt_ep(\d+)\.pt", name)
        if m:
            items.append((int(m.group(1)), os.path.join(d, name)))
    return sorted(items)


def within_policy_slope(res):
    """
    Regression of per-layout D on per-layout return, INSIDE a
    single policy. The policy cannot vary along this line, so
    whatever slope it has cannot have been caused by the
    treatment -- which is exactly what makes it usable as the
    confound's own coefficient.
    """
    b = np.polyfit(res["per_layout_return"], res["per_layout_D"], 1)[0]
    r = np.corrcoef(res["per_layout_return"], res["per_layout_D"])[0, 1]
    return float(b), float(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, help="label:checkpoint_root")
    ap.add_argument("--treatment", required=True, help="label:checkpoint_root")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 128, 456, 789])
    ap.add_argument("--layouts", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=25)
    ap.add_argument("--match-tol", type=float, default=1.0,
                    help="Maximum |return difference| accepted as a match, "
                         "in reward units.")
    ap.add_argument("--sigma-tripwire", type=float, default=0.02,
                    help="Flag any treatment checkpoint whose mean sigma "
                         "exceeds this. The alpha=0 floor is 0.0067.")
    ap.add_argument("--reporting-checkpoint", default="best.pt")
    ap.add_argument("--out-dir", default=os.path.join("results", "exp3_outcome"))
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--max-cycles", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Identical to evaluate_baseline.py and token_eval.py, and
    # disjoint from every training block (training uses
    # SEED * 1_000_000 + episode).
    layout_seeds = [900_000_000 + i for i in range(args.layouts)]

    arms = {}
    for role, spec in [("control", args.control),
                       ("treatment", args.treatment)]:
        label, root = spec.split(":", 1)
        arms[role] = dict(label=label, root=root, seeds={})

    for role in ("control", "treatment"):
        arm = arms[role]
        print("=" * 72)
        print(f"{role.upper()}  {arm['label']}   ({arm['root']})")
        print("=" * 72)
        for seed in args.seeds:
            points = []
            for episode, path in ladder(arm["root"], seed):
                r = evaluate_checkpoint(
                    path, layout_seeds, args.n_agents, args.max_cycles,
                    args.n_perm, seed)
                r["episode"] = episode
                points.append(r)
                print(f"  seed {seed:>3}  ep {episode:>6}  "
                      f"return {r['ret']:8.3f}   D {r['D']:6.4f} bits   "
                      f"null {r['null']:6.4f}   sigma {r['sigma']:.4f}")

            rep_path = os.path.join(arm["root"], f"seed_{seed}",
                                    args.reporting_checkpoint)
            rep = evaluate_checkpoint(
                rep_path, layout_seeds, args.n_agents, args.max_cycles,
                args.n_perm, seed)
            print(f"  seed {seed:>3}  {args.reporting_checkpoint:>10}  "
                  f"return {rep['ret']:8.3f}   D {rep['D']:6.4f} bits   "
                  f"sigma {rep['sigma']:.4f}")
            arm["seeds"][seed] = dict(ladder=points, reporting=rep)
        print()

    # =====================================================
    # PRIMARY: matched-return Delta D
    # =====================================================
    print("=" * 72)
    print("PRIMARY OUTCOME -- matched-return Delta D")
    print("=" * 72)
    print(f"  Control reporting checkpoint: {args.reporting_checkpoint}. "
          f"Treatment matched to it on held-out return, "
          f"tolerance {args.match_tol} reward units.\n")

    dd, gaps, matched_eps, unmatched = [], [], [], []
    for seed in args.seeds:
        c = arms["control"]["seeds"][seed]["reporting"]
        pts = arms["treatment"]["seeds"][seed]["ladder"] + \
              [arms["treatment"]["seeds"][seed]["reporting"]]
        best = min(pts, key=lambda p: abs(p["ret"] - c["ret"]))
        gap = best["ret"] - c["ret"]
        ok = abs(gap) <= args.match_tol
        if not ok:
            unmatched.append(seed)
        dd.append(best["D"] - c["D"])
        gaps.append(gap)
        matched_eps.append(best.get("episode", "reporting"))
        print(f"  seed {seed:>3}: control {c['ret']:8.3f} / D {c['D']:.4f}   "
              f"vs treatment ep {str(best.get('episode','rep')):>7} "
              f"{best['ret']:8.3f} / D {best['D']:.4f}   "
              f"| return gap {gap:+.3f} {'OK' if ok else 'OUT OF TOLERANCE'}  "
              f"| dD {best['D'] - c['D']:+.4f}")

    dd = np.asarray(dd)
    t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else float("nan")
    print(f"\n  Delta D matched = {dd.mean():+.4f} +/- {dd.std(ddof=1):.4f} bits"
          f"   t({len(dd)-1}) = {t:+.2f}"
          f"   mean |return gap| = {np.mean(np.abs(gaps)):.3f}")
    if unmatched:
        print(f"  WARNING: seeds {unmatched} had no checkpoint within "
              f"{args.match_tol} of the control return; their closest match "
              f"is still INCLUDED in the mean above, because dropping a seed "
              f"after seeing its number is worse than an imperfect match. "
              f"A denser ladder (every 2000 episodes) tightens these matches. "
              f"Report the gap; do not widen the tolerance silently.")

    # =====================================================
    # SECONDARY: trend-line residual, three slopes
    # =====================================================
    print()
    print("=" * 72)
    print("SECONDARY -- residual after predicting D from return")
    print("=" * 72)
    sl = {}
    for role in ("control", "treatment"):
        b = [within_policy_slope(arms[role]["seeds"][s]["reporting"])[0]
             for s in args.seeds]
        r = [within_policy_slope(arms[role]["seeds"][s]["reporting"])[1]
             for s in args.seeds]
        sl[role] = float(np.mean(b))
        print(f"  within-policy slope, {arms[role]['label']:>12}: "
              f"{np.mean(b):+.5f} bits/reward unit   (r = {np.mean(r):+.3f})")
    sl["pooled"] = 0.5 * (sl["control"] + sl["treatment"])

    cD = np.array([arms["control"]["seeds"][s]["reporting"]["D"] for s in args.seeds])
    cR = np.array([arms["control"]["seeds"][s]["reporting"]["ret"] for s in args.seeds])
    tD = np.array([arms["treatment"]["seeds"][s]["reporting"]["D"] for s in args.seeds])
    tR = np.array([arms["treatment"]["seeds"][s]["reporting"]["ret"] for s in args.seeds])
    print(f"\n  raw Delta D at each arm's own best.pt: "
          f"{(tD - cD).mean():+.4f} bits "
          f"(with a return difference of {(tR - cR).mean():+.2f})")
    signs = []
    for name in ("control", "pooled", "treatment"):
        resid = (tD - cD) - sl[name] * (tR - cR)
        tt = resid.mean() / (resid.std(ddof=1) / np.sqrt(len(resid)))
        signs.append(np.sign(resid.mean()))
        print(f"    slope from {name:>9} ({sl[name]:+.5f}): residual "
              f"{resid.mean():+.4f} +/- {resid.std(ddof=1):.4f}  t = {tt:+5.2f}")
    if len(set(signs)) > 1:
        print("    NOTE: the residual changes sign across slope choices, so it "
              "is not evidence either way. The matched-return statistic above "
              "is the one to report.")

    # =====================================================
    # GAMING MONITOR
    # =====================================================
    print()
    print("=" * 72)
    print("GAMING MONITOR -- sigma along the treatment ladder")
    print("=" * 72)
    t_alpha = arms["treatment"]["seeds"][args.seeds[0]]["reporting"]["alpha"]
    if t_alpha > 0:
        print(f"  NOTE: this treatment arm trains at alpha = {t_alpha:g}, so a "
              f"large sigma is the INTENDED effect of policy-entropy "
              f"regularisation, not gaming. The tripwire below is only a "
              f"failure signal for an arm run at alpha = 0.\n")
    hits = []
    for seed in args.seeds:
        pts = arms["treatment"]["seeds"][seed]["ladder"]
        s = np.array([p["sigma"] for p in pts])
        print(f"  seed {seed:>3}: sigma {s.min():.4f} -> {s.max():.4f} "
              f"over episodes {pts[0]['episode']}..{pts[-1]['episode']}")
        if s.max() > args.sigma_tripwire:
            hits.append(seed)
    if hits and t_alpha > 0:
        print(f"\n  Tripwire exceeded on seeds {hits}, as expected at "
              f"alpha = {t_alpha:g}. Not a gaming signal for this arm.")
    elif hits:
        print(f"\n  TRIPWIRE HIT on seeds {hits}: sigma exceeded "
              f"{args.sigma_tripwire}. If this arm runs at alpha = 0, the "
              f"bonus may be being earned through action noise rather than "
              f"behaviour. Investigate before reporting.")
    else:
        print(f"\n  No checkpoint exceeded sigma = {args.sigma_tripwire}. "
              f"The quantisation argument holds.")

    # =====================================================
    # SAVE
    # =====================================================
    out = {}
    for role in ("control", "treatment"):
        lab = arms[role]["label"]
        for seed in args.seeds:
            pts = arms[role]["seeds"][seed]["ladder"]
            out[f"{lab}_seed{seed}_episode"] = np.array([p["episode"] for p in pts])
            for key in ("ret", "D", "null", "sigma"):
                out[f"{lab}_seed{seed}_{key}"] = np.array([p[key] for p in pts])
            rep = arms[role]["seeds"][seed]["reporting"]
            for key in ("ret", "D", "null", "sigma"):
                out[f"{lab}_seed{seed}_reporting_{key}"] = np.array(rep[key])
            out[f"{lab}_seed{seed}_reporting_per_layout_return"] = rep["per_layout_return"]
            out[f"{lab}_seed{seed}_reporting_per_layout_D"] = rep["per_layout_D"]
    out["delta_D_matched"] = dd
    out["match_return_gap"] = np.asarray(gaps)
    np.savez(os.path.join(args.out_dir, "outcome.npz"), **out)
    print(f"\nSaved -> {os.path.join(args.out_dir, 'outcome.npz')}")


if __name__ == "__main__":
    main()
