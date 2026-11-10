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
    d = os.path.join(root, f"seed_{seed}")
    items = []
    for name in os.listdir(d):
        m = re.fullmatch(r"ckpt_ep(\d+)\.pt", name)
        if m:
            items.append((int(m.group(1)), os.path.join(d, name)))
    return sorted(items)


def within_policy_slope(res):
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
