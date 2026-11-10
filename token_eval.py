# Evaluate token diversity D of trained policies on held-out layouts.

import os
import argparse
import numpy as np
import torch

from environment import MPEEnvironment
from maddpg_system import MADDPGSystem
from token_diversity import (
    tokenise, episode_diversity, permutation_null, max_diversity
)


# Run the policy on every layout; collect D, its chance level and rewards.
def run(env, maddpg, layout_seeds, deterministic, repeats, torch_seed,
        n_perm):
    torch.manual_seed(torch_seed)
    rng = np.random.default_rng(torch_seed)

    D, Dn, null, rewards, counts = [], [], [], [], []

    for layout_seed in layout_seeds:
        for _ in range(repeats):
            observations = env.reset(seed=layout_seed)
            tokens, total = [], 0.0

            while True:
                actions = maddpg.act(
                    observations, deterministic=deterministic
                )
                observations, reward, term, trunc, _ = env.step(actions)

                tokens.append(tokenise(observations))
                total += reward

                if term or trunc:
                    break

            tokens = np.asarray(tokens)
            d_bits, d_norm, f = episode_diversity(tokens)

            m, _ = permutation_null(
                tokens, n_permutations=n_perm, rng=rng
            )

            D.append(d_bits)
            Dn.append(d_norm)
            null.append(m)
            rewards.append(total)
            counts.append((f * tokens.shape[0]).round().astype(np.int64))

    return (np.array(D), np.array(Dn), np.array(null),
            np.array(rewards), np.array(counts))


# Evaluate each condition and print the results.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", required=True,
                    help="label:checkpoint_root:checkpoint_name")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 128, 456, 789])
    ap.add_argument("--layouts", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=10,
                    help="stochastic rollouts per layout")
    ap.add_argument("--n-perm", type=int, default=60)
    ap.add_argument("--out-dir", default=os.path.join("results", "token_eval"))
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--max-cycles", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    layout_seeds = [900_000_000 + i for i in range(args.layouts)]

    print(f"D bound = log2(min(N,|A|)) = "
          f"{max_diversity(args.n_agents, 3):.4f} bits\n")

    summary = {}

    for spec in args.conditions:
        label, root, name = spec.split(":")
        print("=" * 72)
        print(f"{label}   ({root}/seed_*/{name})")
        print("=" * 72)

        per_seed = {"det": [], "det_null": [], "sto": [], "sto_null": []}

        for seed in args.seeds:
            ckpt = os.path.join(root, f"seed_{seed}", name)
            c = torch.load(ckpt, map_location="cpu", weights_only=False)

            env = MPEEnvironment(n_agents=args.n_agents,
                                 max_cycles=args.max_cycles)
            maddpg = MADDPGSystem(
                n_agents=c["n_agents"],
                observation_dim=c["observation_dim"],
                action_dim=c["action_dim"],
                hidden_dim=c["hidden_dim"],
                gamma=c["gamma"], tau=c["tau"],
                twin_critics=c.get("twin_critics", True),
                context_dim=c.get("context_dim", 0),
                device="cpu",
            )
            maddpg.load(ckpt)
            maddpg.actor.eval()
            maddpg.critic.eval()

            dD, dDn, dnull, dr, dcounts = run(
                env, maddpg, layout_seeds, True, 1, seed, args.n_perm)
            sD, sDn, snull, sr, _ = run(
                env, maddpg, layout_seeds, False, args.repeats, seed,
                args.n_perm)
            env.close()

            print(f"  seed {seed:>3}  "
                  f"det D {dD.mean():6.4f} bits ({dDn.mean():5.3f} norm, "
                  f"null {dnull.mean():5.3f})   "
                  f"stoch D {sD.mean():6.4f} ({sDn.mean():5.3f} norm, "
                  f"null {snull.mean():5.3f})   return {dr.mean():8.3f}")

            per_seed["det"].append(dD.mean())
            per_seed["det_null"].append(dnull.mean() *
                                        max_diversity(args.n_agents, 3))
            per_seed["sto"].append(sD.mean())
            per_seed["sto_null"].append(snull.mean() *
                                        max_diversity(args.n_agents, 3))

            np.savez(
                os.path.join(args.out_dir, f"{label}_seed_{seed}.npz"),
                det_D=dD, det_D_norm=dDn, det_null=dnull,
                det_reward=dr, det_counts=dcounts,
                sto_D=sD, sto_D_norm=sDn, sto_null=snull, sto_reward=sr,
            )

        summary[label] = {k: np.array(v) for k, v in per_seed.items()}
        d = summary[label]["det"]
        n = summary[label]["det_null"]
        print(f"  four-seed mean  det D = {d.mean():.4f} +/- {d.std(ddof=1):.4f} bits"
              f"   null = {n.mean():.4f}"
              f"   excess = {(d - n).mean():+.4f}\n")

    labels = [s.split(":")[0] for s in args.conditions]
    base = labels[0]
    print("=" * 72)
    print(f"PAIRED ACROSS SEEDS, vs {base}   (n = {len(args.seeds)}, df = 3)")
    print("=" * 72)
    for lab in labels[1:]:
        for key in ["det", "sto"]:
            diff = summary[lab][key] - summary[base][key]
            t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
            print(f"  {lab:>22} {key:>4}: dD = {diff.mean():+.4f} bits "
                  f"+/- {diff.std(ddof=1):.4f}   t(3) = {t:+.2f}")

    np.savez(os.path.join(args.out_dir, "summary.npz"),
             **{f"{l}_{k}": v for l, d in summary.items()
                for k, v in d.items()})


if __name__ == "__main__":
    main()
