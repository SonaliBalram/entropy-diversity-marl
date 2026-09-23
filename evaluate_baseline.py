"""
evaluate_baseline.py

Splits within-seed reward variance into its two sources:
environment layout, and the policy's own action sampling.

CORRECTIONS RELATIVE TO THE PREVIOUS VERSION
--------------------------------------------
1. PAIRED DESIGN. The previous version gave the stochastic
   and deterministic passes DIFFERENT environment seeds
   (offsets 10M and 20M). The two reward distributions then
   differed in both the layouts AND the sampling, so their
   std ratio was not attributable to sampling alone. Both
   passes now use the SAME layout seeds, which makes the
   comparison paired and the difference interpretable.

2. PROPER VARIANCE DECOMPOSITION. Running K stochastic
   rollouts on each of M fixed layouts lets the two sources
   be separated directly rather than inferred from a ratio:

       total variance = between-layout + within-layout
       between-layout = environment / initial conditions
       within-layout  = action sampling

   This is a one-way ANOVA decomposition and is what the
   write-up should quote.

3. torch.manual_seed is now set, so the stochastic pass is
   reproducible. It previously was not -- env.reset(seed=)
   seeds the environment, not the policy's sampling.

4. env.step() 5-tuple.

5. Behavioural metrics are recorded DURING evaluation. The
   diversity numbers that answer the research question should
   come from the evaluated policy, not from the exploratory
   actions taken during training.
"""

import os
import argparse

import numpy as np
import torch

from environment import MPEEnvironment
from maddpg_system import MADDPGSystem
from instantaneous_metrics import InstantaneousBehaviourMetrics
from temporal_behaviour import TemporalBehaviourMetrics
from policy_divergence_behaviour import PolicyDivergenceMetrics


def run_episodes(env, maddpg, layout_seeds, deterministic,
                 repeats=1, torch_seed=0, collect_behaviour=True):
    """
    Returns
        rewards  : [n_layouts, repeats]
        metrics  : dict of [n_layouts, repeats] arrays

    The [layout, repeat] SHAPE is the point. Holding the
    layout fixed and varying only the policy's sampling lets
    the within-layout question be asked directly: on the SAME
    starting configuration, do the rollouts where the agents
    homogenise score worse? Layout difficulty is constant down
    each row, so it cannot confound the answer -- which is
    exactly the flaw in correlating these quantities across
    training episodes, where every episode has a different
    layout.
    """

    torch.manual_seed(torch_seed)

    n_agents = maddpg.n_agents
    action_dim = maddpg.action_dim

    instant = InstantaneousBehaviourMetrics(n_agents)
    temporal = TemporalBehaviourMetrics(n_agents, action_dim)
    divergence = PolicyDivergenceMetrics(n_agents, action_dim)

    L, K = len(layout_seeds), repeats
    rewards = np.zeros((L, K))
    metrics = {}

    for li, layout_seed in enumerate(layout_seeds):

        for rep in range(K):

            observations = env.reset(seed=layout_seed)
            total = 0.0

            while True:

                actions = maddpg.act(
                    observations,
                    deterministic=deterministic
                )

                if collect_behaviour:
                    instant.record(actions)
                    temporal.record(actions)
                    m, s = maddpg.policy_params(observations)
                    divergence.record(m, s)

                (
                    observations,
                    reward,
                    terminated,
                    truncated,
                    _
                ) = env.step(actions)

                total += reward

                if terminated or truncated:
                    break

            rewards[li, rep] = total

            if collect_behaviour:
                record = {}
                record.update(instant.get_episode_results())
                record.update(temporal.get_episode_results())
                record.update(divergence.get_episode_results())

                for k, v in record.items():
                    if k not in metrics:
                        metrics[k] = np.full((L, K), np.nan)
                    metrics[k][li, rep] = v

                instant.reset()
                temporal.reset()
                divergence.reset()

    return rewards, metrics


def within_layout_correlation(rewards, metric):
    """
    Correlation between a behaviour metric and reward,
    computed WITHIN each layout and then averaged across
    layouts (Fisher-z averaged, so correlations combine
    properly rather than being averaged raw).

    This is the confound-free version of the training-time
    correlation. Layout difficulty is held constant, so a
    non-zero value here means the behaviour itself is
    associated with the outcome.
    """

    zs = []

    for li in range(rewards.shape[0]):

        r = rewards[li]
        m = metric[li]
        ok = np.isfinite(r) & np.isfinite(m)

        if ok.sum() < 4:
            continue
        if np.std(r[ok]) < 1e-9 or np.std(m[ok]) < 1e-9:
            continue

        c = np.corrcoef(r[ok], m[ok])[0, 1]
        c = np.clip(c, -0.999999, 0.999999)
        zs.append(np.arctanh(c))

    if not zs:
        return np.nan, 0

    return float(np.tanh(np.mean(zs))), len(zs)


def summarize(rewards, label):
    flat = rewards.ravel()
    print(
        f"    {label:>14s} | mean {flat.mean():9.3f} | "
        f"std {flat.std():7.3f} | min {flat.min():9.3f} | "
        f"max {flat.max():9.3f}"
    )


def decompose(rewards):
    """
    One-way variance decomposition over layouts.

    rewards : [n_layouts, n_repeats]
    """
    layout_means = rewards.mean(axis=1)
    between = layout_means.var()
    within = rewards.var(axis=1).mean()
    return between, within


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 128, 456, 789, 1000])
    parser.add_argument("--checkpoint-root", type=str,
                        default=os.path.join("checkpoints", "baseline"))
    parser.add_argument("--layouts", type=int, default=200,
                        help="Distinct evaluation layouts.")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Stochastic rollouts per layout, for the "
                             "within-layout variance term.")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join("results", "baseline_eval"))
    parser.add_argument("--checkpoint-name", type=str, default="final.pt",
                        help="Which checkpoint inside each seed directory "
                             "to evaluate, e.g. best.pt or ckpt_ep14000.pt")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Evaluation layouts are disjoint from every training
    # block (training uses SEED * 1_000_000 + episode).
    layout_seeds = [900_000_000 + i for i in range(args.layouts)]

    pooled_stoch = []
    pooled_det = []
    pooled_behaviour = {}
    pooled_det_behaviour = {}

    for seed in args.seeds:

        ckpt_path = os.path.join(
            args.checkpoint_root, f"seed_{seed}", args.checkpoint_name
        )

        print("-" * 72)
        print(f"Seed {seed}: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            avail = sorted(os.listdir(os.path.dirname(ckpt_path))) \
                if os.path.isdir(os.path.dirname(ckpt_path)) else []
            raise SystemExit(
                f"No checkpoint at {ckpt_path}.\n"
                f"Available in that directory: {avail}\n"
                f"(best.pt only appears once training passes the "
                f"BEST_BURN_IN episode threshold set in train_baseline.py.)"
            )

        checkpoint = torch.load(ckpt_path, map_location="cpu")

        env = MPEEnvironment(
            n_agents=args.n_agents,
            max_cycles=args.max_cycles
        )

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

        # maddpg.load() handles both the single-critic and
        # twin-critic checkpoint formats.
        maddpg.load(ckpt_path)

        maddpg.actor.eval()
        maddpg.critic.eval()

        # --- stochastic: many repeats per layout ---
        stoch, stoch_behaviour = run_episodes(
            env, maddpg, layout_seeds,
            deterministic=False,
            repeats=args.repeats,
            torch_seed=seed,
        )

        # --- deterministic: identical layouts, 1 repeat
        #     (repeats would be identical by construction) ---
        det, det_behaviour = run_episodes(
            env, maddpg, layout_seeds,
            deterministic=True,
            repeats=1,
            torch_seed=seed,
        )

        env.close()

        summarize(stoch, "stochastic")
        summarize(det, "deterministic")

        between, within = decompose(stoch)
        total = between + within

        print(f"      variance decomposition (stochastic):")
        print(f"        between-layout (environment)  : "
              f"{between:8.3f}   ({100*between/total:5.1f}%)")
        print(f"        within-layout  (action sample): "
              f"{within:8.3f}   ({100*within/total:5.1f}%)")

        print(f"      deterministic-policy behaviour:")
        for k in ["cosine_mean", "cosine_raw_mean", "euclidean_mean",
                  "correlation_mean", "js_mean", "w2_mean", "mean_sigma"]:
            if k in det_behaviour:
                print(f"        {k:22s}: "
                      f"{np.nanmean(det_behaviour[k]):8.4f}")

        # ---------------------------------------------
        # WITHIN-LAYOUT TEST
        # ---------------------------------------------
        print(f"      within-layout correlation with reward "
              f"({args.repeats} rollouts x {args.layouts} layouts):")
        for k in ["cosine_mean", "cosine_raw_mean",
                  "correlation_mean", "w2_mean", "mean_sigma"]:
            if k in stoch_behaviour:
                c, nl = within_layout_correlation(
                    stoch, stoch_behaviour[k]
                )
                print(f"        {k:22s}: {c:+8.3f}  "
                      f"(over {nl} layouts)")

        np.savez(
            os.path.join(args.out_dir, f"seed_{seed}.npz"),
            stochastic=stoch,
            deterministic=det,
            **{f"stoch_{k}": v for k, v in stoch_behaviour.items()},
            # The deterministic pass is the sigma-INVARIANT one: every
            # agent acts at its mean mu, so cosine/Euclidean on the
            # physical force carry no sampling-noise term and are
            # comparable across arms with different policy sigma.
            # These are the numbers the diversity question rests on.
            **{f"det_{k}": v for k, v in det_behaviour.items()},
        )

        pooled_stoch.append(stoch)
        pooled_det.append(det)

        for k, v in stoch_behaviour.items():
            pooled_behaviour.setdefault(k, []).extend(
                v.ravel().tolist()
            )

        for k, v in det_behaviour.items():
            pooled_det_behaviour.setdefault(k, []).extend(
                v.ravel().tolist()
            )

    # =====================================================
    # POOLED
    # =====================================================

    all_stoch = np.concatenate([s.ravel() for s in pooled_stoch])
    all_det = np.concatenate([d.ravel() for d in pooled_det])

    print("=" * 72)
    print(f"POOLED: {len(args.seeds)} seeds x {args.layouts} layouts "
          f"x {args.repeats} repeats")
    print("=" * 72)
    summarize(all_stoch, "stochastic")
    summarize(all_det, "deterministic")

    between = np.mean([decompose(s)[0] for s in pooled_stoch])
    within = np.mean([decompose(s)[1] for s in pooled_stoch])
    total = between + within

    print(f"\n  Environment-driven variance : {100*between/total:5.1f}%")
    print(f"  Sampling-driven variance    : {100*within/total:5.1f}%")
    print("\n  If sampling dominates, the reported episode-level 'noise'")
    print("  is the policy's intentional stochasticity, not instability.")
    print("  If environment dominates, it is layout difficulty, which is")
    print("  also expected and not a defect. Either way the correct")
    print("  headline number is the per-layout MEAN, not the raw spread.")

    print("\n  Pooled behavioural metrics (stochastic evaluation):")
    for k in ["cosine_mean", "cosine_raw_mean", "euclidean_mean",
              "correlation_mean", "js_mean", "w2_mean", "mean_sigma"]:
        if k in pooled_behaviour:
            print(f"    {k:22s}: "
                  f"{np.nanmean(pooled_behaviour[k]):8.4f}")

    print("\n  Pooled behavioural metrics (DETERMINISTIC evaluation)"
          " -- sigma-invariant, quote these:")
    for k in ["cosine_mean", "cosine_raw_mean", "euclidean_mean",
              "correlation_mean", "js_mean", "w2_mean", "mean_sigma"]:
        if k in pooled_det_behaviour:
            print(f"    {k:22s}: "
                  f"{np.nanmean(pooled_det_behaviour[k]):8.4f}")

    np.savez(
        os.path.join(args.out_dir, "pooled.npz"),
        stochastic=all_stoch,
        deterministic=all_det,
        **{k: np.array(v) for k, v in pooled_behaviour.items()},
        **{f"det_{k}": np.array(v) for k, v in pooled_det_behaviour.items()},
    )


if __name__ == "__main__":
    main()
