import argparse
import os
import re

import numpy as np
import torch

from environment import MPEEnvironment
from maddpg_system import MADDPGSystem
from instantaneous_metrics import InstantaneousBehaviourMetrics
from temporal_behaviour import TemporalBehaviourMetrics
from policy_divergence_behaviour import PolicyDivergenceMetrics


METRIC_KEYS = [
    "cosine_mean",
    "cosine_raw_mean",
    "euclidean_mean",
    "correlation_mean",
    "w2_mean",
    "js_mean",
    "mean_sigma",
]


def find_checkpoints(directory):
    out = []

    for name in os.listdir(directory):
        if not name.endswith(".pt"):
            continue

        m = re.fullmatch(r"ckpt_ep(\d+)\.pt", name)

        if m:
            out.append((f"ep{int(m.group(1))}", int(m.group(1)),
                        os.path.join(directory, name)))

    out.sort(key=lambda x: x[1])

    for extra in ("best.pt", "final.pt"):
        p = os.path.join(directory, extra)
        if os.path.exists(p):
            out.append((extra.replace(".pt", ""), 10**9, p))

    return out


def evaluate_one(ckpt_path, layout_seeds, deterministic, torch_seed):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    env = MPEEnvironment(
        n_agents=ckpt.get("n_agents", 3),
        max_cycles=25,
    )

    maddpg = MADDPGSystem(
        n_agents=ckpt.get("n_agents", 3),
        observation_dim=ckpt.get("observation_dim", 18),
        action_dim=ckpt.get("action_dim", 5),
        hidden_dim=ckpt.get("hidden_dim", 128),
        twin_critics=ckpt.get("twin_critics", True),
        context_dim=ckpt.get("context_dim", 0),
        device="cpu",
    )

    maddpg.actor.load_state_dict(ckpt["actor"])
    maddpg.actor.eval()

    torch.manual_seed(torch_seed)

    n_agents = maddpg.n_agents
    instant = InstantaneousBehaviourMetrics(n_agents)
    temporal = TemporalBehaviourMetrics(n_agents, maddpg.action_dim)
    divergence = PolicyDivergenceMetrics(n_agents, maddpg.action_dim)

    L = len(layout_seeds)
    rewards = np.zeros(L)
    metrics = {k: np.full(L, np.nan) for k in METRIC_KEYS}

    for li, seed in enumerate(layout_seeds):
        obs = env.reset(seed=seed)
        total = 0.0

        while True:
            actions = maddpg.act(obs, deterministic=deterministic)

            instant.record(actions)
            temporal.record(actions)
            m, s = maddpg.policy_params(obs)
            divergence.record(m, s)

            obs, reward, terminated, truncated, _ = env.step(actions)
            total += reward

            if terminated or truncated:
                break

        rec = {}
        rec.update(instant.get_episode_results())
        rec.update(temporal.get_episode_results())
        rec.update(divergence.get_episode_results())

        for k in METRIC_KEYS:
            if k in rec:
                metrics[k][li] = rec[k]

        rewards[li] = total
        instant.reset()
        temporal.reset()
        divergence.reset()

    env.close()

    return rewards, metrics


def fisher_z_mean(correlations):
    c = np.asarray([x for x in correlations if np.isfinite(x)])
    if c.size == 0:
        return np.nan, 0
    c = np.clip(c, -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(c)))), c.size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-root", type=str,
                        default=os.path.join("checkpoints", "baseline"))
    parser.add_argument("--layouts", type=int, default=200)
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of using the mean. "
                             "Default is deterministic, which removes "
                             "sampling as a nuisance variable -- and for "
                             "this policy makes no measurable difference "
                             "anyway (sigma is pinned at the floor).")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join("results", "checkpoint_sweep"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_dir = os.path.join(args.checkpoint_root, f"seed_{args.seed}")
    checkpoints = find_checkpoints(ckpt_dir)

    if len(checkpoints) < 3:
        raise SystemExit(
            f"Found only {len(checkpoints)} checkpoint(s) in {ckpt_dir}.\n"
            f"This analysis needs the ckpt_ep*.pt series, which is only "
            f"produced by the current train_baseline.py. Re-run the seed."
        )

    layout_seeds = [900_000_000 + i for i in range(args.layouts)]

    print("=" * 78)
    print(f"CHECKPOINT SERIES — seed {args.seed} — {len(checkpoints)} "
          f"checkpoints x {args.layouts} layouts")
    print(f"mode: {'stochastic' if args.stochastic else 'deterministic'}")
    print("=" * 78)
    print(f"{'checkpoint':>12} {'reward':>9} {'cos':>8} {'cos_raw':>9} "
          f"{'w2':>8} {'tcorr':>8} {'sigma':>8}")

    labels, all_rewards = [], []
    all_metrics = {k: [] for k in METRIC_KEYS}

    for label, _, path in checkpoints:
        rew, met = evaluate_one(
            path, layout_seeds,
            deterministic=not args.stochastic,
            torch_seed=args.seed,
        )

        labels.append(label)
        all_rewards.append(rew)
        for k in METRIC_KEYS:
            all_metrics[k].append(met[k])

        print(f"{label:>12} {rew.mean():9.3f} "
              f"{np.nanmean(met['cosine_mean']):8.3f} "
              f"{np.nanmean(met['cosine_raw_mean']):9.3f} "
              f"{np.nanmean(met['w2_mean']):8.3f} "
              f"{np.nanmean(met['correlation_mean']):8.3f} "
              f"{np.nanmean(met['mean_sigma']):8.4f}", flush=True)

    Rew = np.array(all_rewards)
    Met = {k: np.array(v) for k, v in all_metrics.items()}

    print("\n" + "=" * 78)
    print("ACROSS-CHECKPOINT correlation with mean reward")
    print("  (confounded with training stage -- see module docstring)")
    print("=" * 78)

    rm = Rew.mean(axis=1)
    for k in METRIC_KEYS:
        mm = np.nanmean(Met[k], axis=1)
        ok = np.isfinite(mm) & np.isfinite(rm)
        if ok.sum() < 3 or np.std(mm[ok]) < 1e-12:
            print(f"  {k:20s}: {'n/a (no variation)':>20}")
            continue
        print(f"  {k:20s}: {np.corrcoef(rm[ok], mm[ok])[0,1]:+8.3f}   "
              f"(n = {ok.sum()} checkpoints)")

    print("\n" + "=" * 78)
    print("WITHIN-LAYOUT, ACROSS-CHECKPOINT correlation with reward")
    print("  layout held exactly constant; Fisher-z averaged over layouts")
    print("  OBSERVATIONAL, not causal: training stage is a common cause")
    print("=" * 78)

    for k in METRIC_KEYS:
        per_layout = []
        for li in range(Rew.shape[1]):
            r, m = Rew[:, li], Met[k][:, li]
            ok = np.isfinite(r) & np.isfinite(m)
            if ok.sum() < 4 or np.std(m[ok]) < 1e-12 or np.std(r[ok]) < 1e-12:
                continue
            per_layout.append(np.corrcoef(r[ok], m[ok])[0, 1])
        c, nl = fisher_z_mean(per_layout)
        print(f"  {k:20s}: {c:+8.3f}   (over {nl} layouts)")

    np.savez(
        os.path.join(args.out_dir, f"seed_{args.seed}.npz"),
        labels=np.array(labels),
        rewards=Rew,
        **{f"metric_{k}": v for k, v in Met.items()},
    )
    print(f"\nSaved -> {os.path.join(args.out_dir, f'seed_{args.seed}.npz')}")


if __name__ == "__main__":
    main()
