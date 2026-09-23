"""
Hyperparameter_sweep.py

Short pilot search over OPTIMISATION-level hyperparameters
only (batch size, learning rates, tau, clipping, Huber delta).
alpha stays 0 and the reward function is untouched, so the
baseline definition is preserved.

CORRECTIONS RELATIVE TO THE PREVIOUS VERSION
--------------------------------------------
1. env.step() 5-tuple; only `terminated` is stored.

2. episode_seed = seed * 1_000_000 + episode, so pilot seeds
   do not share layouts with each other or with the full runs.

3. UPDATE_EVERY exposed. The previous pilot was not cheap:
   3000 episodes x 25 steps x 5 configs x 2 seeds = 750,000
   updates, which at batch 1024 (39.8 ms measured) is ~8 hours
   -- more than half a full training run, to rank five
   candidates. The defaults here (1500 episodes,
   update_every=4) bring that to well under an hour.

4. Scoring now uses the MEAN OF PER-SEED STANDARD DEVIATIONS
   rather than the std of the pooled tail. Pooling first
   conflates within-seed noise (what is being optimised) with
   between-seed offset (what is not).

5. Configs 1-4 kept, config 0 relabelled honestly: with
   grad_clip_norm=None and the old lr pair it is the
   pre-optimisation control, and it is the ONLY fair
   comparison point for claiming the tuning helped.
"""

import argparse
import csv
import random

import numpy as np
import torch

from environment import MPEEnvironment
from maddpg_system import MADDPGSystem
from replay_buffer import ReplayBuffer


CANDIDATE_CONFIGS = [
    # pre-optimisation control
    {"batch_size": 128,  "lr_actor": 1e-3, "lr_critic": 1e-3,
     "tau": 0.01,  "grad_clip_norm": None, "huber_delta": 1.0},

    {"batch_size": 256,  "lr_actor": 1e-4, "lr_critic": 1e-3,
     "tau": 0.005, "grad_clip_norm": 10.0, "huber_delta": 10.0},

    {"batch_size": 1024, "lr_actor": 1e-4, "lr_critic": 1e-3,
     "tau": 0.005, "grad_clip_norm": 10.0, "huber_delta": 10.0},

    {"batch_size": 1024, "lr_actor": 3e-4, "lr_critic": 1e-3,
     "tau": 0.01,  "grad_clip_norm": 10.0, "huber_delta": 10.0},

    {"batch_size": 512,  "lr_actor": 1e-4, "lr_critic": 5e-4,
     "tau": 0.005, "grad_clip_norm": 5.0,  "huber_delta": 10.0},
]


def train_short(config, seed, num_episodes, update_every,
                n_agents=3, observation_dim=18, action_dim=5,
                max_cycles=25):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = MPEEnvironment(n_agents=n_agents, max_cycles=max_cycles)

    maddpg = MADDPGSystem(
        n_agents=n_agents,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=128,
        gamma=0.95,
        tau=config["tau"],
        lr_actor=config["lr_actor"],
        lr_critic=config["lr_critic"],
        grad_clip_norm=config["grad_clip_norm"],
        huber_delta=config["huber_delta"],
        device="cpu",
    )

    replay_buffer = ReplayBuffer(
        capacity=100000,
        n_agents=n_agents,
        observation_dim=observation_dim,
        action_dim=action_dim,
        seed=seed,
    )

    batch_size = config["batch_size"]
    warmup_steps = max(5000, 5 * batch_size)

    action_rng = np.random.default_rng(seed)

    episode_rewards = []
    global_step = 0

    for episode in range(num_episodes):

        observations = env.reset(seed=seed * 1_000_000 + episode)
        episode_reward = 0.0

        while True:

            global_step += 1

            if global_step < warmup_steps:
                actions = action_rng.random(
                    (n_agents, action_dim)
                ).astype(np.float32)
            else:
                actions = maddpg.act(observations, deterministic=False)

            (
                next_observations,
                reward,
                terminated,
                truncated,
                _
            ) = env.step(actions)

            replay_buffer.add(
                observations, actions, reward,
                next_observations, terminated
            )

            observations = next_observations
            episode_reward += reward

            if (
                len(replay_buffer) >= batch_size
                and global_step >= warmup_steps
                and global_step % update_every == 0
            ):
                maddpg.update(replay_buffer.sample(batch_size))

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)

    env.close()

    return np.array(episode_rewards)


def score_config(config, seeds, episodes, tail_episodes,
                 lam, update_every):

    means = []
    stds = []

    for seed in seeds:

        rewards = train_short(config, seed, episodes, update_every)
        tail = rewards[-tail_episodes:]

        means.append(tail.mean())
        stds.append(tail.std())

        print(f"    seed {seed:5d} | tail mean {tail.mean():9.3f} "
              f"| tail std {tail.std():7.3f}")

    mean_r = float(np.mean(means))

    # Mean of per-seed stds -- within-seed noise only.
    std_r = float(np.mean(stds))

    # Reported separately so the two are never confused.
    between_seed = float(np.std(means))

    return {
        "mean": mean_r,
        "within_seed_std": std_r,
        "between_seed_std": between_seed,
        "score": mean_r - lam * std_r,
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--tail-episodes", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 128])
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--out-csv", type=str,
                        default="hyperparameter_sweep_results.csv")
    args = parser.parse_args()

    results = []

    for i, config in enumerate(CANDIDATE_CONFIGS):
        print("=" * 72)
        print(f"Config {i}: {config}")
        stats = score_config(
            config, args.seeds, args.episodes,
            args.tail_episodes, args.lam, args.update_every
        )
        print(f"  -> mean {stats['mean']:.3f} | "
              f"within-seed std {stats['within_seed_std']:.3f} | "
              f"between-seed std {stats['between_seed_std']:.3f} | "
              f"score {stats['score']:.3f}")
        results.append({**config, **stats, "config_index": i})

    results.sort(key=lambda r: r["score"], reverse=True)

    print("\n" + "=" * 72)
    print("RANKED (best first)")
    print("=" * 72)
    for r in results:
        print(f"config {r['config_index']} | score {r['score']:8.3f} | "
              f"mean {r['mean']:8.3f} | within-std {r['within_seed_std']:6.3f} | "
              f"bs={r['batch_size']} lra={r['lr_actor']} "
              f"lrc={r['lr_critic']} tau={r['tau']} "
              f"clip={r['grad_clip_norm']} delta={r['huber_delta']}")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to {args.out_csv}")
    print("A 1500-episode pilot on 2 seeds ranks candidates; it does NOT")
    print("establish a result. Re-run the winner at full length on all 5 seeds.")


if __name__ == "__main__":
    main()
