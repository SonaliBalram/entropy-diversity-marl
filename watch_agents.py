# Watch trained agents play in an on-screen window.

import argparse
import time

import numpy as np
import torch
from mpe2 import simple_spread_v3

from environment import MPEEnvironment
from maddpg_system import MADDPGSystem
from instantaneous_metrics import InstantaneousBehaviourMetrics


# The training environment with MPE's window turned on.
class RenderableEnvironment(MPEEnvironment):
    # Create the MPE environment with rendering.
    def __init__(self, n_agents=3, max_cycles=25, render_mode="human"):
        self.n_agents = n_agents
        self.max_cycles = max_cycles

        self.env = simple_spread_v3.parallel_env(
            N=n_agents,
            max_cycles=max_cycles,
            continuous_actions=True,
            render_mode=render_mode,
        )

        self.agents = self.env.possible_agents

        self.observation_dim = self.env.observation_space(
            self.agents[0]
        ).shape[0]

        self.action_dim = self.env.action_space(
            self.agents[0]
        ).shape[0]

        self.collision_penalty = 1.0
        self.time_penalty = 0.01

        self._episode_over = True


# Load a checkpoint and play a few episodes on screen.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/baseline/seed_42/latest.pt"
    )
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.08,
                        help="Seconds between frames. Raise it to slow "
                             "the animation down.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use the mean action instead of sampling. "
                             "Shows the policy without its noise.")
    parser.add_argument("--seed", type=int, default=900_000_000,
                        help="Starting layout seed. These are disjoint "
                             "from every training layout.")
    args = parser.parse_args()

    env = RenderableEnvironment()

    maddpg = MADDPGSystem(
        n_agents=3,
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        device="cpu",
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    maddpg.actor.load_state_dict(ckpt["actor"])
    maddpg.actor.eval()

    print(f"Loaded {args.checkpoint}")
    print(f"Mode: {'deterministic' if args.deterministic else 'stochastic'}")
    print("-" * 56)

    metrics = InstantaneousBehaviourMetrics(n_agents=3)

    rewards = []

    for ep in range(args.episodes):
        observations = env.reset(seed=args.seed + ep)
        total = 0.0

        while True:
            actions = maddpg.act(
                observations,
                deterministic=args.deterministic
            )

            metrics.record(actions)

            observations, reward, terminated, truncated, _ = env.step(actions)
            total += reward

            time.sleep(args.delay)

            if terminated or truncated:
                break

        stats = metrics.get_episode_results()
        metrics.reset()
        rewards.append(total)

        cos = stats["cosine_mean"]

        if cos > 0.5:
            verdict = "CLUMPED  <- homogenised, agents moved together"
        elif cos < -0.3:
            verdict = "opposed"
        else:
            verdict = "spread out"

        print(
            f"Episode {ep + 1:3d} | reward {total:9.2f} | "
            f"cos {cos:+6.3f} | {verdict}"
        )

    print("-" * 56)
    print(f"mean reward over {args.episodes} episodes: "
          f"{np.mean(rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()
