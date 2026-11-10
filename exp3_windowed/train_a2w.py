# Windowed Experiment 3 (A2w, A2cw, A2cwT): diversity bonus measured over 5 windows per episode.

import os
import sys
import time
import random
import argparse
from collections import deque

import numpy as np
import torch

# Let this script import from the project root and this folder.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)

from environment import MPEEnvironment
from maddpg_system import MADDPGSystem
from replay_buffer import ReplayBuffer
from instantaneous_metrics import InstantaneousBehaviourMetrics
from temporal_behaviour import TemporalBehaviourMetrics
from policy_divergence_behaviour import PolicyDivergenceMetrics
from token_diversity import tokenise
from windowed_diversity import (
    windowed_diversity,
    permutation_null_windowed,
    critic_context,
    critic_context_empty,
    bonus_from_context,
)


# Command-line options.
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--episodes", type=int, default=50000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--update-every", type=int, default=2)
parser.add_argument("--print-every", type=int, default=1)
parser.add_argument("--checkpoint-every", type=int, default=2000)
parser.add_argument("--alpha", type=float, default=0.0,
                    help="Policy-entropy coefficient. Stays 0 for A2 / A2c.")
parser.add_argument("--beta", type=float, required=True,
                    help="Maximum bonus per episode, in reward units.")
parser.add_argument("--n-windows", type=int, default=5,
                    help="Equal windows per 25-step episode. 5 is the design; "
                         "1 reproduces the old whole-episode D (reduction check).")
parser.add_argument("--augment-critic", action="store_true",
                    help="Arm A2c: give the centralised critic the 11-number "
                         "windowed context. The actor is untouched.")
parser.add_argument("--terminal-at-limit", action="store_true",
                    help="A2c only: store done=1 on the final transition, so the "
                         "critic does not bootstrap from the state after the time "
                         "limit. Correct when the critic can tell the time, which "
                         "the windowed context allows (Pardo et al., 2018).")
parser.add_argument("--null-every", type=int, default=500)
args = parser.parse_args()
# --terminal-at-limit only makes sense with the augmented critic.
if args.terminal_at_limit and not args.augment_critic:
    parser.error("--terminal-at-limit requires --augment-critic: without the "
                 "context the critic cannot tell the time, so the time limit "
                 "is a truncation and must be bootstrapped through.")

# The diversity bonus weight cannot be negative.
if args.beta < 0.0:
    raise SystemExit(f"--beta must be >= 0 (received {args.beta}).")

SEED = args.seed


# Seed every random number generator so runs are reproducible.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# Environment, network and training settings.
N_AGENTS = 3
OBSERVATION_DIM = 18
ACTION_DIM = 5
MAX_CYCLES = 25

NUM_EPISODES = args.episodes
BATCH_SIZE = args.batch_size
UPDATE_EVERY = args.update_every
BUFFER_CAPACITY = 100000
WARMUP_STEPS = max(5000, 5 * BATCH_SIZE)

HIDDEN_DIM = 128
GAMMA = 0.95
TAU = 0.005
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GRAD_CLIP_NORM = 10.0
HUBER_DELTA = 10.0
TWIN_CRITICS = True
POLICY_DELAY = 2

ALPHA = args.alpha
BETA = args.beta
N_LANDMARKS = 3
N_WINDOWS = args.n_windows
# The episode length must split evenly into windows.
if MAX_CYCLES % N_WINDOWS != 0:
    raise SystemExit(f"--n-windows {N_WINDOWS} must divide {MAX_CYCLES}.")

AUGMENT_CRITIC = args.augment_critic
CONTEXT_DIM = (N_AGENTS * N_LANDMARKS + 2) if AUGMENT_CRITIC else 0

DEVICE = "cpu"

EXPERIMENT = f"beta_{BETA:g}_w{N_WINDOWS}"
if ALPHA != 0.0:
    EXPERIMENT += f"_alpha_{ALPHA:g}"
if AUGMENT_CRITIC:
    EXPERIMENT += "_augcritic"
if args.terminal_at_limit:
    EXPERIMENT += "_term"

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", EXPERIMENT, f"seed_{SEED}")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", EXPERIMENT, f"seed_{SEED}")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Create the environment, learner, replay buffer and behaviour metrics.
env = MPEEnvironment(n_agents=N_AGENTS, max_cycles=MAX_CYCLES)

maddpg = MADDPGSystem(
    n_agents=N_AGENTS,
    observation_dim=OBSERVATION_DIM,
    action_dim=ACTION_DIM,
    hidden_dim=HIDDEN_DIM,
    gamma=GAMMA,
    tau=TAU,
    lr_actor=LR_ACTOR,
    lr_critic=LR_CRITIC,
    grad_clip_norm=GRAD_CLIP_NORM,
    huber_delta=HUBER_DELTA,
    twin_critics=TWIN_CRITICS,
    policy_delay=POLICY_DELAY,
    alpha=ALPHA,
    device=DEVICE,
    context_dim=CONTEXT_DIM,
)

replay_buffer = ReplayBuffer(
    capacity=BUFFER_CAPACITY,
    n_agents=N_AGENTS,
    observation_dim=OBSERVATION_DIM,
    action_dim=ACTION_DIM,
    seed=SEED,
    context_dim=CONTEXT_DIM,
)

instant_metrics = InstantaneousBehaviourMetrics(n_agents=N_AGENTS)
temporal_metrics = TemporalBehaviourMetrics(N_AGENTS, ACTION_DIM)
divergence_metrics = PolicyDivergenceMetrics(N_AGENTS, ACTION_DIM)

action_rng = np.random.default_rng(SEED)

null_rng = np.random.default_rng(SEED + 7_000_000)


# Lists that collect the training logs.
episode_rewards = []
episode_shaped_rewards = []
episode_diversity_norm = []
episode_diversity_bits = []
window_diversity_bits = []
episode_tokens_log = []
null_episodes = []
null_values = []
episode_discounted = []
episode_value_avg = []
critic_losses = []
actor_losses = []
mean_q_values = []
policy_stds = []
critic_disagreements = []
update_steps = []

REWARD_WINDOW = 100
reward_window = deque(maxlen=REWARD_WINDOW)
running_mean_rewards = []
running_std_rewards = []

behaviour_history = []

global_step = 0
update_count = 0

best_reward = -float("inf")
best_episode = -1
BEST_BURN_IN = 2000


# Run settings stored inside every saved checkpoint.
CHECKPOINT_METADATA = {
    "seed": SEED,
    "n_agents": N_AGENTS,
    "observation_dim": OBSERVATION_DIM,
    "action_dim": ACTION_DIM,
    "hidden_dim": HIDDEN_DIM,
    "gamma": GAMMA,
    "tau": TAU,
    "alpha": ALPHA,
    "beta": BETA,
    "measure": "windowed",
    "n_windows": N_WINDOWS,
    "bonus_schedule": "terminal",
    "augment_critic": AUGMENT_CRITIC,
    "terminal_at_limit": args.terminal_at_limit,
    "context_dim": CONTEXT_DIM,
    "n_landmarks": N_LANDMARKS,
    "lr_actor": LR_ACTOR,
    "lr_critic": LR_CRITIC,
    "grad_clip_norm": GRAD_CLIP_NORM,
    "huber_delta": HUBER_DELTA,
    "batch_size": BATCH_SIZE,
    "warmup_steps": WARMUP_STEPS,
    "update_every": UPDATE_EVERY,
    "policy_delay": POLICY_DELAY,
    "twin_critics": TWIN_CRITICS,
}


print("-" * 64)
print(f"MADDPG + WINDOWED DIVERSITY  (A2{'c' if AUGMENT_CRITIC else ''}, beta = {BETA:g})")
print("-" * 64)
for k, v in [
    ("Seed", SEED),
    ("Episodes", NUM_EPISODES),
    ("Batch size", BATCH_SIZE),
    ("Update every", f"{UPDATE_EVERY} env steps"),
    ("Warmup steps", WARMUP_STEPS),
    ("Entropy coefficient", ALPHA),
    ("Diversity coefficient", f"{BETA:g}  (max bonus per episode, terminal)"),
    ("Windows", f"{N_WINDOWS} x {MAX_CYCLES // N_WINDOWS} steps"),
    ("Critic context", f"{CONTEXT_DIM} "
                       f"({'window counts + finished D + index' if AUGMENT_CRITIC else 'none -- arm A2'})"),
    ("Final step", "terminal (no bootstrap)" if args.terminal_at_limit
                   else "bootstrapped through the time limit"),
    ("Results dir", RESULTS_DIR),
]:
    print(f"  {k:24s}: {v}")
# Warn if the run is not comparable with the other arms.
if BATCH_SIZE != 256 or UPDATE_EVERY != 2:
    print("  WARNING: batch size / update frequency differ from every real run "
          "(256 / 2). This breaks comparability unless it is deliberate.")
print(f"  {'Planned updates':24s}: {(NUM_EPISODES * MAX_CYCLES) // UPDATE_EVERY:,}")
print("-" * 64)


start_time = time.time()

# Main training loop, one episode at a time.
for episode in range(NUM_EPISODES):
    episode_seed = SEED * 1_000_000 + episode
    observations = env.reset(seed=episode_seed)

    episode_reward = 0.0
    episode_steps = 0
    step_rewards = []

    pending = []
    episode_tokens = []

    while True:
        global_step += 1
        episode_steps += 1

        # Random actions during warmup, then actions from the policy.
        if global_step < WARMUP_STEPS:
            actions = action_rng.random((N_AGENTS, ACTION_DIM)).astype(np.float32)
        else:
            actions = maddpg.act(observations, deterministic=False)

        # Record this step for the behaviour metrics.
        instant_metrics.record(actions)
        temporal_metrics.record(actions)
        means, stds = maddpg.policy_params(observations)
        divergence_metrics.record(means, stds)

        next_observations, reward, terminated, truncated, info = env.step(actions)

        # Which landmark each agent is nearest to after this step.
        token_t = tokenise(next_observations, N_LANDMARKS)

        # Augmented critic: windowed context before and after this step.
        if CONTEXT_DIM:
            ctx_before = (critic_context(np.asarray(episode_tokens), N_WINDOWS,
                                         N_LANDMARKS, MAX_CYCLES)
                          if episode_tokens else
                          critic_context_empty(N_AGENTS, N_WINDOWS, N_LANDMARKS))
        episode_tokens.append(token_t)
        if CONTEXT_DIM:
            ctx_after = critic_context(np.asarray(episode_tokens), N_WINDOWS,
                                       N_LANDMARKS, MAX_CYCLES)
        else:
            ctx_before = ctx_after = None

        # Hold transitions until the episode ends, so the bonus can be added first.
        pending.append([observations, actions, reward, next_observations,
                        terminated, ctx_before, ctx_after])

        observations = next_observations
        episode_reward += reward

        if (len(replay_buffer) >= BATCH_SIZE
                and global_step >= WARMUP_STEPS
                and global_step % UPDATE_EVERY == 0):
            # One gradient update of the critics and actor from a replay sample.
            info_update = maddpg.update(replay_buffer.sample(BATCH_SIZE))
            critic_losses.append(info_update["critic_loss"])
            actor_losses.append(info_update["actor_loss"])
            mean_q_values.append(info_update["mean_q"])
            policy_stds.append(info_update["policy_std"])
            critic_disagreements.append(info_update["critic_disagreement"])
            update_steps.append(global_step)
            update_count += 1

        if terminated or truncated:
            break

    # Token diversity D for the finished episode.
    tokens = np.asarray(episode_tokens)
    d_bits, d_norm, d_per_window = windowed_diversity(tokens.T, N_WINDOWS, N_LANDMARKS)

    # Check the critic's final context reproduces the bonus.
    if CONTEXT_DIM:
        recovered = bonus_from_context(pending[-1][6], N_AGENTS, N_WINDOWS, N_LANDMARKS)
        assert abs(recovered - d_norm) < 1e-9, (
            f"terminal context gives {recovered}, bonus uses {d_norm}")

    # Diversity bonus, added to the final step's reward.
    bonus = BETA * d_norm
    pending[-1][2] += bonus

    # A2cwT: mark the final step as terminal.
    if args.terminal_at_limit:
        pending[-1][4] = True

    # Store the episode's transitions in the replay buffer.
    for (o, a, r, o2, done, c, c2) in pending:
        replay_buffer.add(o, a, r, o2, done, c, c2)

    step_rewards = [transition[2] for transition in pending]

    # Every few hundred episodes, estimate the chance level of D by shuffling agents.
    if (episode + 1) % args.null_every == 0:
        null_episodes.append(episode + 1)
        null_values.append(permutation_null_windowed(
            tokens.T, N_WINDOWS, N_LANDMARKS, n_permutations=200, rng=null_rng))

    episode_rewards.append(episode_reward)
    episode_shaped_rewards.append(episode_reward + bonus)
    episode_diversity_bits.append(d_bits)
    episode_diversity_norm.append(d_norm)
    window_diversity_bits.append(d_per_window)
    episode_tokens_log.append(tokens.astype(np.int8))

    # Discounted return from each step, to compare with the critic's estimates.
    running = 0.0
    returns_to_go = np.empty(len(step_rewards), dtype=np.float64)
    for t_idx in range(len(step_rewards) - 1, -1, -1):
        running = step_rewards[t_idx] + GAMMA * running
        returns_to_go[t_idx] = running
    episode_discounted.append(float(returns_to_go[0]))
    episode_value_avg.append(float(returns_to_go.mean()))

    reward_window.append(episode_reward)
    running_mean_rewards.append(float(np.mean(reward_window)))
    running_std_rewards.append(float(np.std(reward_window)))

    # Collect this episode's behaviour metrics.
    record = {}
    record.update(instant_metrics.get_episode_results())
    record.update(temporal_metrics.get_episode_results())
    record.update(divergence_metrics.get_episode_results())
    record["token_diversity_bits"] = d_bits
    record["token_diversity_norm"] = d_norm
    behaviour_history.append(record)

    instant_metrics.reset()
    temporal_metrics.reset()
    divergence_metrics.reset()

    # Print a progress line.
    if (episode + 1) % args.print_every == 0 or episode == 0:
        line = (f"Episode {episode + 1:6d}"
                f" | Reward: {episode_reward:9.3f}"
                f" | Steps: {episode_steps:2d}"
                f" | Buffer: {len(replay_buffer):6d}")
        if mean_q_values:
            emp = float(np.mean(episode_value_avg[-REWARD_WINDOW:]))
            line += (f" | Critic: {critic_losses[-1]:8.4f}"
                     f" | Actor: {actor_losses[-1]:8.4f}"
                     f" | Q: {mean_q_values[-1]:9.4f}"
                     f" | G: {emp:9.4f}"
                     f" | sig: {policy_stds[-1]:6.4f}"
                     f" | dQ: {critic_disagreements[-1]:6.3f}")
        else:
            line += "  (warmup: random actions, no learning yet)"
        line += (f" | cos: {record['cosine_mean']:+6.3f}"
                 f" | Dw: {d_norm:5.3f}"
                 f" | Avg{REWARD_WINDOW}: {running_mean_rewards[-1]:8.2f}")
        if (episode + 1) % 500 == 0:
            rate = (episode + 1) / (time.time() - start_time)
            line += f" | ETA {(NUM_EPISODES - episode - 1) / rate / 3600:5.2f}h"
        print(line, flush=True)

    # Save a periodic checkpoint.
    if (episode + 1) % args.checkpoint_every == 0:
        maddpg.save(os.path.join(CHECKPOINT_DIR, f"ckpt_ep{episode + 1}.pt"),
                    metadata=CHECKPOINT_METADATA)

    if episode + 1 >= BEST_BURN_IN and running_mean_rewards[-1] > best_reward:
        # Save best.pt whenever the 100-episode average reward improves.
        best_reward = running_mean_rewards[-1]
        best_episode = episode + 1
        maddpg.save(os.path.join(CHECKPOINT_DIR, "best.pt"),
                    metadata={**CHECKPOINT_METADATA,
                              "best_episode": best_episode,
                              "best_avg_reward": best_reward})


print("-" * 64)
print(f"Total environment steps : {global_step:,}")
print(f"Total gradient updates  : {update_count:,}")
print(f"Wall clock              : {(time.time() - start_time) / 3600:.2f} h")

# Save all training logs to the results folder.
arrays = {
    "episode_rewards": episode_rewards,
    "episode_shaped_rewards": episode_shaped_rewards,
    "episode_diversity_norm": episode_diversity_norm,
    "episode_diversity_bits": episode_diversity_bits,
    "window_diversity_bits": window_diversity_bits,
    "permutation_null_episodes": null_episodes,
    "permutation_null_values": null_values,
    "episode_discounted": episode_discounted,
    "episode_value_avg": episode_value_avg,
    "critic_losses": critic_losses,
    "actor_losses": actor_losses,
    "mean_q_values": mean_q_values,
    "policy_stds": policy_stds,
    "critic_disagreements": critic_disagreements,
    "update_steps": update_steps,
    "running_mean_rewards": running_mean_rewards,
    "running_std_rewards": running_std_rewards,
}
for name, values in arrays.items():
    np.save(os.path.join(RESULTS_DIR, f"{name}.npy"), np.array(values))

np.save(os.path.join(RESULTS_DIR, "tokens.npy"), np.asarray(episode_tokens_log))

behaviour_keys = sorted(behaviour_history[0].keys())
np.savez(os.path.join(RESULTS_DIR, "behaviour_history.npz"),
         **{k: np.array([r[k] for r in behaviour_history]) for k in behaviour_keys})

if null_values:
    print(f"Within-window null (mean of {len(null_values)}): {float(np.mean(null_values)):.4f}")
print(f"Mean windowed D over last 100 episodes: "
      f"{float(np.mean(episode_diversity_norm[-100:])):.4f}")

# Save the final model.
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "final.pt")
maddpg.save(MODEL_PATH, metadata=CHECKPOINT_METADATA)

if best_episode > 0:
    print(f"Best policy: episode {best_episode} "
          f"(Avg{REWARD_WINDOW} = {best_reward:.3f}) -> "
          f"{os.path.join(CHECKPOINT_DIR, 'best.pt')}")
else:
    print(f"No best.pt written (run shorter than the {BEST_BURN_IN}-episode burn-in).")
print(f"Results -> {RESULTS_DIR}")
print(f"Model   -> {MODEL_PATH}")
print("-" * 64)

env.close()
