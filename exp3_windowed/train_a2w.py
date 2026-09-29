"""
train_a2w.py

Experiment 3, rebuilt: arm A2 (and A2c with --augment-critic) using the
WINDOWED diversity measure.

This file is exp3/train_exp3.py with exactly these changes:

  1. THE MEASURE. The terminal bonus is beta * D_win_norm, where

         D_w        = H(f_bar_w) - (1/N) sum_i H(f_i,w)     per 5-step window
         D_win      = (1/5) sum_w D_w                       = I(X; Z | W)
         D_win_norm = D_win / log2(3)                       in [0, 1]

     computed by windowed_diversity.py. With --n-windows 1 this is exactly
     the old D, so --n-windows 1 must reproduce train_exp3.py (checked by
     test_a2w.py). Everything else about the bonus is unchanged: paid once,
     on the last transition, beta in reward units, alpha = 0.

  2. THE A2c CONTEXT (--augment-critic). 11 numbers instead of 9:
         current window's counts / 5            (9)
         finished windows' D / (5 log2 3)       (1)
         window index (k+1) / 5                 (1)
     The counts roll into the finished sum one step AFTER a window ends, so
     the last transition's context reproduces the bonus exactly. That is
     asserted every episode (bonus_from_context).

  3. SAFER DEFAULTS. --batch-size 256 and --update-every 2, the values every
     real run used. train_exp3.py defaulted to 1024 / 1, so a forgotten flag
     silently broke comparability.

  4. REMOVED --bonus-schedule spread. Decision A fixed the bonus as terminal.

  5. LOGGING. The raw per-step tokens (episodes x 25 x 3, int8) and the five
     per-window D values are saved, so any measure can be recomputed offline.
     The permutation null is the WITHIN-WINDOW shuffle.

Results go to results/beta_<b>_w<n>[ _augcritic]/seed_<s>/, so the old
beta_3/ and beta_3_augcritic/ trees are never touched.

Everything else -- the actor, the algorithm, held transitions, terminated-only
bootstrapping, episode seeds, warmup, checkpoints, best.pt on the UNSHAPED
return -- is identical to train_exp3.py.
"""

import os
import sys
import time
import random
import argparse
from collections import deque

import numpy as np
import torch

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


# =========================================================
# ARGUMENTS
# =========================================================

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
parser.add_argument("--null-every", type=int, default=500)
args = parser.parse_args()

if args.beta < 0.0:
    raise SystemExit(f"--beta must be >= 0 (received {args.beta}).")

SEED = args.seed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================================================
# CONFIGURATION
# =========================================================

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
if MAX_CYCLES % N_WINDOWS != 0:
    raise SystemExit(f"--n-windows {N_WINDOWS} must divide {MAX_CYCLES}.")

AUGMENT_CRITIC = args.augment_critic
CONTEXT_DIM = (N_AGENTS * N_LANDMARKS + 2) if AUGMENT_CRITIC else 0

DEVICE = "cpu"

# Every setting that changes what the run IS appears in the name.
EXPERIMENT = f"beta_{BETA:g}_w{N_WINDOWS}"
if ALPHA != 0.0:
    EXPERIMENT += f"_alpha_{ALPHA:g}"
if AUGMENT_CRITIC:
    EXPERIMENT += "_augcritic"

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", EXPERIMENT, f"seed_{SEED}")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", EXPERIMENT, f"seed_{SEED}")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =========================================================
# SETUP
# =========================================================

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

# Separate stream: the null can never shift the action or replay streams.
null_rng = np.random.default_rng(SEED + 7_000_000)


# =========================================================
# STORAGE
# =========================================================

episode_rewards = []            # UNSHAPED -- reported, and used for best.pt
episode_shaped_rewards = []
episode_diversity_norm = []
episode_diversity_bits = []
window_diversity_bits = []      # (episodes, N_WINDOWS)
episode_tokens_log = []         # (episodes, T, N) int8
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


# =========================================================
# BANNER
# =========================================================

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
    ("Results dir", RESULTS_DIR),
]:
    print(f"  {k:24s}: {v}")
if BATCH_SIZE != 256 or UPDATE_EVERY != 2:
    print("  WARNING: batch size / update frequency differ from every real run "
          "(256 / 2). This breaks comparability unless it is deliberate.")
print(f"  {'Planned updates':24s}: {(NUM_EPISODES * MAX_CYCLES) // UPDATE_EVERY:,}")
print("-" * 64)


# =========================================================
# TRAINING LOOP
# =========================================================

start_time = time.time()

for episode in range(NUM_EPISODES):

    episode_seed = SEED * 1_000_000 + episode
    observations = env.reset(seed=episode_seed)

    episode_reward = 0.0
    episode_steps = 0
    step_rewards = []

    pending = []            # transitions held until the bonus is known
    episode_tokens = []     # one (N,) token vector per step

    while True:

        global_step += 1
        episode_steps += 1

        # 1. ACTION
        if global_step < WARMUP_STEPS:
            actions = action_rng.random((N_AGENTS, ACTION_DIM)).astype(np.float32)
        else:
            actions = maddpg.act(observations, deterministic=False)

        # 2. RECORD BEHAVIOUR
        instant_metrics.record(actions)
        temporal_metrics.record(actions)
        means, stds = maddpg.policy_params(observations)
        divergence_metrics.record(means, stds)

        # 3. STEP
        next_observations, reward, terminated, truncated, info = env.step(actions)

        # 3b. TOKENISE (from next_observations: every token reflects an action)
        token_t = tokenise(next_observations, N_LANDMARKS)

        # 3b-ii. CRITIC CONTEXT (A2c only): before and after this step's token
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

        # 3c. HOLD THE TRANSITION (terminated only -- never `or truncated`)
        pending.append([observations, actions, reward, next_observations,
                        terminated, ctx_before, ctx_after])

        observations = next_observations
        episode_reward += reward

        # 4. UPDATE
        if (len(replay_buffer) >= BATCH_SIZE
                and global_step >= WARMUP_STEPS
                and global_step % UPDATE_EVERY == 0):
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

    # =====================================================
    # END OF EPISODE -- THE WINDOWED BONUS
    # =====================================================

    tokens = np.asarray(episode_tokens)                        # [T, N]
    d_bits, d_norm, d_per_window = windowed_diversity(tokens.T, N_WINDOWS, N_LANDMARKS)

    # A2c's claim, checked every episode: the last context reproduces the bonus.
    if CONTEXT_DIM:
        recovered = bonus_from_context(pending[-1][6], N_AGENTS, N_WINDOWS, N_LANDMARKS)
        assert abs(recovered - d_norm) < 1e-9, (
            f"terminal context gives {recovered}, bonus uses {d_norm}")

    bonus = BETA * d_norm
    pending[-1][2] += bonus                                    # terminal only

    for (o, a, r, o2, done, c, c2) in pending:
        replay_buffer.add(o, a, r, o2, done, c, c2)

    # Shaped rewards, so the Q-vs-G health check compares like with like.
    step_rewards = [transition[2] for transition in pending]

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

    # PROGRESS
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

    # CHECKPOINTS (fixed grid + best.pt on the UNSHAPED rolling mean)
    if (episode + 1) % args.checkpoint_every == 0:
        maddpg.save(os.path.join(CHECKPOINT_DIR, f"ckpt_ep{episode + 1}.pt"),
                    metadata=CHECKPOINT_METADATA)

    if episode + 1 >= BEST_BURN_IN and running_mean_rewards[-1] > best_reward:
        best_reward = running_mean_rewards[-1]
        best_episode = episode + 1
        maddpg.save(os.path.join(CHECKPOINT_DIR, "best.pt"),
                    metadata={**CHECKPOINT_METADATA,
                              "best_episode": best_episode,
                              "best_avg_reward": best_reward})


# =========================================================
# SAVE
# =========================================================

print("-" * 64)
print(f"Total environment steps : {global_step:,}")
print(f"Total gradient updates  : {update_count:,}")
print(f"Wall clock              : {(time.time() - start_time) / 3600:.2f} h")

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