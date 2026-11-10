# Experiment 3 (A2, A2c, A2z): MADDPG with a whole-episode token-diversity bonus of weight beta.

import os
import sys
import time
import random
import argparse
from collections import deque

import numpy as np
import torch


# Let this script import modules from the project root.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


from environment import MPEEnvironment
from maddpg_system import MADDPGSystem
from replay_buffer import ReplayBuffer
from instantaneous_metrics import InstantaneousBehaviourMetrics
from temporal_behaviour import TemporalBehaviourMetrics
from policy_divergence_behaviour import PolicyDivergenceMetrics
from token_diversity import (
    tokenise,
    episode_diversity,
    permutation_null,
    max_diversity,
)


# Command-line options.
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--episodes", type=int, default=50000)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--update-every", type=int, default=1,
                    help="Gradient updates are done every k env steps. "
                         "k=1 reproduces the original schedule; k=2 or 4 "
                         "roughly halves/quarters wall-clock with little "
                         "loss on this task.")
parser.add_argument("--print-every", type=int, default=1)
parser.add_argument("--checkpoint-every", type=int, default=2000)
parser.add_argument("--alpha", type=float, default=0.0,
                    help="Policy-entropy coefficient. Experiment 3 runs at "
                         "0.0 so that this arm moves D only and Experiment 2 "
                         "moves sigma only. Set it > 0 only for a deliberate "
                         "combined arm.")
parser.add_argument("--beta", type=float, required=True,
                    help="Trajectory-diversity coefficient, in REWARD UNITS. "
                         "D is normalised to [0, 1], so beta is the maximum "
                         "bonus an episode can earn against a task return of "
                         "roughly -29. Sensible bracket: 1 to 10. "
                         "--beta 0 reduces to Experiment 2 -- see the "
                         "module docstring for exactly how far.")
parser.add_argument("--bonus-schedule", choices=["terminal", "spread"],
                    default="terminal",
                    help="terminal: the whole bonus on the last transition, "
                         "which is where D is actually defined. spread: paid "
                         "in equal instalments across the episode, for when "
                         "gamma^24 = 0.29 attenuates the terminal signal too "
                         "far to reach the early actions.")
parser.add_argument("--augment-critic", action="store_true",
                    help="Give the CENTRALISED CRITIC the 3x3 running "
                         "token-count matrix (9 numbers, each n_i(x)/T) "
                         "alongside the joint observation and action. This "
                         "is arm A2c. At the terminal step the matrix IS "
                         "the frequency matrix f, so the terminal bonus "
                         "beta*D_norm becomes a deterministic function of "
                         "the critic's own input instead of an "
                         "unpredictable shock. The ACTOR is untouched, so "
                         "execution stays decentralised and the policy "
                         "class is identical to every other arm. Omitting "
                         "the flag reproduces arm A2 exactly.")
parser.add_argument("--null-every", type=int, default=500,
                    help="Episodes between permutation-null evaluations. The "
                         "null is the empirical zero for D and is reported "
                         "against it; it does not enter the reward.")
args = parser.parse_args()

# The diversity bonus weight cannot be negative.
if args.beta < 0.0:
    raise SystemExit(
        f"--beta must be >= 0 (received {args.beta})."
    )

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
BONUS_SCHEDULE = args.bonus_schedule
N_LANDMARKS = 3
D_MAX_BITS = max_diversity(N_AGENTS, N_LANDMARKS)

AUGMENT_CRITIC = args.augment_critic
CONTEXT_DIM = (N_AGENTS * N_LANDMARKS) if AUGMENT_CRITIC else 0

DEVICE = "cpu"

EXPERIMENT = (
    f"beta_{BETA:g}" if ALPHA == 0.0
    else f"beta_{BETA:g}_alpha_{ALPHA:g}"
)

if BONUS_SCHEDULE != "terminal":
    EXPERIMENT += f"_{BONUS_SCHEDULE}"
if AUGMENT_CRITIC:
    EXPERIMENT += "_augcritic"

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
token_frequencies = []
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
    "bonus_schedule": BONUS_SCHEDULE,
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


print("-" * 64)
print(f"MADDPG + TRAJECTORY DIVERSITY  (Experiment 3, beta = {BETA:g})")
print("-" * 64)
for k, v in [
    ("Seed", SEED),
    ("Episodes", NUM_EPISODES),
    ("Batch size", BATCH_SIZE),
    ("Update every", f"{UPDATE_EVERY} env steps"),
    ("Warmup steps", WARMUP_STEPS),
    ("Gamma / Tau", f"{GAMMA} / {TAU}"),
    ("LR actor / critic", f"{LR_ACTOR} / {LR_CRITIC}"),
    ("Grad clip / Huber delta", f"{GRAD_CLIP_NORM} / {HUBER_DELTA}"),
    ("Twin critics / delay", f"{TWIN_CRITICS} / {POLICY_DELAY}"),
    ("Entropy coefficient", ALPHA),
    ("Diversity coefficient", f"{BETA:g}  (max bonus per episode)"),
    ("Bonus schedule", BONUS_SCHEDULE),
    ("Critic context", f"{CONTEXT_DIM} "
                       f"({'3x3 running counts / T' if AUGMENT_CRITIC else 'none -- arm A2'})"),
    ("D bound", f"log2 min({N_AGENTS}, {N_LANDMARKS}) = {D_MAX_BITS:.4f} bits"),
    ("Results dir", RESULTS_DIR),
]:
    print(f"  {k:24s}: {v}")

planned_updates = (NUM_EPISODES * MAX_CYCLES) // UPDATE_EVERY
print(f"  {'Planned updates':24s}: {planned_updates:,}")
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

    token_counts = np.zeros((N_AGENTS, N_LANDMARKS), dtype=np.int64)

    while True:
        global_step += 1
        episode_steps += 1

        # Random actions during warmup, then actions from the policy.
        if global_step < WARMUP_STEPS:
            actions = action_rng.random(
                (N_AGENTS, ACTION_DIM)
            ).astype(np.float32)
        else:
            actions = maddpg.act(observations, deterministic=False)

        # Record this step for the behaviour metrics.
        instant_metrics.record(actions)
        temporal_metrics.record(actions)

        means, stds = maddpg.policy_params(observations)
        divergence_metrics.record(means, stds)

        (
            next_observations,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(actions)

        # Which landmark each agent is nearest to after this step.
        token_t = tokenise(next_observations, N_LANDMARKS)
        episode_tokens.append(token_t)

        # Augmented critic: running landmark counts before and after this step.
        if CONTEXT_DIM:
            ctx_before = token_counts.reshape(-1) / MAX_CYCLES
            token_counts[np.arange(N_AGENTS), token_t] += 1
            ctx_after = token_counts.reshape(-1) / MAX_CYCLES
        else:
            ctx_before = ctx_after = None

        # Hold transitions until the episode ends, so the bonus can be added first.
        pending.append(
            [observations, actions, reward,
             next_observations, terminated,
             ctx_before, ctx_after]
        )

        observations = next_observations
        episode_reward += reward

        if (
            len(replay_buffer) >= BATCH_SIZE
            and global_step >= WARMUP_STEPS
            and global_step % UPDATE_EVERY == 0
        ):
            # One gradient update of the critics and actor from a replay sample.
            info_update = maddpg.update(
                replay_buffer.sample(BATCH_SIZE)
            )

            critic_losses.append(info_update["critic_loss"])
            actor_losses.append(info_update["actor_loss"])
            mean_q_values.append(info_update["mean_q"])
            policy_stds.append(info_update["policy_std"])
            critic_disagreements.append(
                info_update["critic_disagreement"]
            )
            update_steps.append(global_step)
            update_count += 1

        if terminated or truncated:
            break

    # Token diversity D for the finished episode.
    tokens = np.asarray(episode_tokens)

    d_bits, d_norm, token_f = episode_diversity(
        tokens, n_landmarks=N_LANDMARKS
    )

    # Check the critic's final context equals the episode's frequencies.
    if CONTEXT_DIM:
        assert np.allclose(
            pending[-1][6].reshape(N_AGENTS, N_LANDMARKS),
            token_f,
            atol=1e-6,
        ), "terminal context is not the frequency matrix f"

    # Diversity bonus: on the final step, or spread evenly over all steps.
    bonus = BETA * d_norm

    if BONUS_SCHEDULE == "terminal":
        pending[-1][2] += bonus
    else:
        share = bonus / len(pending)
        for transition in pending:
            transition[2] += share

    # Store the episode's transitions in the replay buffer.
    for (observation_t, action_t, reward_t, next_t, done_t,
         ctx_t, next_ctx_t) in pending:
        replay_buffer.add(
            observation_t, action_t, reward_t, next_t, done_t,
            ctx_t, next_ctx_t
        )

    step_rewards = [transition[2] for transition in pending]

    # Every few hundred episodes, estimate the chance level of D by shuffling agents.
    if (episode + 1) % args.null_every == 0:
        null_mean, _ = permutation_null(
            tokens,
            n_landmarks=N_LANDMARKS,
            n_permutations=200,
            rng=null_rng,
        )
        null_episodes.append(episode + 1)
        null_values.append(null_mean)

    episode_rewards.append(episode_reward)
    episode_shaped_rewards.append(episode_reward + bonus)
    episode_diversity_bits.append(d_bits)
    episode_diversity_norm.append(d_norm)
    token_frequencies.append(token_f)

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
        line = (
            f"Episode {episode + 1:6d}"
            f" | Reward: {episode_reward:9.3f}"
            f" | Steps: {episode_steps:2d}"
            f" | Buffer: {len(replay_buffer):6d}"
        )

        if mean_q_values:
            emp = float(np.mean(episode_value_avg[-REWARD_WINDOW:]))

            line += (
                f" | Critic: {critic_losses[-1]:8.4f}"
                f" | Actor: {actor_losses[-1]:8.4f}"
                f" | Q: {mean_q_values[-1]:9.4f}"
                f" | G: {emp:9.4f}"
                f" | sig: {policy_stds[-1]:6.4f}"
                f" | dQ: {critic_disagreements[-1]:6.3f}"
            )

        else:
            line += "  (warmup: random actions, no learning yet)"

        line += (
            f" | cos: {record['cosine_mean']:+6.3f}"
            f" | D: {d_norm:5.3f}"
            f" | Avg{REWARD_WINDOW}: {running_mean_rewards[-1]:8.2f}"
        )

        if (episode + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (episode + 1) / elapsed
            line += f" | ETA {(NUM_EPISODES - episode - 1) / rate / 3600:5.2f}h"

        print(line, flush=True)

    # Save a periodic checkpoint.
    if (episode + 1) % args.checkpoint_every == 0:
        maddpg.save(
            os.path.join(
                CHECKPOINT_DIR, f"ckpt_ep{episode + 1}.pt"
            ),
            metadata=CHECKPOINT_METADATA,
        )

    if (
        episode + 1 >= BEST_BURN_IN
        and running_mean_rewards[-1] > best_reward
    ):
        # Save best.pt whenever the 100-episode average reward improves.
        best_reward = running_mean_rewards[-1]
        best_episode = episode + 1
        maddpg.save(
            os.path.join(CHECKPOINT_DIR, "best.pt"),
            metadata={**CHECKPOINT_METADATA,
                      "best_episode": best_episode,
                      "best_avg_reward": best_reward},
        )


print("-" * 64)
print(f"Total environment steps : {global_step:,}")
print(f"Total gradient updates  : {update_count:,}")
print(f"Wall clock              : {(time.time()-start_time)/3600:.2f} h")

# Save all training logs to the results folder.
arrays = {
    "episode_rewards": episode_rewards,
    "episode_shaped_rewards": episode_shaped_rewards,
    "episode_diversity_norm": episode_diversity_norm,
    "episode_diversity_bits": episode_diversity_bits,
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
    np.save(
        os.path.join(RESULTS_DIR, f"{name}.npy"),
        np.array(values)
    )

behaviour_keys = sorted(behaviour_history[0].keys())

np.savez(
    os.path.join(RESULTS_DIR, "behaviour_history.npz"),
    **{
        k: np.array([r[k] for r in behaviour_history])
        for k in behaviour_keys
    }
)

np.save(
    os.path.join(RESULTS_DIR, "token_frequencies.npy"),
    np.asarray(token_frequencies)
)

if null_values:
    print(f"Permutation null (mean over {len(null_values)} "
          f"evaluations): {float(np.mean(null_values)):.4f}")
print(f"Mean D over last 100 episodes: "
      f"{float(np.mean(episode_diversity_norm[-100:])):.4f}")

# Save the final model.
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "final.pt")

maddpg.save(MODEL_PATH, metadata=CHECKPOINT_METADATA)

print(f"Best policy: episode {best_episode} "
      f"(Avg{REWARD_WINDOW} = {best_reward:.3f}) -> "
      f"{os.path.join(CHECKPOINT_DIR, 'best.pt')}")

print(f"Results -> {RESULTS_DIR}")
print(f"Model   -> {MODEL_PATH}")
print("-" * 64)

env.close()
