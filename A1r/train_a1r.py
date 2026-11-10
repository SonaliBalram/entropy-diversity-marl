import os
import sys
import time
import random
import json
import argparse
from collections import deque

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


from environment import MPEEnvironment
from maddpg_system_a1r import MADDPGSystem
from voronoi_gate import OwnershipGate
from replay_buffer import ReplayBuffer
from instantaneous_metrics import InstantaneousBehaviourMetrics
from temporal_behaviour import TemporalBehaviourMetrics
from policy_divergence_behaviour import PolicyDivergenceMetrics


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
parser.add_argument("--alpha", type=float, required=True,
                    help="Entropy BUDGET. The gate redistributes this "
                         "across states without changing its geometric "
                         "mean. Use 0.01 to stay matched to A1.")
parser.add_argument("--gate", default="both",
                    choices=["none", "both", "ownership", "landmark"],
                    help="none = no gate at all, reproduces Exp 2 "
                         "bit-identically (reduction check R1). "
                         "both = A1r as designed.")
parser.add_argument("--constants",
                    default="results/a1r_pilot/a1r_constants.json",
                    help="frozen kappa and log Z from a1r_pilot.py, "
                         "relative to the project root")
parser.add_argument("--kappa", type=float, default=None,
                    help="override the frozen kappa (for the kappa "
                         "sweep only; leave unset for the real run)")
parser.add_argument("--log-z", type=float, default=None,
                    help="override the frozen log Z")
parser.add_argument("--clamp-ratio", type=float, default=10.0,
                    help="alpha_eff is clamped to alpha x [1/r, r]")
args = parser.parse_args()

if args.alpha <= 0.0:
    raise SystemExit(
        f"--alpha must be > 0 for A1r (received {args.alpha}). "
        "A1r reallocates an entropy budget; it cannot reallocate zero."
    )


A1R_CONSTANTS = None
GATE = None

if args.gate != "none":
    cpath = os.path.join(PROJECT_ROOT, args.constants)
    if not os.path.exists(cpath):
        raise SystemExit(
            f"missing {cpath}\n"
            "Run  python A1r/a1r_pilot.py  first: it freezes kappa and "
            "log Z from a reference set of states, and reports the "
            "kill-switch verdict."
        )
    with open(cpath) as _f:
        A1R_CONSTANTS = json.load(_f)

    if A1R_CONSTANTS.get("verdict", "").startswith("DO NOT RUN"):
        raise SystemExit(
            f"the pilot verdict was {A1R_CONSTANTS['verdict']!r}. "
            "Re-read results/a1r_pilot/report.txt before overriding."
        )

    GATE = OwnershipGate(
        alpha=args.alpha,
        kappa=(args.kappa if args.kappa is not None
               else A1R_CONSTANTS["kappa"]),
        log_z=(args.log_z if args.log_z is not None
               else A1R_CONSTANTS["log_z"]),
        mode=args.gate,
        clamp_ratio=args.clamp_ratio,
    )

SEED = args.seed


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


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

DEVICE = "cpu"

EXPERIMENT = f"a1r_{args.gate}"

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", EXPERIMENT, f"seed_{SEED}")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", EXPERIMENT, f"seed_{SEED}")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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
    alpha_fn=GATE,
    device=DEVICE,
)

replay_buffer = ReplayBuffer(
    capacity=BUFFER_CAPACITY,
    n_agents=N_AGENTS,
    observation_dim=OBSERVATION_DIM,
    action_dim=ACTION_DIM,
    seed=SEED,
)

instant_metrics = InstantaneousBehaviourMetrics(n_agents=N_AGENTS)
temporal_metrics = TemporalBehaviourMetrics(N_AGENTS, ACTION_DIM)
divergence_metrics = PolicyDivergenceMetrics(N_AGENTS, ACTION_DIM)

action_rng = np.random.default_rng(SEED)


episode_rewards = []
episode_discounted = []
episode_value_avg = []
critic_losses = []
actor_losses = []
mean_q_values = []
policy_stds = []
critic_disagreements = []
update_steps = []
gate_geomeans = []
gate_spreads = []

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
    "lr_actor": LR_ACTOR,
    "lr_critic": LR_CRITIC,
    "grad_clip_norm": GRAD_CLIP_NORM,
    "huber_delta": HUBER_DELTA,
    "batch_size": BATCH_SIZE,
    "warmup_steps": WARMUP_STEPS,
    "update_every": UPDATE_EVERY,
    "policy_delay": POLICY_DELAY,
    "twin_critics": TWIN_CRITICS,
    "a1r_gate_mode": args.gate,
    "a1r_constants": A1R_CONSTANTS,
}


print("-" * 64)
print(f"MADDPG + OWNERSHIP-GATED ENTROPY  (A1r, gate = {args.gate})")
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
    ("Entropy budget alpha", ALPHA),
    ("Gate", args.gate if GATE is None else repr(GATE)),
    ("Results dir", RESULTS_DIR),
]:
    print(f"  {k:24s}: {v}")

planned_updates = (NUM_EPISODES * MAX_CYCLES) // UPDATE_EVERY
print(f"  {'Planned updates':24s}: {planned_updates:,}")
print("-" * 64)


start_time = time.time()

for episode in range(NUM_EPISODES):
    episode_seed = SEED * 1_000_000 + episode

    observations = env.reset(seed=episode_seed)

    episode_reward = 0.0
    episode_steps = 0
    step_rewards = []

    while True:
        global_step += 1
        episode_steps += 1

        if global_step < WARMUP_STEPS:
            actions = action_rng.random(
                (N_AGENTS, ACTION_DIM)
            ).astype(np.float32)
        else:
            actions = maddpg.act(observations, deterministic=False)

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

        replay_buffer.add(
            observations,
            actions,
            reward,
            next_observations,
            terminated
        )

        observations = next_observations
        episode_reward += reward
        step_rewards.append(reward)

        if (
            len(replay_buffer) >= BATCH_SIZE
            and global_step >= WARMUP_STEPS
            and global_step % UPDATE_EVERY == 0
        ):
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
            gate_geomeans.append(
                info_update.get("alpha_eff_geomean", float("nan")))
            gate_spreads.append(
                info_update.get("alpha_eff_spread_decades",
                                float("nan")))
            update_steps.append(global_step)
            update_count += 1

        if terminated or truncated:
            break

    episode_rewards.append(episode_reward)

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
    behaviour_history.append(record)

    instant_metrics.reset()
    temporal_metrics.reset()
    divergence_metrics.reset()

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
            f" | Avg{REWARD_WINDOW}: {running_mean_rewards[-1]:8.2f}"
        )

        if (episode + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (episode + 1) / elapsed
            line += f" | ETA {(NUM_EPISODES - episode - 1) / rate / 3600:5.2f}h"

        print(line, flush=True)

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

arrays = {
    "episode_rewards": episode_rewards,
    "episode_discounted": episode_discounted,
    "episode_value_avg": episode_value_avg,
    "critic_losses": critic_losses,
    "actor_losses": actor_losses,
    "mean_q_values": mean_q_values,
    "policy_stds": policy_stds,
    "critic_disagreements": critic_disagreements,
    "update_steps": update_steps,
    "gate_geomeans": gate_geomeans,
    "gate_spreads": gate_spreads,
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

MODEL_PATH = os.path.join(CHECKPOINT_DIR, "final.pt")

maddpg.save(MODEL_PATH, metadata=CHECKPOINT_METADATA)

print(f"Best policy: episode {best_episode} "
      f"(Avg{REWARD_WINDOW} = {best_reward:.3f}) -> "
      f"{os.path.join(CHECKPOINT_DIR, 'best.pt')}")

print(f"Results -> {RESULTS_DIR}")
print(f"Model   -> {MODEL_PATH}")
print("-" * 64)

env.close()
