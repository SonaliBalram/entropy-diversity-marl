"""
train_exp3.py

Experiment 3: entropy at the TRAJECTORY level.

    Experiment 1:   J = E[Q(o, a)]
    Experiment 2:   J = E[Q(o, a)] + alpha * H(pi(.|o))
    Experiment 3:   J = E[Q(o, a)]           + beta * D(tau)

D is the tokenised trajectory diversity defined in
../token_diversity.py:

    D = H(f_bar) - (1/N) sum_i H(f_i)    = I(X ; Z)

the mutual information between agent identity and behaviour
token, normalised by log2 min(N, |A|) into [0, 1].

WHAT CHANGES RELATIVE TO train_exp2.py
--------------------------------------
1. --beta replaces --alpha as the required flag. alpha now
   defaults to 0.0, so Experiment 3 moves D only while
   Experiment 2 moves sigma only, and the two arms are
   separable. Running with --beta 0 --alpha a is
   BIT-IDENTICAL to Experiment 2, and --beta 0 --alpha 0 to
   Experiment 1, for the whole warmup window (200 episodes at
   WARMUP_STEPS = 5000 and MAX_CYCLES = 25), where no gradient
   update occurs. Verified on all 41 logged arrays.

   From the first update the runs diverge, by construction.
   The deferred write in (2) leaves this buffer up to
   MAX_CYCLES smaller during an episode, and
   ReplayBuffer.sample draws its indices from
   rng.integers(0, self.size), so a different size gives
   different mini-batches from an identical RNG state.
   Measured: at global_step 5000 the Experiment 2 buffer holds
   5000 transitions and this one 4975. The deficit is at most
   25 -- 0.5% of the buffer at the first update, under 0.05%
   by episode 2000 -- and it changes only WHEN transitions
   become visible, never which ones are stored.

   ../test_reductions.py checks both halves of this: the exact
   agreement inside the warmup window, and that the divergence
   begins at exactly the first learning episode. A different
   boundary would mean something other than the deferred write
   is changing the run, and is worth investigating before an
   arm is launched.

2. Transitions are HELD for the length of the episode instead
   of being written to the replay buffer as they occur. D is a
   function of the whole episode, so the terminal reward is
   not known until the episode ends. The buffer receives
   exactly the same transitions in the same order, at most 25
   steps later.

        r'_t = r_t                for t < T
        r'_T = r_T + beta * D     (--bonus-schedule terminal)

3. The UNSHAPED task return is tracked separately and is what
   gets reported and what best.pt is selected on, so
   Experiments 1, 2 and 3 are compared on the same number.
   The shaped return is logged alongside it, never instead.

4. D is logged every episode, along with the 3x3 token
   frequency matrix f it was computed from. f is the episode
   behaviour descriptor phi, so the secondary robustness
   check H(phi) can be estimated offline from the logged
   descriptors -- no rolling archive during training, and no
   non-stationarity.

WHAT DOES NOT CHANGE
--------------------
The actor, its inputs and the algorithm. D is not
differentiable through the environment and does not decompose
over timesteps, so it enters as a terminal shaped reward, NOT
in the actor loss. Under CTDE the deployed policy stays
byte-identical in architecture to Experiments 1 and 2, which
is what makes the comparison mean anything.

FALSIFIABLE CHECK, STATED BEFORE THE RUNS
-----------------------------------------
Tokens here come from the REALISED positions. At alpha = 0
sigma sits at the 0.0067 floor, far below the scale that moves
an agent across a Voronoi boundary between landmarks, so
quantisation discards the sampling noise outright. If sigma
RISES during an Experiment 3 run, the bonus is being gamed
through action noise rather than through behaviour -- that is
a result to investigate, not a result to report. The `sig`
column in the log is the monitor for it.

Results go to results/beta_<value>/seed_<seed>/, so neither
the Experiment 1 tree under results/baseline/ nor the
Experiment 2 tree under results/alpha_<value>/ is touched.

CORRECTIONS RELATIVE TO THE PREVIOUS VERSION
--------------------------------------------
1. env.step() now returns 5 values and only `terminated`
   goes into the replay buffer. Previously `terminated or
   truncated` was stored, which cut the Bellman bootstrap at
   every time limit and stalled the critic at roughly a
   fifth of the true value scale.

2. episode_seed = SEED * 1_000_000 + episode.
   The previous `SEED + episode` made the five runs share
   almost all their environment layouts: seeds 42 and 128
   overlapped on 49,914 of 50,000 episodes (99.8%), and
   across all five seeds only 50,958 distinct layouts were
   drawn out of 250,000 episodes. The runs therefore differed
   only in network initialisation and sampling noise, which
   is why the across-seed std (1.678) looked implausibly
   small. It was not measuring seed variability.

3. Warmup now takes UNIFORM RANDOM actions rather than
   untrained-policy actions, so the initial buffer covers the
   action space instead of the narrow band the freshly
   initialised network happens to produce.

4. UPDATE_EVERY added. One gradient update per environment
   step at batch 1024 is 1.25M updates per seed, measured at
   39.8 ms each on CPU: ~13.8 h per seed, ~69 h for five
   seeds. The startup banner now prints this estimate before
   committing.

5. Periodic checkpointing, so a crash at episode 40,000 does
   not lose the run.

6. All three behaviour metric families are now actually
   recorded. Previously only InstantaneousBehaviourMetrics
   was imported; TemporalBehaviourMetrics and
   PolicyDivergenceMetrics existed but were never called.

7. mean_q is logged against the empirical discounted return,
   which is the single diagnostic that catches a broken
   critic.
"""

import os
import sys
import time
import random
import argparse
from collections import deque

import numpy as np
import torch

# =========================================================
# LOCATE THE PROJECT FOLDER
# =========================================================
#
# This script lives in a subfolder, but the algorithm modules
# (maddpg_system.py, models/, environment.py, ...) live one
# level up. Add that folder to the import path so the imports
# below resolve, and anchor the output directories to it so
# results land in the SHARED results/ tree no matter which
# directory the script is launched from.

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


# =========================================================
# ARGUMENTS
# =========================================================

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

if args.beta < 0.0:
    raise SystemExit(
        f"--beta must be >= 0 (received {args.beta})."
    )

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

# Raised from 1.0: measured pre-clip critic gradient norms sit
# at 0.2-3.3, so a clip of 1.0 fired on most updates.
# update() now logs the pre-clip norms so this can be tuned
# from data.
GRAD_CLIP_NORM = 10.0
HUBER_DELTA = 10.0

# Clipped double-Q. Single-critic MADDPG was measured to
# diverge here (Q drifted to -0.03 against a true return of
# -22.8 while reward degraded). See maddpg_system.py note (e).
# Set TWIN_CRITICS = False to reproduce that divergence.
TWIN_CRITICS = True
POLICY_DELAY = 2

# Baseline definition: NO entropy bonus. The actor maximises
# E[Q(s,a)] only. Experiments 2-5 change this and nothing else.
ALPHA = args.alpha

# Trajectory-level term. BETA multiplies the NORMALISED D, so
# it is directly in reward units: the largest bonus a single
# episode can earn.
BETA = args.beta
BONUS_SCHEDULE = args.bonus_schedule
N_LANDMARKS = 3
D_MAX_BITS = max_diversity(N_AGENTS, N_LANDMARKS)

# Width of the critic's extra context vector. 0 is arm A2 and
# leaves the critic bit-identical to Experiments 1 and 2.
AUGMENT_CRITIC = args.augment_critic
CONTEXT_DIM = (N_AGENTS * N_LANDMARKS) if AUGMENT_CRITIC else 0

DEVICE = "cpu"

# Results are separated by beta so Experiment 3 cannot
# overwrite the Experiment 1 or Experiment 2 trees.
EXPERIMENT = (
    f"beta_{BETA:g}" if ALPHA == 0.0
    else f"beta_{BETA:g}_alpha_{ALPHA:g}"
)

# Every setting that changes what the run IS must appear in the
# directory name, or two different experiments silently write to
# the same tree. `spread` was previously missing from this name,
# so a terminal and a spread run at the same beta overwrote each
# other; `augcritic` would have had the same problem.
if BONUS_SCHEDULE != "terminal":
    EXPERIMENT += f"_{BONUS_SCHEDULE}"
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

# Dedicated RNG for the permutation null, so adding or removing
# null evaluations cannot shift the action or replay streams.
null_rng = np.random.default_rng(SEED + 7_000_000)


# =========================================================
# STORAGE
# =========================================================

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

# Best-checkpoint tracking.
best_reward = -float("inf")
best_episode = -1
BEST_BURN_IN = 2000


# =========================================================
# CHECKPOINT METADATA (written into every saved file)
# =========================================================

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


# =========================================================
# BANNER
# =========================================================

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


# =========================================================
# TRAINING LOOP
# =========================================================

start_time = time.time()

for episode in range(NUM_EPISODES):

    # Disjoint seed blocks per run -- see correction (2).
    episode_seed = SEED * 1_000_000 + episode

    observations = env.reset(seed=episode_seed)

    episode_reward = 0.0
    episode_steps = 0
    step_rewards = []

    # Held transitions and the episode's token string. Both
    # are cleared at every reset.
    pending = []
    episode_tokens = []

    # Running count matrix n_i(x), agents x landmarks, reset at
    # every episode. This is the 9-integer sufficient statistic
    # for D: at t = T, n_i(x)/T is exactly the frequency matrix
    # f that the bonus is computed from.
    token_counts = np.zeros((N_AGENTS, N_LANDMARKS), dtype=np.int64)

    while True:

        global_step += 1
        episode_steps += 1

        # -------------------------------------------------
        # 1. ACTION
        # -------------------------------------------------
        # Uniform random during warmup so the initial buffer
        # covers the action space -- see correction (3).

        if global_step < WARMUP_STEPS:
            actions = action_rng.random(
                (N_AGENTS, ACTION_DIM)
            ).astype(np.float32)
        else:
            actions = maddpg.act(observations, deterministic=False)

        # -------------------------------------------------
        # 2. RECORD BEHAVIOUR
        # -------------------------------------------------

        instant_metrics.record(actions)
        temporal_metrics.record(actions)

        means, stds = maddpg.policy_params(observations)
        divergence_metrics.record(means, stds)

        # -------------------------------------------------
        # 3. STEP
        # -------------------------------------------------

        (
            next_observations,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(actions)

        # -------------------------------------------------
        # 3b. TOKENISE
        # -------------------------------------------------
        #
        # One symbol per agent per timestep: which landmark is
        # it nearest. Read straight out of the relative
        # landmark block of the agent's own observation, so the
        # token is a function of what the agent can see and no
        # reach into env.unwrapped.world is needed.
        #
        # Tokens come from next_observations, i.e. states
        # s_1..s_T, so every symbol reflects at least one
        # action. s_0 is the random initial layout and carries
        # no information about the policy.

        token_t = tokenise(next_observations, N_LANDMARKS)
        episode_tokens.append(token_t)

        # -------------------------------------------------
        # 3b-ii. CRITIC CONTEXT  (arm A2c only)
        # -------------------------------------------------
        #
        # c_t is the running count matrix BEFORE this step's
        # token, c_{t+1} the same matrix AFTER it, both scaled
        # by 1/T. Two things follow from the 1/T scaling:
        #
        #   * every entry lies in [0, 1], so no feature
        #     normalisation is needed alongside the raw
        #     observations, and
        #   * each agent's row sums to t/T, so the episode
        #     PHASE is recoverable from the context itself and
        #     no separate time feature has to be added. That
        #     matters: a count of 4 means something different
        #     at t = 5 than at t = 24, and without phase the
        #     critic could not tell those apart.
        #
        # The transition needs BOTH because the Bellman target
        # evaluates the target critic at (s', a'), which must
        # be paired with c'.

        if CONTEXT_DIM:
            ctx_before = token_counts.reshape(-1) / MAX_CYCLES
            token_counts[np.arange(N_AGENTS), token_t] += 1
            ctx_after = token_counts.reshape(-1) / MAX_CYCLES
        else:
            ctx_before = ctx_after = None

        # -------------------------------------------------
        # 3c. HOLD THE TRANSITION
        # -------------------------------------------------
        #
        # D is a function of the WHOLE episode, so the terminal
        # reward is not known until the episode ends.
        # Transitions are held for at most MAX_CYCLES steps and
        # written below in one block with the bonus attached.
        # The buffer receives exactly the same transitions in
        # the same order; only the write is deferred.
        #
        # `terminated` is still the TRUE termination flag,
        # never `terminated or truncated` -- correction (1).

        pending.append(
            [observations, actions, reward,
             next_observations, terminated,
             ctx_before, ctx_after]
        )

        observations = next_observations
        episode_reward += reward

        # -------------------------------------------------
        # 4. UPDATE
        # -------------------------------------------------

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
            update_steps.append(global_step)
            update_count += 1

        if terminated or truncated:
            break

    # =====================================================
    # END OF EPISODE -- THE TRAJECTORY-LEVEL BONUS
    # =====================================================
    #
    #     D      = H(f_bar) - (1/N) sum_i H(f_i)   in bits
    #     D_norm = D / log2 min(N, |A|)            in [0, 1]
    #
    #     r'_t = r_t                    for t < T
    #     r'_T = r_T + beta * D_norm
    #
    # D is not differentiable through the environment and does
    # not decompose over timesteps, so it is paid terminally
    # and never enters the actor loss.

    tokens = np.asarray(episode_tokens)              # [T, N]

    d_bits, d_norm, token_f = episode_diversity(
        tokens, n_landmarks=N_LANDMARKS
    )

    # THE CLAIM THE AUGMENTED ARM RESTS ON, CHECKED EVERY
    # EPISODE (it costs nine float comparisons).
    #
    # The context attached to the LAST transition must equal
    # the frequency matrix the bonus was computed from. If it
    # does, the terminal reward is a deterministic function of
    # the critic's own input and the augmented arm is doing what
    # it claims. If it ever does not -- an off-by-one in when
    # the counts are incremented, a missed reset -- the run is
    # measuring something else and should stop here rather than
    # produce a plausible-looking curve.
    if CONTEXT_DIM:
        assert np.allclose(
            pending[-1][6].reshape(N_AGENTS, N_LANDMARKS),
            token_f,
            atol=1e-6,
        ), "terminal context is not the frequency matrix f"

    bonus = BETA * d_norm

    if BONUS_SCHEDULE == "terminal":
        pending[-1][2] += bonus
    else:
        # Equal instalments. With gamma = 0.95 and T = 25 a
        # terminal bonus reaches the first action attenuated by
        # gamma^24 = 0.29. Spreading removes that attenuation
        # at the cost of making r_t depend on the episode's
        # future, which is why `terminal` is the default.
        share = bonus / len(pending)
        for transition in pending:
            transition[2] += share

    for (observation_t, action_t, reward_t, next_t, done_t,
         ctx_t, next_ctx_t) in pending:
        replay_buffer.add(
            observation_t, action_t, reward_t, next_t, done_t,
            ctx_t, next_ctx_t
        )

    # step_rewards is rebuilt from the SHAPED rewards, because
    # the critic is trained on r' and mean_q therefore
    # estimates the shaped return. Q vs G stays a valid health
    # check only if G is computed on the same reward the critic
    # sees. episode_rewards below stays UNSHAPED and is the
    # number reported against Experiments 1 and 2.

    step_rewards = [transition[2] for transition in pending]

    # -----------------------------------------------------
    # PERMUTATION NULL -- the empirical zero for D
    # -----------------------------------------------------
    #
    # Shuffle the agent labels across the pooled token stream
    # and recompute D. This preserves T, the alphabet and the
    # pooled frequencies exactly, so what is left is the
    # plug-in estimator's upward bias, measured rather than
    # assumed. Reporting is D against D_shuffled, not against
    # zero. It never enters the reward.

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

    # -----------------------------------------------------
    # DISCOUNTED RETURN-TO-GO FROM EVERY TIMESTEP
    # -----------------------------------------------------
    #
    # G_t = sum_{k>=t} gamma^(k-t) r_k, computed backwards.
    #
    # mean_q is Q averaged over states drawn UNIFORMLY from
    # the replay buffer, i.e. states at ALL timesteps. So the
    # quantity it estimates is the MEAN of G_t over t, not
    # G_0. Logging G_0 (as the previous version did) made Q
    # look wrong by a factor of ~0.62 on this task and forced
    # a mental correction when reading the log.
    #
    # `value_avg` is directly comparable to mean_q.
    # `discounted` (= G_0) is kept because it is the natural
    # "episode return" for reporting.

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

    # Logged next to the other behaviour metrics so the
    # analysis code finds them in the same place.
    record["token_diversity_bits"] = d_bits
    record["token_diversity_norm"] = d_norm
    behaviour_history.append(record)

    instant_metrics.reset()
    temporal_metrics.reset()
    divergence_metrics.reset()

    # =====================================================
    # PROGRESS
    # =====================================================

    if (episode + 1) % args.print_every == 0 or episode == 0:

        # Per-episode line. Columns follow the original
        # baseline's layout so runs stay comparable by eye,
        # with the diagnostics added on the right.
        #
        # NOTE: Actor is EXACTLY -Q by construction
        # (actor_loss = -q_val.mean()), so it carries no
        # information beyond the Q column. Kept only for
        # continuity with the earlier logs.

        line = (
            f"Episode {episode + 1:6d}"
            f" | Reward: {episode_reward:9.3f}"
            f" | Steps: {episode_steps:2d}"
            f" | Buffer: {len(replay_buffer):6d}"
        )

        if mean_q_values:

            # G is the discounted return-to-go averaged over
            # every timestep, over the last REWARD_WINDOW
            # episodes. This is exactly what mean_q estimates,
            # so a healthy critic has Q ~= G. Q drifting ABOVE
            # G is overestimation bias; far BELOW it is the
            # underestimation that clipped double-Q can cause.
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

        # ETA only occasionally, so it does not clutter a
        # per-episode log.
        if (episode + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (episode + 1) / elapsed
            line += f" | ETA {(NUM_EPISODES - episode - 1) / rate / 3600:5.2f}h"

        print(line, flush=True)

    # =====================================================
    # PERIODIC CHECKPOINT
    # =====================================================

    # =====================================================
    # CHECKPOINTS
    # =====================================================
    #
    # Two kinds, and NEITHER overwrites a previous one:
    #
    #   ckpt_ep<N>.pt : a fixed grid, so any point on the
    #                   learning curve can be revisited.
    #   best.pt       : whichever policy had the highest
    #                   rolling-100 reward so far.
    #
    # The previous version wrote a single latest.pt and
    # overwrote it every time, so when seed 42 peaked at
    # episode ~13,000 and then degraded 7.8 points, the good
    # policy was gone by the end of the run.

    if (episode + 1) % args.checkpoint_every == 0:
        maddpg.save(
            os.path.join(
                CHECKPOINT_DIR, f"ckpt_ep{episode + 1}.pt"
            ),
            metadata=CHECKPOINT_METADATA,
        )

    # Track the best policy. Burn-in guards against an early
    # lucky window winning on a barely-trained network.
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


# =========================================================
# SAVE
# =========================================================

print("-" * 64)
print(f"Total environment steps : {global_step:,}")
print(f"Total gradient updates  : {update_count:,}")
print(f"Wall clock              : {(time.time()-start_time)/3600:.2f} h")

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

# Behaviour history saved as a named-column archive so new
# metrics can be added without breaking the column ordering
# that a bare 2-D array would depend on.
behaviour_keys = sorted(behaviour_history[0].keys())

np.savez(
    os.path.join(RESULTS_DIR, "behaviour_history.npz"),
    **{
        k: np.array([r[k] for r in behaviour_history])
        for k in behaviour_keys
    }
)

# The episode behaviour descriptor phi, one 3x3 frequency
# matrix per episode. D was computed from these; logging them
# lets the secondary robustness check H(phi) be estimated
# offline, with no rolling archive during training.
np.save(
    os.path.join(RESULTS_DIR, "token_frequencies.npy"),
    np.asarray(token_frequencies)
)

if null_values:
    print(f"Permutation null (mean over {len(null_values)} "
          f"evaluations): {float(np.mean(null_values)):.4f}")
print(f"Mean D over last 100 episodes: "
      f"{float(np.mean(episode_diversity_norm[-100:])):.4f}")

MODEL_PATH = os.path.join(CHECKPOINT_DIR, "final.pt")

# One save, not two. The previous version called
# maddpg.save() and then immediately overwrote the file with
# a second torch.save(), so the optimiser state was silently
# lost. Metadata now rides along with the network state.
maddpg.save(MODEL_PATH, metadata=CHECKPOINT_METADATA)

print(f"Best policy: episode {best_episode} "
      f"(Avg{REWARD_WINDOW} = {best_reward:.3f}) -> "
      f"{os.path.join(CHECKPOINT_DIR, 'best.pt')}")

print(f"Results -> {RESULTS_DIR}")
print(f"Model   -> {MODEL_PATH}")
print("-" * 64)

env.close()
