import numpy as np
import torch


class ReplayBuffer:
    """
    Uniform replay buffer for a shared-actor / centralized-critic
    system with a single TEAM reward.

    Stores `terminated` (true task completion) in `self.dones`,
    NOT `terminated or truncated`. See note in `add()`.
    """

    def __init__(
        self,
        capacity,
        n_agents,
        observation_dim,
        action_dim,
        seed=None,
        context_dim=0
    ):

        self.capacity = capacity
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.context_dim = context_dim

        self.position = 0
        self.size = 0

        # Dedicated RNG so replay sampling is reproducible
        # and independent of any other np.random consumer.
        self.rng = np.random.default_rng(seed)

        # =================================================
        # STORAGE
        # =================================================

        self.observations = np.zeros(
            (capacity, n_agents, observation_dim),
            dtype=np.float32
        )

        self.actions = np.zeros(
            (capacity, n_agents, action_dim),
            dtype=np.float32
        )

        self.rewards = np.zeros(
            (capacity, 1),
            dtype=np.float32
        )

        self.next_observations = np.zeros(
            (capacity, n_agents, observation_dim),
            dtype=np.float32
        )

        self.dones = np.zeros(
            (capacity, 1),
            dtype=np.float32
        )

        # =================================================
        # OPTIONAL CONTEXT (Experiment 3 augmented arm)
        # =================================================
        #
        # Two vectors per transition, not one. The Bellman
        # target needs the context of s' as well as of s:
        #
        #   y = r + gamma * (1 - done) * Q_target(s', a', c')
        #
        # Storing only c would force the target critic to be
        # evaluated with the wrong context, which is a subtler
        # bug than it sounds -- the run would train and the
        # curves would look ordinary.
        #
        # Nothing is allocated when context_dim = 0, so an
        # unaugmented buffer is byte-for-byte what it was.

        if context_dim > 0:
            self.contexts = np.zeros(
                (capacity, context_dim),
                dtype=np.float32
            )
            self.next_contexts = np.zeros(
                (capacity, context_dim),
                dtype=np.float32
            )
        else:
            self.contexts = None
            self.next_contexts = None


    # =====================================================
    # ADD ONE EXPERIENCE
    # =====================================================

    def add(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminated,
        context=None,
        next_context=None
    ):
        """
        `terminated` must be the TRUE termination flag, not
        `terminated or truncated`.

        Reason: the Bellman target is

            y = r + gamma * (1 - done) * Q(s', a')

        Setting done=1 because a time limit was reached tells
        the critic that the world genuinely ends there, so it
        learns V(s_last) = r_last instead of the true value.
        In simple_spread there is NO terminal state, so this
        flag is always 0.0 -- and that is correct.
        """

        self.observations[self.position] = observation
        self.actions[self.position] = action

        self.rewards[self.position] = np.asarray(
            reward,
            dtype=np.float32
        ).reshape(1)

        self.next_observations[self.position] = next_observation

        self.dones[self.position] = np.asarray(
            terminated,
            dtype=np.float32
        ).reshape(1)

        if self.context_dim > 0:
            if context is None or next_context is None:
                raise ValueError(
                    "buffer was built with context_dim="
                    f"{self.context_dim} but add() got None"
                )
            self.contexts[self.position] = np.asarray(
                context, dtype=np.float32
            ).reshape(-1)
            self.next_contexts[self.position] = np.asarray(
                next_context, dtype=np.float32
            ).reshape(-1)

        self.position = (
            self.position + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity
        )


    # =====================================================
    # SAMPLE MINI-BATCH
    # =====================================================

    def sample(
        self,
        batch_size,
        device="cpu"
    ):
        """
        Sampling is WITH replacement.

        np.random.choice(n, k, replace=False) builds a full
        permutation of n internally: at capacity 1e6 that is
        ~27 ms per mini-batch versus ~0.01 ms for integer
        sampling -- a ~2000x slowdown that would dominate
        total training time. Sampling with replacement is also
        the standard choice (DQN, DDPG, SAC, MADDPG all do
        this); duplicate transitions within a batch of 256
        drawn from >1e4 stored transitions are rare and
        harmless.
        """

        if self.size < batch_size:
            raise ValueError(
                "Not enough experiences in replay buffer."
            )

        indices = self.rng.integers(
            0,
            self.size,
            size=batch_size
        )

        observations = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_observations = self.next_observations[indices]
        dones = self.dones[indices]

        # =================================================
        # NUMPY -> PYTORCH
        # =================================================

        to_t = lambda x: torch.as_tensor(
            x,
            dtype=torch.float32,
            device=device
        )

        if self.context_dim > 0:
            return (
                to_t(observations),
                to_t(actions),
                to_t(rewards),
                to_t(next_observations),
                to_t(dones),
                to_t(self.contexts[indices]),
                to_t(self.next_contexts[indices])
            )

        return (
            to_t(observations),
            to_t(actions),
            to_t(rewards),
            to_t(next_observations),
            to_t(dones)
        )


    # =====================================================
    # CURRENT NUMBER OF EXPERIENCES
    # =====================================================

    def __len__(self):

        return self.size
