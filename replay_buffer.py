# Replay buffer of past transitions for off-policy training.

import numpy as np
import torch


# Fixed-size buffer storing transitions with one team reward per step.
class ReplayBuffer:
    # Allocate the storage arrays.
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

        self.rng = np.random.default_rng(seed)

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

    # Store one transition, overwriting the oldest when full.
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

    # Sample a random batch of transitions, with replacement.
    def sample(
        self,
        batch_size,
        device="cpu"
    ):
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

    # Number of transitions currently stored.
    def __len__(self):
        return self.size
