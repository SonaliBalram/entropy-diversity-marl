# Per-step diversity metrics: how differently agents act at the same moment.

import itertools

import numpy as np

from action_utils import to_force, safe_cosine


# Pairwise cosine and Euclidean distances between agents' actions.
class InstantaneousBehaviourMetrics:
    # Set up the agent pairs and empty stores.
    def __init__(self, n_agents=3):
        self.n_agents = n_agents

        self.pairs = list(
            itertools.combinations(range(n_agents), 2)
        )

        self.reset()

    # Record one step's actions for every agent pair.
    def record(self, actions):
        actions = np.asarray(actions, dtype=np.float64)

        if actions.shape[0] != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} agents, "
                f"received {actions.shape[0]}"
            )

        forces = to_force(actions)

        for (i, j) in self.pairs:
            key = f"{i + 1}{j + 1}"

            self.cosine_force[key].append(
                safe_cosine(forces[i], forces[j])
            )

            self.euclidean_force[key].append(
                float(np.linalg.norm(forces[i] - forces[j]))
            )

            self.cosine_raw[key].append(
                safe_cosine(actions[i], actions[j])
            )

            self.euclidean_raw[key].append(
                float(np.linalg.norm(actions[i] - actions[j]))
            )

    # Average each metric over the episode, per pair and overall.
    def get_episode_results(self):
        results = {}

        for name, store in [
            ("cosine", self.cosine_force),
            ("euclidean", self.euclidean_force),
            ("cosine_raw", self.cosine_raw),
            ("euclidean_raw", self.euclidean_raw),
        ]:
            per_pair = []

            for (i, j) in self.pairs:
                key = f"{i + 1}{j + 1}"

                value = float(
                    np.nanmean(store[key])
                ) if len(store[key]) else np.nan

                results[f"{name}_{key}"] = value

                per_pair.append(value)

            results[f"{name}_mean"] = float(
                np.nanmean(per_pair)
            )

        return results

    # Clear the stores for a new episode.
    def reset(self):
        keys = [f"{i + 1}{j + 1}" for (i, j) in self.pairs]

        self.cosine_force = {k: [] for k in keys}
        self.euclidean_force = {k: [] for k in keys}
        self.cosine_raw = {k: [] for k in keys}
        self.euclidean_raw = {k: [] for k in keys}
