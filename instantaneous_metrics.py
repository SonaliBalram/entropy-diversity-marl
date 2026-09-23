"""
instantaneous_metrics.py

Timestep-level behavioural similarity between agents.

CHANGES FROM THE ORIGINAL
-------------------------
1. Metrics are computed on the PHYSICAL FORCE u = (u_x, u_y),
   not the raw 5-D action. See action_utils.to_force for the
   full argument -- in short, raw-action cosine similarity
   cannot represent opposition and scores ~0.77 for
   uniformly random agents.
2. Degenerate cases return NaN and are excluded via nanmean,
   rather than being recorded as 0.0.
3. Generalised to any n_agents (the original hard-required 3).
4. Both raw-action and force-based cosine are reported, so
   the original numbers remain available for comparison and
   the difference can be shown in the write-up.
"""

import itertools

import numpy as np

from action_utils import to_force, safe_cosine


class InstantaneousBehaviourMetrics:

    def __init__(self, n_agents=3):

        self.n_agents = n_agents

        self.pairs = list(
            itertools.combinations(range(n_agents), 2)
        )

        self.reset()


    # =====================================================
    # RECORD ONE TIMESTEP
    # =====================================================

    def record(self, actions):
        """
        actions : (n_agents, 5) raw MPE actions
        """

        actions = np.asarray(actions, dtype=np.float64)

        if actions.shape[0] != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} agents, "
                f"received {actions.shape[0]}"
            )

        forces = to_force(actions)

        for (i, j) in self.pairs:

            key = f"{i + 1}{j + 1}"

            # --- primary: physical force space ---
            self.cosine_force[key].append(
                safe_cosine(forces[i], forces[j])
            )

            self.euclidean_force[key].append(
                float(np.linalg.norm(forces[i] - forces[j]))
            )

            # --- retained for comparison with the original ---
            self.cosine_raw[key].append(
                safe_cosine(actions[i], actions[j])
            )

            self.euclidean_raw[key].append(
                float(np.linalg.norm(actions[i] - actions[j]))
            )


    # =====================================================
    # EPISODE RESULTS
    # =====================================================

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


    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        keys = [f"{i + 1}{j + 1}" for (i, j) in self.pairs]

        self.cosine_force = {k: [] for k in keys}
        self.euclidean_force = {k: [] for k in keys}
        self.cosine_raw = {k: [] for k in keys}
        self.euclidean_raw = {k: [] for k in keys}
