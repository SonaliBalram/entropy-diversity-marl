# Trajectory-level diversity: how correlated agents' actions are over an episode.

import itertools

import numpy as np

from action_utils import to_force


# Correlation between agents' action time series within an episode.
class TemporalBehaviourMetrics:
    # Set up the agent pairs.
    def __init__(self, n_agents, action_dim=5, use_force=True):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.use_force = use_force

        self.pairs = list(
            itertools.combinations(range(n_agents), 2)
        )

        self.actions = []

    # Store one step's actions.
    def record(self, actions):
        actions = np.asarray(actions, dtype=np.float64)

        if actions.shape != (self.n_agents, self.action_dim):
            raise ValueError(
                f"Expected actions with shape "
                f"({self.n_agents}, {self.action_dim}), "
                f"but received {actions.shape}"
            )

        self.actions.append(actions.copy())

    # Pairwise correlations over the whole episode.
    def get_episode_results(self):
        if len(self.actions) < 3:
            raise ValueError(
                "At least three timesteps are required "
                "for a meaningful correlation."
            )

        series = np.array(self.actions)

        if self.use_force:
            series = to_force(series)

        n_channels = series.shape[-1]

        results = {}
        per_pair = []

        for (i, j) in self.pairs:
            channel_correlations = []

            for d in range(n_channels):
                x = series[:, i, d]
                y = series[:, j, d]

                if np.std(x) < 1e-8 or np.std(y) < 1e-8:
                    channel_correlations.append(np.nan)
                else:
                    channel_correlations.append(
                        float(np.corrcoef(x, y)[0, 1])
                    )

            value = float(np.nanmean(channel_correlations)) \
                if not np.all(np.isnan(channel_correlations)) \
                else np.nan

            results[f"correlation_{i + 1}{j + 1}"] = value
            per_pair.append(value)

        results["correlation_mean"] = float(
            np.nanmean(per_pair)
        )

        results["degenerate_fraction"] = float(
            np.mean([
                np.std(series[:, i, d]) < 1e-8
                for (i, j) in self.pairs
                for d in range(n_channels)
            ])
        )

        return results

    # Clear the stored actions for a new episode.
    def reset(self):
        self.actions = []
