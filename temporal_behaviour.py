"""
temporal_behaviour.py

Within-episode temporal co-movement between agents.

CHANGES FROM THE ORIGINAL
-------------------------
1. Correlations are computed on the PHYSICAL FORCE channels
   (u_x, u_y) rather than the 5 raw action components. The
   raw version correlated a[0] (which MPE ignores) and split
   each physical axis across two anti-correlated components.
2. Degenerate (constant) sequences yield NaN, not 0.0. The
   original mapped "undefined" to "uncorrelated", which is
   the WRONG DIRECTION for this research question: a
   collapsed, near-deterministic policy triggers the
   degenerate branch most often, so homogenised agents would
   have been scored as maximally independent.
3. A caveat worth stating in the write-up: with max_cycles=25
   a Pearson correlation has a standard error of roughly
   1/sqrt(T-3) ~ 0.21, so PER-EPISODE values are very noisy.
   Only the average over many episodes is interpretable.
4. NOTE the confound this metric shares with all
   correlation-based diversity measures: correlation rises
   as sampling noise falls. Three agents following an
   identical trajectory score 0.27 at sigma=0.5 but 1.00 at
   sigma=0.005 -- same strategy, different score. Always
   report policy sigma alongside it.
"""

import itertools

import numpy as np

from action_utils import to_force


class TemporalBehaviourMetrics:

    def __init__(self, n_agents, action_dim=5, use_force=True):

        self.n_agents = n_agents
        self.action_dim = action_dim
        self.use_force = use_force

        self.pairs = list(
            itertools.combinations(range(n_agents), 2)
        )

        self.actions = []


    def record(self, actions):

        actions = np.asarray(actions, dtype=np.float64)

        if actions.shape != (self.n_agents, self.action_dim):
            raise ValueError(
                f"Expected actions with shape "
                f"({self.n_agents}, {self.action_dim}), "
                f"but received {actions.shape}"
            )

        self.actions.append(actions.copy())


    def get_episode_results(self):

        if len(self.actions) < 3:
            raise ValueError(
                "At least three timesteps are required "
                "for a meaningful correlation."
            )

        series = np.array(self.actions)          # [T, N, 5]

        if self.use_force:
            series = to_force(series)            # [T, N, 2]

        n_channels = series.shape[-1]

        results = {}
        per_pair = []

        for (i, j) in self.pairs:

            channel_correlations = []

            for d in range(n_channels):

                x = series[:, i, d]
                y = series[:, j, d]

                if np.std(x) < 1e-8 or np.std(y) < 1e-8:
                    # Undefined, NOT zero. Excluded from the
                    # mean rather than scored as independent.
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

        # Fraction of channel-pairs that were degenerate.
        # If this climbs over training, the policy is
        # collapsing and the correlation numbers should be
        # read with that in mind.
        results["degenerate_fraction"] = float(
            np.mean([
                np.std(series[:, i, d]) < 1e-8
                for (i, j) in self.pairs
                for d in range(n_channels)
            ])
        )

        return results


    def reset(self):

        self.actions = []
