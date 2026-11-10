# Policy-level diversity: how far apart agents' action distributions are.

import itertools

import numpy as np


# KL, JS and Wasserstein distances between agents' Gaussian policies.
class PolicyDivergenceMetrics:
    # Set up the agent pairs.
    def __init__(self, n_agents, action_dim):
        self.n_agents = n_agents
        self.action_dim = action_dim

        self.pairs = list(
            itertools.combinations(range(n_agents), 2)
        )

        self.policy_distributions = []

    # Store one step's policy means and standard deviations.
    def record(self, means, stds):
        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)

        expected = (self.n_agents, self.action_dim)

        if means.shape != expected:
            raise ValueError(
                f"Expected means with shape {expected}, "
                f"received {means.shape}"
            )

        if stds.shape != expected:
            raise ValueError(
                f"Expected stds with shape {expected}, "
                f"received {stds.shape}"
            )

        self.policy_distributions.append(
            (means.copy(), stds.copy())
        )

    # KL divergence between two Gaussians.
    @staticmethod
    def gaussian_kl(mean_p, std_p, mean_q, std_q):
        variance_p = std_p ** 2
        variance_q = std_q ** 2

        kl = (
            np.log(std_q / std_p)
            + (variance_p + (mean_p - mean_q) ** 2)
            / (2.0 * variance_q)
            - 0.5
        )

        return float(np.sum(kl))

    # Jensen-Shannon divergence between two Gaussians.
    @staticmethod
    def gaussian_js(mean_p, std_p, mean_q, std_q):
        mean_m = (mean_p + mean_q) / 2.0

        variance_m = (
            0.5 * (std_p ** 2 + (mean_p - mean_m) ** 2)
            + 0.5 * (std_q ** 2 + (mean_q - mean_m) ** 2)
        )

        std_m = np.sqrt(variance_m)

        return 0.5 * (
            PolicyDivergenceMetrics.gaussian_kl(
                mean_p, std_p, mean_m, std_m
            )
            + PolicyDivergenceMetrics.gaussian_kl(
                mean_q, std_q, mean_m, std_m
            )
        )

    # 2-Wasserstein distance between the two squashed (post-sigmoid) policies.
    @staticmethod
    def squashed_w2(mean_p, std_p, mean_q, std_q, n_samples=512, rng=None):
        if rng is None:
            rng = np.random

        t = (np.arange(n_samples) + 0.5) / n_samples

        from scipy.special import erfinv
        zq = np.sqrt(2.0) * erfinv(2.0 * t - 1.0)

        total = 0.0

        for d in range(len(mean_p)):
            qp = 1.0 / (1.0 + np.exp(-(mean_p[d] + std_p[d] * zq)))
            qq = 1.0 / (1.0 + np.exp(-(mean_q[d] + std_q[d] * zq)))

            total += np.mean((qp - qq) ** 2)

        return float(np.sqrt(total))

    # Average each divergence over the episode, per pair and overall.
    def get_episode_results(self):
        if len(self.policy_distributions) == 0:
            raise ValueError(
                "No policy distributions recorded."
            )

        results = {}

        js_all = []
        w2_all = []

        for (i, j) in self.pairs:
            key = f"{i + 1}{j + 1}"

            js_values = []
            w2_values = []

            for means, stds in self.policy_distributions:
                js_values.append(
                    self.gaussian_js(
                        means[i], stds[i], means[j], stds[j]
                    )
                )

                w2_values.append(
                    self.squashed_w2(
                        means[i], stds[i], means[j], stds[j]
                    )
                )

            results[f"js_{key}"] = float(np.mean(js_values))
            results[f"w2_{key}"] = float(np.mean(w2_values))

            js_all.append(results[f"js_{key}"])
            w2_all.append(results[f"w2_{key}"])

        results["js_mean"] = float(np.mean(js_all))
        results["w2_mean"] = float(np.mean(w2_all))

        all_stds = np.array([
            s for _, s in self.policy_distributions
        ])

        results["mean_sigma"] = float(all_stds.mean())

        return results

    # Clear the stored policies for a new episode.
    def reset(self):
        self.policy_distributions = []
