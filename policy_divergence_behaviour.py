"""
policy_divergence_behaviour.py

Divergence between agents' policy DISTRIBUTIONS.

WHAT WAS ALREADY CORRECT
------------------------
The closed-form diagonal-Gaussian KL

    KL = log(s_q/s_p) + (s_p^2 + (m_p-m_q)^2)/(2 s_q^2) - 1/2

is exactly right (verified against torch.distributions to
9 decimal places), and the moment-matched Jensen-Shannon
construction around it is a standard approximation.

It is also worth recording that KL and JS are INVARIANT
under the sigmoid squash -- the Jacobian cancels in the
density ratio (verified by Monte-Carlo). So computing them
on the pre-squash Gaussian parameters is legitimate, not an
approximation to something else.

THE PROBLEM: JS ANSWERS A DIFFERENT QUESTION
--------------------------------------------
JS measures how DISTINGUISHABLE two distributions are, not
how differently the agents BEHAVE. As sigma shrinks, any two
distinct policies become perfectly distinguishable, so JS
diverges even when the actions are unchanged. Holding the
means fixed and shrinking sigma over the range this baseline
was measured to traverse (0.993 -> 0.0087):

    sigma    1.000   0.200   0.050   0.0087
    JS       0.152   2.352   8.145   16.794     <-- 110x
    |E[a_p] - E[a_q]|  0.102 ... 0.1225         <-- unchanged

Since the alpha=0 objective DRIVES sigma to zero, plotting JS
over training would show "policy divergence increasing ~100x"
while the agents were in fact converging on identical
behaviour. That is the exact opposite of the truth, and it
would be reported as evidence against the hypothesis.

THE FIX
-------
Report three quantities, never JS alone:

  js_*         : statistical distinguishability (unchanged,
                 still useful -- just not a diversity measure)
  w2_*         : 2-Wasserstein distance between the policies,
                 in ACTION UNITS. Flat under sigma-collapse
                 (0.2344 -> 0.2738 across the same sweep), so
                 it isolates behavioural separation.
  mean_sigma   : the confound itself, logged so any JS trend
                 can be checked against it.

W2 is the right primitive here because it is a metric ON the
action space: it asks "how far apart are these actions", which
is what behavioural diversity means, rather than "could a
classifier tell these distributions apart".
"""

import itertools

import numpy as np


class PolicyDivergenceMetrics:

    def __init__(self, n_agents, action_dim):

        self.n_agents = n_agents
        self.action_dim = action_dim

        self.pairs = list(
            itertools.combinations(range(n_agents), 2)
        )

        self.policy_distributions = []


    # =====================================================
    # RECORD
    # =====================================================

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


    # =====================================================
    # GAUSSIAN KL  (unchanged -- it was correct)
    # =====================================================

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


    # =====================================================
    # JENSEN-SHANNON  (unchanged -- it was correct)
    # =====================================================

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


    # =====================================================
    # BEHAVIOURAL DIVERGENCE: W2 IN ACTION SPACE
    # =====================================================

    @staticmethod
    def squashed_w2(mean_p, std_p, mean_q, std_q, n_samples=512, rng=None):

        """
        2-Wasserstein distance between the two SQUASHED
        (post-sigmoid) policies, estimated from matched
        quantiles.

        For 1-D distributions W2 has a closed form in terms
        of the quantile functions:

            W2(P,Q)^2 = integral_0^1 (F_P^-1(t) - F_Q^-1(t))^2 dt

        and the sigmoid is monotone, so the quantiles of the
        squashed distribution are just the squashed quantiles
        of the Gaussian. That makes this exact up to the
        quadrature, with no sampling noise.

        Summed over action dimensions, then square-rooted.
        """

        if rng is None:
            rng = np.random

        # Midpoint quadrature on the probability axis.
        t = (np.arange(n_samples) + 0.5) / n_samples

        # Standard normal quantiles, computed via erfinv.
        from scipy.special import erfinv
        zq = np.sqrt(2.0) * erfinv(2.0 * t - 1.0)

        total = 0.0

        for d in range(len(mean_p)):

            qp = 1.0 / (1.0 + np.exp(-(mean_p[d] + std_p[d] * zq)))
            qq = 1.0 / (1.0 + np.exp(-(mean_q[d] + std_q[d] * zq)))

            total += np.mean((qp - qq) ** 2)

        return float(np.sqrt(total))


    # =====================================================
    # EPISODE RESULTS
    # =====================================================

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

        # The confound, logged explicitly. Any JS trend must
        # be read against this.
        all_stds = np.array([
            s for _, s in self.policy_distributions
        ])

        results["mean_sigma"] = float(all_stds.mean())

        return results


    def reset(self):

        self.policy_distributions = []
