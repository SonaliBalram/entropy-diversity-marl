"""
maddpg_system.py

Baseline stochastic MADDPG under CTDE.

Architecture:
    - ONE shared stochastic Gaussian actor (all agents)
    - ONE centralized critic over the joint (o, a)
    - Target networks for actor and critic
    - Polyak target updates
    - Team-level reward
    - NO entropy regularization  (alpha = 0 implicitly:
      the actor objective is exactly -E[Q])

=========================================================
DESIGN DECISIONS TO DOCUMENT IN THE WRITE-UP
=========================================================

(a) ONE critic, not N critics. Vanilla MADDPG gives each
    agent its own centralized critic Q_i, because rewards
    may differ per agent. Here the reward is a single team
    reward and the actor is shared, so Q_1 = ... = Q_N in
    expectation and a single critic is the correct
    reduction -- not a simplification. Say so explicitly,
    otherwise a reader will read it as an error.

(b) The actor update resamples ALL N agents' actions from
    the current policy, and differentiates
        Q(s, pi(o_1), ..., pi(o_N))
    w.r.t. the shared parameters. Vanilla MADDPG replaces
    only agent i's action and takes the others from the
    buffer. With a shared actor and a team reward, the
    joint form is the correct total gradient
        dQ/dtheta = sum_i (dQ/da_i)(da_i/dtheta)
    and is what this file implements.

(c) The Bellman target uses a SINGLE stochastic sample
    a' ~ pi_target(.|s'). This is unbiased for
    E_{a'~pi}[Q(s',a')] but has variance. Set
    `n_target_samples > 1` to average K samples if the
    critic loss is too noisy.

(d) Optimisation-level stability only (asymmetric learning
    rates, gradient clipping, Huber critic loss). None of
    these add an exploration bonus or change the objective,
    so Experiment 1 remains a valid zero-entropy baseline.

(e) CLIPPED DOUBLE-Q. Plain single-critic MADDPG was measured
    to diverge on this task: over episodes 4000-6500 the
    critic's estimate drifted from -12.1 to -0.03 while the
    true discounted return sat at -22.8, critic loss rose
    0.51 -> 1.85, and Avg100 reward degraded from -31.5 to
    -38.9.

    Mechanism: the actor maximises Q, so it actively seeks
    actions where the critic is WRONGLY OPTIMISTIC. Those
    inflated values are bootstrapped into the next target and
    compound. Nothing in single-critic DDPG pushes back.

    Fix: train two independent centralised critics and use
    min(Q1, Q2) in the Bellman target. The minimum of two
    noisy estimates is biased DOWNWARD, which cancels the
    upward drift. This is TD3's central idea, and the
    multi-agent version is Ackermann et al. (2019),
    "Reducing Overestimation Bias in Multi-Agent Domains
    Using Double Centralized Critics" -- cite that rather
    than TD3 alone.

    Also included is TD3's delayed policy update
    (policy_delay=2): the actor and the target networks are
    updated once per `policy_delay` critic updates, so the
    actor always differentiates through a critic that has had
    time to catch up.

    NOTE FOR THE WRITE-UP: this changes the baseline's name.
    It is no longer plain MADDPG -- describe it as MADDPG
    with clipped double-Q centralised critics (MATD3-style).
    It must be applied identically in Experiments 2-5, or the
    comparison is confounded.
"""

import copy

import torch
import torch.nn.functional as F

from models.actor import SharedContinuousActor
from models.critic import CentralizedCritic


class MADDPGSystem:

    def __init__(
        self,
        n_agents,
        observation_dim,
        action_dim,
        hidden_dim=128,
        gamma=0.95,
        tau=0.005,
        lr_actor=1e-4,
        lr_critic=1e-3,
        grad_clip_norm=10.0,
        huber_delta=10.0,
        n_target_samples=1,
        policy_delay=2,
        twin_critics=True,
        alpha=0.0,
        device="cpu",
        context_dim=0,
    ):

        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.tau = tau

        # Max gradient norm. Raised 1.0 -> 10.0: with a
        # 128-unit 3-layer net the natural gradient norm is
        # routinely above 1, so a clip of 1.0 fires on nearly
        # every update and turns Adam into a globally rescaled
        # sign-descent. update() now logs the PRE-clip norms
        # so this can be set from data (aim for the ~90th
        # percentile of the observed norms).
        # Set to None to disable clipping entirely.
        self.grad_clip_norm = grad_clip_norm

        # Huber transition point. Below delta the loss is
        # quadratic (= MSE), above it linear. delta=1.0 is
        # too small here: early TD errors are O(1-10) because
        # returns are O(-40), so the loss sits in its linear
        # regime and the critic effectively learns from the
        # SIGN of the TD error only. delta=10 keeps outlier
        # protection while leaving normal errors quadratic.
        self.huber_delta = huber_delta

        # Number of target-action samples to average (see (c)).
        self.n_target_samples = n_target_samples

        # TD3 delayed policy update. The actor and all target
        # networks update once per `policy_delay` critic
        # updates. Set to 1 to disable.
        self.policy_delay = policy_delay
        # Start so that the FIRST update also updates the
        # actor, otherwise the first logged line shows nan.
        self._update_index = policy_delay - 1

        # Set False to recover the single-critic behaviour
        # that was measured to diverge. Kept as a switch so
        # the divergence can be reproduced for the write-up.
        self.twin_critics = twin_critics

        # =================================================
        # CRITIC CONTEXT WIDTH  (Experiment 3 augmented arm)
        # =================================================
        #
        # 0 -> the critic sees (o_1..o_N, a_1..a_N), which is
        # what Experiments 1, 2 and the unaugmented Exp 3 arm
        # use. The critic input width, the layer shapes and
        # hence the parameter-initialisation RNG stream are
        # all unchanged at 0.
        #
        # 9 -> the critic additionally sees the 3x3 running
        # token-count matrix. This is CTDE-legal: it is extra
        # information at TRAINING time only, given to the
        # centralised critic. The actor's input is untouched,
        # so execution stays decentralised and the policy class
        # is identical to every other arm.
        self.context_dim = context_dim

        # =================================================
        # ENTROPY COEFFICIENT
        # =================================================
        #
        # alpha = 0.0 is Experiment 1: the actor maximises
        # E[Q(s,a)] and nothing else.
        #
        # alpha > 0 adds the policy-entropy bonus used by
        # Experiments 2-5:
        #
        #     J = E[ Q(s,a) + alpha * H(pi(.|o)) ]
        #
        # with H estimated from the same reparameterised
        # sample already drawn for the Q term
        # (H ~= -log pi(a|o)), which is exactly the SAC actor
        # objective.
        #
        # The alpha term is applied ONLY when alpha != 0, so
        # an alpha=0 run is bit-identical to the version that
        # produced the Experiment 1 numbers -- adding
        # `0.0 * entropy` would differ if entropy ever went
        # non-finite.
        self.alpha = alpha

        # Carried between updates so the stats dict keeps a
        # stable set of keys on steps where the actor is
        # skipped.
        self._last_actor_stats = {
            "actor_loss": float("nan"),
            "mean_q": float("nan"),
            "actor_grad_norm": float("nan"),
        }

        self.device = device

        # =================================================
        # SHARED ACTOR + TARGET
        # =================================================

        self.actor = SharedContinuousActor(
            observation_dim,
            action_dim,
            hidden_dim
        ).to(device)

        self.actor_target = copy.deepcopy(self.actor).to(device)

        # =================================================
        # CENTRALIZED CRITIC + TARGET
        # =================================================

        def make_critic():
            return CentralizedCritic(
                n_agents,
                observation_dim,
                action_dim,
                hidden_dim,
                context_dim=context_dim
            ).to(device)

        # Two INDEPENDENTLY initialised critics. The
        # independence matters: if both started from the same
        # weights they would make correlated errors and the
        # minimum would not correct anything.
        self.critic = make_critic()
        self.critic_2 = make_critic() if twin_critics else None

        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_target_2 = (
            copy.deepcopy(self.critic_2).to(device)
            if twin_critics else None
        )

        # =================================================
        # FREEZE TARGET NETWORKS
        # =================================================

        for p in self.actor_target.parameters():
            p.requires_grad = False

        for p in self.critic_target.parameters():
            p.requires_grad = False

        if self.twin_critics:
            for p in self.critic_target_2.parameters():
                p.requires_grad = False

        # =================================================
        # OPTIMIZERS
        # =================================================

        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr_actor
        )

        critic_params = list(self.critic.parameters())

        if self.twin_critics:
            critic_params += list(self.critic_2.parameters())

        self.critic_opt = torch.optim.Adam(
            critic_params,
            lr=lr_critic
        )

        self._critic_params = critic_params


    # =====================================================
    # ACTION SELECTION
    # =====================================================

    def act(self, obs_n, deterministic=False):

        obs_t = torch.as_tensor(
            obs_n,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():

            if deterministic:
                # NOTE: sigmoid(mu) is the MEDIAN of the
                # squashed policy, not its mean. Standard
                # practice (SAC does the same) but worth
                # stating in the methodology section.
                actions = self.actor.deterministic_action(obs_t)

            else:
                actions, _, _ = self.actor.sample_action(obs_t)

        return actions.cpu().numpy()


    def policy_params(self, obs_n):

        """
        Per-agent Gaussian policy parameters at the current
        observations, as numpy arrays.

        Returns (means, stds), each [n_agents, action_dim].
        Feeds PolicyDivergenceMetrics.record().
        """

        obs_t = torch.as_tensor(
            obs_n,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            d = self.actor.get_distribution(obs_t)

        return (
            d.mean.cpu().numpy(),
            d.stddev.cpu().numpy()
        )


    # =====================================================
    # POLYAK TARGET UPDATE
    # =====================================================

    def _polyak_update(self, net, target_net):

        with torch.no_grad():

            for p, tp in zip(
                net.parameters(),
                target_net.parameters()
            ):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * p.data)

            # Buffers too (no-op for the current architecture,
            # but required the moment LayerNorm/BatchNorm is
            # added).
            for b, tb in zip(
                net.buffers(),
                target_net.buffers()
            ):
                tb.data.copy_(b.data)


    # =====================================================
    # TRAINING UPDATE
    # =====================================================

    def update(self, batch):

        """
        batch:
            obs      : [B, N, observation_dim]
            acts     : [B, N, action_dim]
            team_rew : [B, 1]
            next_obs : [B, N, observation_dim]
            done     : [B, 1]   <-- TERMINATED ONLY.
                                    Never `terminated or
                                    truncated`. See
                                    ReplayBuffer.add().
        """

        # The buffer returns a 5-tuple when unaugmented and a
        # 7-tuple when augmented. Unpacking on self.context_dim
        # rather than on len(batch) means a buffer/system
        # mismatch raises here instead of training quietly on
        # the wrong object.
        if self.context_dim > 0:
            (obs, acts, team_rew, next_obs, done,
             ctx, next_ctx) = batch
        else:
            obs, acts, team_rew, next_obs, done = batch
            ctx = next_ctx = None

        B = obs.shape[0]
        N = self.n_agents

        # =================================================
        # 1. CRITIC TARGET
        # =================================================

        with torch.no_grad():

            next_obs_flat = next_obs.reshape(
                B * N,
                self.observation_dim
            )

            q_next = 0.0

            for _ in range(self.n_target_samples):

                next_acts_flat, _, _ = (
                    self.actor_target.sample_action(
                        next_obs_flat
                    )
                )

                next_acts = next_acts_flat.reshape(
                    B, N, self.action_dim
                )

                q1 = self.critic_target(
                    next_obs, next_acts, next_ctx
                )

                if self.twin_critics:

                    q2 = self.critic_target_2(
                        next_obs, next_acts, next_ctx
                    )

                    # CLIPPED DOUBLE-Q.
                    #
                    # Both critics estimate the same quantity
                    # with independent errors. Taking the
                    # elementwise minimum yields an estimate
                    # biased DOWNWARD, which is what cancels
                    # the actor-driven upward drift.
                    #
                    # It is a deliberate pessimism: the
                    # resulting Q slightly UNDER-estimates the
                    # true return. That is the trade being
                    # made -- a small controllable bias in
                    # exchange for not diverging.
                    q_next = q_next + torch.min(q1, q2)

                else:
                    q_next = q_next + q1

            q_next = q_next / self.n_target_samples

            y = (
                team_rew
                + self.gamma
                * (1.0 - done)
                * q_next
            )

        # =================================================
        # 2. CURRENT CRITIC
        # =================================================

        q = self.critic(obs, acts, ctx)

        td_error = (y - q).detach()

        # =================================================
        # 3. CRITIC LOSS (Huber)
        # =================================================
        #
        # BOTH critics are regressed onto the SAME target y.
        # They stay different only because they started from
        # different random initialisations and see different
        # gradient noise -- that residual disagreement is
        # exactly what the min() exploits.

        critic_loss = F.huber_loss(q, y, delta=self.huber_delta)

        if self.twin_critics:

            q_b = self.critic_2(obs, acts, ctx)

            critic_loss = critic_loss + F.huber_loss(
                q_b, y, delta=self.huber_delta
            )

        # =================================================
        # 4. UPDATE CRITIC
        # =================================================

        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward()

        # clip_grad_norm_ RETURNS the pre-clip total norm --
        # log it so grad_clip_norm can be chosen from data
        # rather than guessed.
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._critic_params,
            self.grad_clip_norm
            if self.grad_clip_norm is not None
            else float("inf")
        )

        self.critic_opt.step()

        self._update_index += 1

        # TD3 delayed policy update: the actor and the target
        # networks move once every `policy_delay` critic
        # updates, so the actor never chases a critic that has
        # just moved under it.
        do_actor_update = (
            self._update_index % self.policy_delay == 0
        )

        # =================================================
        # 5. ACTOR UPDATE
        # =================================================
        #
        # Freeze the critic while differentiating through it.
        # Functionally this changed nothing before (the stale
        # grads were zeroed on the next call), but it makes
        # the intent explicit and saves the backward pass
        # through the critic's parameters.

        obs_flat = obs.reshape(
            B * N,
            self.observation_dim
        )

        if do_actor_update:

            for p in self.critic.parameters():
                p.requires_grad = False

            acts_flat, log_probs, entropy_samples = (
                self.actor.sample_action(obs_flat)
            )

            acts_resampled = acts_flat.reshape(
                B, N, self.action_dim
            )

            # The actor differentiates through critic 1 ONLY.
            # Using min(Q1,Q2) here would make the policy
            # gradient depend on whichever critic happens to
            # be lower at each sample, which is noisier and is
            # not what TD3 does.
            q_val = self.critic(obs, acts_resampled, ctx)

        # =================================================
        # BASELINE ACTOR OBJECTIVE  (alpha = 0)
        # =================================================
        #
        #   J(theta) = E[ Q(s, pi_theta(o)) ]
        #   loss     = -J
        #
        # No entropy term. Experiment 2 adds
        #   loss = -(q_val.mean() + alpha * entropy.mean())
        # here and nowhere else.

            if self.alpha != 0.0:
                # Experiments 2-5. entropy_samples here CARRIES
                # GRADIENT (it comes from the rsample above),
                # so d/dtheta reaches both mu and sigma. Do not
                # substitute the no_grad copy computed in the
                # statistics block below.
                actor_loss = -(
                    q_val.mean()
                    + self.alpha * entropy_samples.mean()
                )
            else:
                actor_loss = -q_val.mean()

            self.actor_opt.zero_grad(set_to_none=True)

            actor_loss.backward()

            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.grad_clip_norm
                if self.grad_clip_norm is not None
                else float("inf")
            )

            self.actor_opt.step()

            for p in self.critic.parameters():
                p.requires_grad = True

            self._last_actor_stats = {
                "actor_loss": actor_loss.item(),
                "mean_q": q_val.mean().item(),
                "actor_grad_norm": float(actor_grad_norm),
            }

            # =============================================
            # 6. UPDATE TARGET NETWORKS
            # =============================================
            #
            # Targets move on the SAME delayed schedule as the
            # actor. Updating them every critic step while
            # delaying the actor would defeat the purpose.

            self._polyak_update(self.actor, self.actor_target)
            self._polyak_update(self.critic, self.critic_target)

            if self.twin_critics:
                self._polyak_update(
                    self.critic_2, self.critic_target_2
                )

        # =================================================
        # 7. TRAINING + DIVERSITY STATISTICS
        # =================================================
        #
        # The entropy / std / spread terms are NOT used in the
        # loss (alpha = 0). They are logged because the
        # research question is precisely whether a stochastic
        # shared policy homogenizes: `policy_std` measures
        # action-level noise, `mean_spread` measures how
        # differently the shared policy responds to different
        # agents' observations. Those two coming apart is the
        # finding.

        with torch.no_grad():

            distribution = self.actor.get_distribution(obs_flat)

            # Named distinctly so it cannot be confused with the
            # gradient-carrying entropy_samples used in the
            # actor objective above.
            sampled, _, entropy_stat = (
                self.actor.sample_action(obs_flat)
            )

            means_per_agent = distribution.mean.reshape(
                B, N, self.action_dim
            )

            inter_agent_spread = (
                means_per_agent.std(dim=1).mean().item()
            )

            realised_spread = (
                sampled.reshape(B, N, self.action_dim)
                .std(dim=1).mean().item()
            )

            # Diagnostic that would have caught the divergence
            # immediately: how far the two critics disagree.
            # A healthy pair disagrees a little; agreement
            # collapsing to zero means the min() has stopped
            # correcting anything.
            if self.twin_critics:
                critic_disagreement = (
                    (self.critic(obs, acts, ctx)
                     - self.critic_2(obs, acts, ctx))
                    .abs().mean().item()
                )
            else:
                critic_disagreement = float("nan")

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": self._last_actor_stats["actor_loss"],
            "mean_q": self._last_actor_stats["mean_q"],
            "td_error_abs": td_error.abs().mean().item(),
            "critic_grad_norm": float(critic_grad_norm),
            "actor_grad_norm": self._last_actor_stats["actor_grad_norm"],
            "critic_disagreement": critic_disagreement,
            # --- diversity diagnostics ---
            "policy_std": distribution.stddev.mean().item(),
            "policy_entropy_mc": entropy_stat.mean().item(),
            "policy_entropy_analytic":
                distribution.entropy().sum(-1).mean().item(),
            "inter_agent_mean_spread": inter_agent_spread,
            "realised_action_spread": realised_spread,
        }


    # =====================================================
    # CHECKPOINTING
    # =====================================================

    def save(self, path, metadata=None):

        state = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "twin_critics": self.twin_critics,
            "policy_delay": self.policy_delay,
            "alpha": self.alpha,
            # Recorded so evaluation and analysis scripts can
            # rebuild the right critic shape from the file
            # alone. exp3_outcome.py reads alpha the same way.
            "context_dim": self.context_dim,
        }

        if self.twin_critics:
            state["critic_2"] = self.critic_2.state_dict()
            state["critic_target_2"] = \
                self.critic_target_2.state_dict()

        if metadata:
            state.update(metadata)

        torch.save(state, path)

    def load(self, path):

        ckpt = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])

        if self.twin_critics and "critic_2" in ckpt:
            self.critic_2.load_state_dict(ckpt["critic_2"])
            self.critic_target_2.load_state_dict(
                ckpt["critic_target_2"]
            )

        if "actor_opt" in ckpt:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
