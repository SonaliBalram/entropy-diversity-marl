"""
maddpg_system_a1r.py  --  A1r ONLY.  Do not import this from any other arm.

A COPY of ../maddpg_system.py, taken when that file had

    SHA256 edd298702ee2cd7c613d8408ca3d7cd26074a86dc06de24c5540a0ce1386122c

carrying exactly ONE functional change: the actor loss can take a
per-(sample, agent) entropy coefficient instead of the scalar alpha.
Experiments 1, 2 and 3 keep using ../maddpg_system.py and are not
touched by anything in this folder.

WHAT CHANGED, in full
---------------------
1.  __init__ takes a new keyword `alpha_fn=None` -- a callable mapping an
    observation batch [B, N, obs_dim] to alpha_eff of shape [B, N].  In
    practice this is A1r.voronoi_gate.OwnershipGate.

2.  The actor update gains a THIRD branch, placed FIRST:

        if self.alpha_fn is not None:        <- A1r
        elif self.alpha != 0.0:              <- unchanged, Exp 2/3
        else:                                <- unchanged, Exp 1

    The two original branches are byte-identical to the parent file.
    That matters: `(alpha * H).mean()` and `alpha * H.mean()` agree in
    exact arithmetic but NOT in floating point, so routing A1 through the
    new branch would break bit-identity with Exp 2 for no reason.  With
    alpha_fn=None this file reproduces the parent exactly, which is what
    reduction check R1 verifies.

3.  alpha_eff is detached before use.  It is a function of observations
    only, so no gradient reaches it in any case; the detach makes that
    explicit and stops a graph being built.  It blocks the DIRECT path --
    the policy cannot differentiate through its own temperature.  It does
    not block the indirect path, since the policy still chooses which
    states it visits; what closes that hole is the geometric-mean
    normalisation inside the gate, not the detach.

4.  Gate diagnostics are added to the dict update() returns, and the gate
    configuration is written into every checkpoint by save().

NOT CHANGED: the critic update, the target computation, the replay
buffer, the entropy estimator (still the MC estimate from rsample, still
missing the sigmoid Jacobian, exactly as A1 has it), the observations
used (CURRENT, not next), or anything else.
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
        alpha_fn=None,
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

        # A1r: callable obs[B, N, obs_dim] -> alpha_eff[B, N].
        # None reproduces the parent file exactly.
        self.alpha_fn = alpha_fn
        self._last_gate_stats = {}

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

            if self.alpha_fn is not None:
                # ---------------- A1r ----------------
                #
                #   L = -E[Q] - E[ stopgrad(alpha_eff,i) H(pi_i) ]
                #
                # alpha_eff comes from the frozen ownership gate and is
                # a function of the CURRENT observations only.  obs is
                # [B, N, obs_dim] and obs.reshape(B*N, ...) is
                # row-major, so index b*N + i; alpha_eff.reshape(B*N)
                # uses the same order and lines up with
                # entropy_samples element for element.
                #
                # entropy_samples still CARRIES GRADIENT (it comes from
                # the rsample above) so d/dtheta reaches both mu and
                # sigma, exactly as in the scalar-alpha branch.  Only
                # the coefficient is detached.
                alpha_eff = self.alpha_fn(obs).detach().reshape(B * N)

                actor_loss = -(
                    q_val.mean()
                    + (alpha_eff * entropy_samples).mean()
                )

                with torch.no_grad():
                    _la = alpha_eff.double().log()
                    _LN10 = 2.302585092994046
                    self._last_gate_stats = {
                        # realised budget.  Frozen Z means this drifts
                        # as training moves the state distribution; the
                        # drift is reported, not corrected.
                        "alpha_eff_geomean": float(_la.mean().exp()),
                        "alpha_eff_mean": float(alpha_eff.double().mean()),
                        # if this collapses towards 0 the allocation has
                        # flattened and A1r has quietly become A1
                        "alpha_eff_spread_decades": float(
                            (torch.quantile(_la, 0.90)
                             - torch.quantile(_la, 0.10)) / _LN10
                        ),
                        "alpha_eff_min": float(alpha_eff.min()),
                        "alpha_eff_max": float(alpha_eff.max()),
                    }

            elif self.alpha != 0.0:
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
            # --- A1r gate diagnostics (empty dict when alpha_fn is None) ---
            **self._last_gate_stats,
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
            # A1r: the frozen gate constants travel with the
            # weights, so any later analysis can reconstruct the
            # exact alpha_eff field this run was trained under.
            "a1r_gate": (self.alpha_fn.to_dict()
                         if self.alpha_fn is not None else None),
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
