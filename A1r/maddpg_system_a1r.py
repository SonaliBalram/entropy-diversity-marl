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

        self.grad_clip_norm = grad_clip_norm

        self.huber_delta = huber_delta

        self.n_target_samples = n_target_samples

        self.policy_delay = policy_delay
        self._update_index = policy_delay - 1

        self.twin_critics = twin_critics

        self.context_dim = context_dim

        self.alpha = alpha

        self.alpha_fn = alpha_fn
        self._last_gate_stats = {}

        self._last_actor_stats = {
            "actor_loss": float("nan"),
            "mean_q": float("nan"),
            "actor_grad_norm": float("nan"),
        }

        self.device = device

        self.actor = SharedContinuousActor(
            observation_dim,
            action_dim,
            hidden_dim
        ).to(device)

        self.actor_target = copy.deepcopy(self.actor).to(device)

        def make_critic():
            return CentralizedCritic(
                n_agents,
                observation_dim,
                action_dim,
                hidden_dim,
                context_dim=context_dim
            ).to(device)

        self.critic = make_critic()
        self.critic_2 = make_critic() if twin_critics else None

        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_target_2 = (
            copy.deepcopy(self.critic_2).to(device)
            if twin_critics else None
        )

        for p in self.actor_target.parameters():
            p.requires_grad = False

        for p in self.critic_target.parameters():
            p.requires_grad = False

        if self.twin_critics:
            for p in self.critic_target_2.parameters():
                p.requires_grad = False

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

    def act(self, obs_n, deterministic=False):
        obs_t = torch.as_tensor(
            obs_n,
            dtype=torch.float32,
            device=self.device
        )

        with torch.no_grad():
            if deterministic:
                actions = self.actor.deterministic_action(obs_t)

            else:
                actions, _, _ = self.actor.sample_action(obs_t)

        return actions.cpu().numpy()

    def policy_params(self, obs_n):
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

    def _polyak_update(self, net, target_net):
        with torch.no_grad():
            for p, tp in zip(
                net.parameters(),
                target_net.parameters()
            ):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * p.data)

            for b, tb in zip(
                net.buffers(),
                target_net.buffers()
            ):
                tb.data.copy_(b.data)

    def update(self, batch):
        if self.context_dim > 0:
            (obs, acts, team_rew, next_obs, done,
             ctx, next_ctx) = batch
        else:
            obs, acts, team_rew, next_obs, done = batch
            ctx = next_ctx = None

        B = obs.shape[0]
        N = self.n_agents

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

        q = self.critic(obs, acts, ctx)

        td_error = (y - q).detach()

        critic_loss = F.huber_loss(q, y, delta=self.huber_delta)

        if self.twin_critics:
            q_b = self.critic_2(obs, acts, ctx)

            critic_loss = critic_loss + F.huber_loss(
                q_b, y, delta=self.huber_delta
            )

        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward()

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self._critic_params,
            self.grad_clip_norm
            if self.grad_clip_norm is not None
            else float("inf")
        )

        self.critic_opt.step()

        self._update_index += 1

        do_actor_update = (
            self._update_index % self.policy_delay == 0
        )

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

            q_val = self.critic(obs, acts_resampled, ctx)

            if self.alpha_fn is not None:
                alpha_eff = self.alpha_fn(obs).detach().reshape(B * N)

                actor_loss = -(
                    q_val.mean()
                    + (alpha_eff * entropy_samples).mean()
                )

                with torch.no_grad():
                    _la = alpha_eff.double().log()
                    _LN10 = 2.302585092994046
                    self._last_gate_stats = {
                        "alpha_eff_geomean": float(_la.mean().exp()),
                        "alpha_eff_mean": float(alpha_eff.double().mean()),
                        "alpha_eff_spread_decades": float(
                            (torch.quantile(_la, 0.90)
                             - torch.quantile(_la, 0.10)) / _LN10
                        ),
                        "alpha_eff_min": float(alpha_eff.min()),
                        "alpha_eff_max": float(alpha_eff.max()),
                    }

            elif self.alpha != 0.0:
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

            self._polyak_update(self.actor, self.actor_target)
            self._polyak_update(self.critic, self.critic_target)

            if self.twin_critics:
                self._polyak_update(
                    self.critic_2, self.critic_target_2
                )

        with torch.no_grad():
            distribution = self.actor.get_distribution(obs_flat)

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
            "policy_std": distribution.stddev.mean().item(),
            "policy_entropy_mc": entropy_stat.mean().item(),
            "policy_entropy_analytic":
                distribution.entropy().sum(-1).mean().item(),
            "inter_agent_mean_spread": inter_agent_spread,
            "realised_action_spread": realised_spread,
            **self._last_gate_stats,
        }

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
            "a1r_gate": (self.alpha_fn.to_dict()
                         if self.alpha_fn is not None else None),
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
