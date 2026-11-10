import torch
import torch.nn as nn

class CentralizedCritic(nn.Module):
    def __init__(
        self,
        n_agents,
        observation_dim,
        action_dim,
        hidden_dim=128,
        context_dim=0
    ):
        super().__init__()
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        total_observation_dim = (
            n_agents * observation_dim
        )

        total_action_dim = (
            n_agents * action_dim
        )

        critic_input_dim = (
            total_observation_dim
            + total_action_dim
            + context_dim
        )

        self.network = nn.Sequential(

            nn.Linear(
                critic_input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1
            )
        )

    def forward(
        self,
        observations,
        actions,
        context=None
    ):
        observations = observations.reshape(
            observations.shape[0],
            -1
        )

        actions = actions.reshape(
            actions.shape[0],
            -1
        )

        parts = [observations, actions]

        if self.context_dim > 0:
            if context is None:
                raise ValueError(
                    "critic was built with context_dim="
                    f"{self.context_dim} but forward() got "
                    "context=None"
                )
            parts.append(
                context.reshape(context.shape[0], -1)
            )
        elif context is not None:
            raise ValueError(
                "critic was built with context_dim=0 but "
                "forward() was given a context"
            )

        critic_input = torch.cat(
            parts,
            dim=-1
        )

        q_value = self.network(
            critic_input
        )
        return q_value
