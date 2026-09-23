import torch
import torch.nn as nn

#creates a neural network that inherits from nn.Module
#so that pytorch know it is a trainable neural network
class CentralizedCritic(nn.Module):

#constructor for the critic class
#tells the critic how many agents exist
#number of values in one agent's local observation
#number of continous action components each agent has ( 2 components)
#number of neurons in each hidden layer
    def __init__(
        self,
        n_agents,
        observation_dim,
        action_dim,
        hidden_dim=128,
        context_dim=0
    ):
# context_dim = 0 reproduces the original critic EXACTLY: the input
# width, the layer shapes and therefore the number and order of RNG
# draws during initialisation are all unchanged. Experiments 1, 2 and
# the unaugmented Experiment 3 arm never pass it.
#
# context_dim > 0 appends a per-transition CONTEXT vector to the
# critic input. Experiment 3's augmented arm passes the 3x3 running
# token-count matrix (9 numbers), which is what makes the terminal
# diversity bonus a deterministic function of the critic's input
# rather than an unpredictable shock.
#calls the parent constructor to initialize the nn.Module class       
        super().__init__()
#stores these values (dimensions) inside the critic
        self.n_agents = n_agents
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # -------------------------------------------------
        # Total input dimensions
        # -------------------------------------------------

#calculates total observation dimension, say we have N=3, and each agents has 18 observations, then total_dim=54
        total_observation_dim = (
            n_agents * observation_dim
        )

        total_action_dim = (
            n_agents * action_dim
        )

#critic get both observations and actions as input, so the input dimension is the sum of both
        critic_input_dim = (
            total_observation_dim
            + total_action_dim
            + context_dim
        )

        # -------------------------------------------------
        # Centralized critic network
        # -------------------------------------------------

#this creates the actual neural network that approximates the Q-function
        self.network = nn.Sequential(

#critic takes the entire joint state-action vector and transforms it into a learned feature representation, then outputs a single Q-value
            nn.Linear(
                critic_input_dim,
                hidden_dim
            ),

            nn.ReLU(),

#second hidden layer
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

#final hidden layer
#critic ouputs one number, the Q-value
            nn.Linear(
                hidden_dim,
                1
            )
        )

#this defines what happens when you call critic
    def forward(
        self,
        observations,
        actions,
        context=None
    ):

        """
        observations:
            [batch_size, n_agents, observation_dim]

        actions:
            [batch_size, n_agents, action_dim]

        context:
            [batch_size, context_dim] or None.
            Required when context_dim > 0, ignored otherwise.

        returns:
            [batch_size, 1]
        """

        # -------------------------------------------------
        # Flatten observations
        # -------------------------------------------------

#turning individual agent observations into one long vector of total observations for every batch sample
        observations = observations.reshape(
            observations.shape[0],
            -1
        )

        # -------------------------------------------------
        # Flatten actions
        # -------------------------------------------------

        actions = actions.reshape(
            actions.shape[0],
            -1
        )

        # -------------------------------------------------
        # Joint state-action vector
        # -------------------------------------------------
#concatenates the flattened observations and actions into one long vector for each batch sample
        parts = [observations, actions]

        # -------------------------------------------------
        # Optional context
        # -------------------------------------------------
        #
        # Guarded both ways on purpose. Silently ignoring a
        # context that was passed to an unaugmented critic, or
        # silently zero-filling one that was not passed to an
        # augmented critic, would both produce a run that looks
        # healthy and answers a different question.

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

        # -------------------------------------------------
        # Q(s,a)
        # -------------------------------------------------

#feed your concatenated vector into the critic network
        q_value = self.network(
            critic_input
        )
#return the Q-value for each batch sample
        return q_value