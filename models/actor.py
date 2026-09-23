
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
 
 
#This class creates your actor neural network
#Same actor is used for every agent
#Actions are continous
#inherits from pytorch nn.Module class
class SharedContinuousActor(nn.Module):
 
#This is the constructor for the actor class
#it runs when you create an instance of the class, actor
#3 parameters
#observation_dim: tells the actor how many values are in one agent's local observation
#action_dim: tells the actor how many continous action components each agent has ( 2 components)
#hidden_dim: number of neurons in the hidden layers of the actor network, default is 128
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim=128
    ):
#calls the parent constructor to initialize the nn.Module class
        super().__init__()
#stores the action dimension inside the actor class for later use
        self.action_dim = action_dim
 
        # =================================================
        # SHARED POLICY NETWORK
        # =================================================
        #
        # The same network parameters are used by every
        # agent. Each agent supplies its own observation.
        #
        # Input:
        #     [n_agents, observation_dim]
        #
        # Output:
        #     [n_agents, hidden_dim]
        #
 
 
        self.network = nn.Sequential(
#an 18-dimensional observation becomes a 128-dimensional feature vector
            nn.Linear(observation_dim, hidden_dim),
#gives network non-linearity
            nn.ReLU(),
#another hidden layer with 128 neurons
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
 
        # =================================================
        # GAUSSIAN POLICY PARAMETERS
        # =================================================
 
#network predicts the mean of the Gaussian Distribution for each action dimension
        self.mean_layer = nn.Linear(
            hidden_dim,
            action_dim
        )
 
#predicts Log standard deviation of the Gaussian Distribution for each action dimension
        self.log_std_layer = nn.Linear(
            hidden_dim,
            action_dim
        )
#at this point, actor has learned the mean and standard deviation from the observation
 
        # =================================================
        # NUMERICAL STABILITY BOUNDS ON log sigma
        # =================================================
        #
        # LOG_STD_MAX = 2 is kept unchanged (sigma_max ~ 7.4).
        # Tightening it would cap the maximum achievable
        # stochasticity and bias the Exp-1 vs Exp-2 comparison,
        # since Exp 2 adds an entropy bonus on the same
        # architecture.
        #
        # LOG_STD_MIN is raised from -20 to -5.
        #   -20 allows sigma ~ 2e-9. At that scale
        #   log pi(a|o) ~ +19 per dimension and the Monte-Carlo
        #   entropy estimate blows up numerically, which
        #   corrupts exactly the diversity statistic this
        #   baseline exists to measure.
        #   -5 gives sigma_min ~ 6.7e-3, still effectively
        #   deterministic relative to the [0,1] action range,
        #   so the policy can still collapse -- it just cannot
        #   collapse into numerical garbage.
        #
        # This floor is applied identically in every
        # experiment, so it does not bias the comparison: it
        # bounds the collapse from below, not the entropy
        # bonus from above.
        self.LOG_STD_MIN = -5
        self.LOG_STD_MAX = 2
 
#this function takes obervations and constructs the actor's probability distribution
#this function does not sample an action
    def get_distribution(self, observations):
 
        """
        Calculate the Gaussian policy distribution.
 
        observations:
            [n_agents, observation_dim]
 
        Returns:
            Normal distribution with:
                mean: [n_agents, action_dim]
                std:  [n_agents, action_dim]
        """
 
        # -------------------------------------------------
        # Shared feature extraction
        # -------------------------------------------------
 
        features = self.network(observations)
 
        # -------------------------------------------------
        # Gaussian mean
        # -------------------------------------------------
 
        mean = self.mean_layer(features)
 
        # -------------------------------------------------
        # Gaussian log standard deviation
        # -------------------------------------------------
 
        log_std = self.log_std_layer(features)
 
        log_std = torch.clamp(
            log_std,
            self.LOG_STD_MIN,
            self.LOG_STD_MAX
        )
 
        std = torch.exp(log_std)
 
        # -------------------------------------------------
        # Gaussian policy
        # -------------------------------------------------
 
#constructs the normal distribution for every agent and every action dimension
        distribution = Normal(mean, std)
 
        return distribution
 
#we use this function when we want the policy to act stochastically
    def sample_action(self, observations):
 
        """
        Sample stochastic continuous actions.
 
        observations:
            Tensor of shape
 
                [n_agents, observation_dim]
 
        Returns:
 
            actions:
                [n_agents, action_dim]
 
            log_probs:
                [n_agents]
 
            entropy_samples:
                [n_agents]
        """
 
        # =================================================
        # 1. Construct Gaussian policy
        # =================================================
#construct the probability distribution for each agent's action given its observation
        distribution = self.get_distribution(
            observations
        )
 
        # =================================================
        # 2. Reparameterized sample
        # =================================================
        #
        # z = μ + σ ε
        #
        # rsample() allows gradients to flow through z.
        #
#reparameterized sampling, different epsilons
        z = distribution.rsample()
 
        # =================================================
        # 3. Squash into [0,1]
        # =================================================
        #
        # MPE continuous actions have bounds [0,1].
        #
# squash the sampled action into the range [0,1] using sigmoid
        actions = torch.sigmoid(z)
 
        # =================================================
        # 4. Gaussian log probability
        # =================================================
 
        gaussian_log_prob = distribution.log_prob(z)
 
        # Sum across action dimensions
        #
        # [n_agents, action_dim]
        #
        # becomes
        #
        # [n_agents]
        #
 
        gaussian_log_prob = gaussian_log_prob.sum(
            dim=-1
        )
 
        # =================================================
        # 5. Sigmoid Jacobian correction
        # =================================================
        #
        # a = sigmoid(z)
        #
        # da/dz = a(1-a)
        #
        # Therefore:
        #
        # log π(a|o)
        #
        # = log p(z|o)
        #   - log|da/dz|
        #
 
#because your action action is not z, it is a=sigmoid(z), so we need to correct the log probability of the action by subtracting the log of the absolute value of the Jacobian determinant of the sigmoid function
#change of variables formula for probability distributions
        log_jacobian = (
            -F.softplus(-z)
            -F.softplus(z)
        )
 
        # Sum across action dimensions
        log_jacobian = log_jacobian.sum(
            dim=-1
        )
 
        # =================================================
        # 6. Correct log probability
        # =================================================
 
        log_probs = (
            gaussian_log_prob
            - log_jacobian
        )
 
        # =================================================
        # 7. Monte-Carlo entropy estimate
        # =================================================
        #
        # H(π) = E[-log π(a|s)]
        #
        # For a sampled action:
        #
        # H ≈ -log π(a|s)
        #
 
        entropy_samples = -log_probs
 
#function gives the training system 3 things
#actions, goes into centralised critic to evaluate
#log_probs, needed for entropy
#entropy samples, needed for entropy regularization
        return (
            actions,
            log_probs,
            entropy_samples
        )
 
#this is used when u want the actor to produce an action without sampling
    def deterministic_action(self, observations):
 
        """
        Return the deterministic mean action.
 
        Useful for evaluation/testing.
        """
#construct the distribution
        distribution = self.get_distribution(
            observations
        )
#instead of sampling, you simply take z=mean
        mean = distribution.mean
 
#apply sigmoid
        actions = torch.sigmoid(mean)
 
#central action predicted by policy, no randomness
        return actions
 
#default process to perform stochastic sampling
    def forward(self, observations):
 
        """
        Default forward pass.
 
        Equivalent to sample_action().
        """
        return self.sample_action(observations)
 

    # =====================================================
    # DIAGNOSTICS FOR THE HOMOGENIZATION ANALYSIS
    # =====================================================

    def policy_stats(self, observations):

        """
        Summary statistics of the policy at a set of
        observations. Used for logging, never for training.

        observations:
            [n, observation_dim]

        Returns dict with:
            mean_std     : average sigma over dims/agents
                           -> ACTION-LEVEL stochasticity
            mean_entropy : analytic Gaussian entropy (pre-squash)
            mean_spread  : std ACROSS the n rows of the
                           predicted means
                           -> how differently the shared policy
                              responds to different agents'
                              observations. This is the
                              quantity that distinguishes
                              "stochastic" from "diverse".
        """

        with torch.no_grad():

            distribution = self.get_distribution(observations)

            return {
                "mean_std": distribution.stddev.mean().item(),
                "mean_entropy": distribution.entropy()
                                    .sum(-1).mean().item(),
                "mean_spread": distribution.mean
                                    .std(dim=0).mean().item(),
            }
