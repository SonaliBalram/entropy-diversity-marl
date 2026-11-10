import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
 
 
class SharedContinuousActor(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden_dim=128
    ):
        super().__init__()
        self.action_dim = action_dim
 
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
 
        self.mean_layer = nn.Linear(
            hidden_dim,
            action_dim
        )
 
        self.log_std_layer = nn.Linear(
            hidden_dim,
            action_dim
        )
 
        self.LOG_STD_MIN = -5
        self.LOG_STD_MAX = 2
 
    def get_distribution(self, observations):
        features = self.network(observations)
 
        mean = self.mean_layer(features)
 
        log_std = self.log_std_layer(features)
 
        log_std = torch.clamp(
            log_std,
            self.LOG_STD_MIN,
            self.LOG_STD_MAX
        )
 
        std = torch.exp(log_std)
 
        distribution = Normal(mean, std)
 
        return distribution
 
    def sample_action(self, observations):
        distribution = self.get_distribution(
            observations
        )
 
        z = distribution.rsample()
 
        actions = torch.sigmoid(z)
 
        gaussian_log_prob = distribution.log_prob(z)
 
        gaussian_log_prob = gaussian_log_prob.sum(
            dim=-1
        )
 
        log_jacobian = (
            -F.softplus(-z)
            -F.softplus(z)
        )
 
        log_jacobian = log_jacobian.sum(
            dim=-1
        )
 
        log_probs = (
            gaussian_log_prob
            - log_jacobian
        )
 
        entropy_samples = -log_probs
 
        return (
            actions,
            log_probs,
            entropy_samples
        )
 
    def deterministic_action(self, observations):
        distribution = self.get_distribution(
            observations
        )
        mean = distribution.mean
 
        actions = torch.sigmoid(mean)
 
        return actions
 
    def forward(self, observations):
        return self.sample_action(observations)
 
    def policy_stats(self, observations):
        with torch.no_grad():
            distribution = self.get_distribution(observations)

            return {
                "mean_std": distribution.stddev.mean().item(),
                "mean_entropy": distribution.entropy()
                                    .sum(-1).mean().item(),
                "mean_spread": distribution.mean
                                    .std(dim=0).mean().item(),
            }
