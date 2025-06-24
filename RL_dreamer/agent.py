import torch
import torch.nn as nn
import torch.distributions as D
from typing import Dict, Tuple

class Actor(nn.Module):
    """
    The policy network (actor).
    Takes a latent state and outputs a distribution over actions.
    """
    def __init__(self, config: Dict, stoch_dim: int, deter_dim: int, action_dim: int):
        super().__init__()
        self.hidden_dim = config['hidden_units']
        self.action_dim = action_dim
        self.activation = getattr(nn, config.get('activation', 'ELU'))()

        self.net = nn.Sequential(
            nn.Linear(stoch_dim + deter_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, 2 * self.action_dim) # For mean and std of a Gaussian action distribution
        )
    
    def forward(self, stoch_state: torch.Tensor, deter_state: torch.Tensor) -> D.Distribution:
        latent_state = torch.cat([stoch_state, deter_state], dim=-1)
        action_params = self.net(latent_state)
        mean, std = torch.chunk(action_params, 2, dim=-1)
        std = torch.nn.functional.softplus(std) + 1e-4 # Ensure std is positive
        
        # We can add tanh normalization for bounded action spaces if needed
        return D.Normal(mean, std)

class Critic(nn.Module):
    """
    The value network (critic).
    Takes a latent state and estimates the expected return.
    """
    def __init__(self, config: Dict, stoch_dim: int, deter_dim: int):
        super().__init__()
        self.hidden_dim = config['hidden_units']
        self.activation = getattr(nn, config.get('activation', 'ELU'))()
        
        self.net = nn.Sequential(
            nn.Linear(stoch_dim + deter_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, stoch_state: torch.Tensor, deter_state: torch.Tensor) -> torch.Tensor:
        latent_state = torch.cat([stoch_state, deter_state], dim=-1)
        return self.net(latent_state)

class Agent:
    """
    The main agent class that contains the actor, critic, and the policy training logic.
    """
    def __init__(self, config: Dict, world_model: nn.Module, stoch_dim: int, deter_dim: int, action_dim: int):
        self.config = config
        self.world_model = world_model
        
        self.actor = Actor(config['agent'], stoch_dim, deter_dim, action_dim)
        self.critic = Critic(config['agent'], stoch_dim, deter_dim)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['agent']['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['agent']['critic_lr'])

    def train_policy(self, initial_states: Tuple[torch.Tensor, torch.Tensor]):
        """
        Trains the actor and critic networks using imagined trajectories.
        """
        stoch_state, deter_state = initial_states

        # 1. Imagine trajectories using the world model and current actor
        # Detach the initial states so we don't backpropagate into the world model from the policy training
        stoch_state, deter_state = stoch_state.detach(), deter_state.detach()

        with torch.no_grad():
             (imagined_stoch, imagined_deter) = self.world_model.rssm.imagine(
                 stoch_state, deter_state, self.actor, self.config['agent']['horizon']
             )
        
        # 2. Predict rewards and values for the imagined states
        imagined_rewards = self.world_model.reward_decoder(imagined_stoch, imagined_deter).squeeze(-1)
        imagined_values = self.critic(imagined_stoch, imagined_deter).squeeze(-1)

        # 3. Calculate lambda-returns (GAE)
        lambda_returns = self.compute_lambda_returns(imagined_rewards, imagined_values)
        
        # 4. Detach the lambda returns target for the actor update
        with torch.no_grad():
            advantage = lambda_returns - imagined_values

        # 5. Compute actor and critic losses
        actor_loss = self.compute_actor_loss(imagined_stoch, imagined_deter, advantage)
        critic_loss = self.compute_critic_loss(lambda_returns, imagined_values)

        # 6. Update actor and critic
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
        
    def compute_lambda_returns(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Computes lambda-returns (Generalized Advantage Estimation).
        """
        gamma = self.config['agent']['gamma']
        lambda_ = self.config['agent']['lambda_gae']
        
        # The last value is the bootstrap value
        next_value = values[:, -1]
        returns = torch.empty_like(rewards)

        # Iterate backwards from the second to last step
        for t in reversed(range(rewards.shape[1])):
            next_value = rewards[:, t] + gamma * next_value
            returns[:, t] = next_value
            next_value = (1 - lambda_) * values[:, t] + lambda_ * next_value
        
        return returns

    def compute_actor_loss(self, stoch: torch.Tensor, deter: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        """
        Computes the actor loss.
        """
        # The actor is updated based on the policy gradient theorem.
        # We need to re-evaluate the actions under the current policy to get log_probs.
        # We detach the states to prevent gradients from flowing into the world model.
        action_dist = self.actor(stoch.detach(), deter.detach())
        
        # The advantage is the target for the policy gradient update.
        actor_loss = -(action_dist.log_prob(action_dist.sample()).sum(-1) * advantage.detach()).mean()
        
        # Entropy bonus to encourage exploration
        entropy_loss = -self.config['agent']['entropy_weight'] * action_dist.entropy().mean()
        
        return actor_loss + entropy_loss

    def compute_critic_loss(self, lambda_returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Computes the critic loss.
        """
        # The critic is trained to predict the lambda-returns.
        critic_loss = torch.nn.functional.mse_loss(values, lambda_returns)
        return critic_loss
