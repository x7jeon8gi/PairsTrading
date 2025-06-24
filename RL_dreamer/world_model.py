import torch
import torch.nn as nn
import torch.distributions as D
from typing import Dict, Tuple

class Encoder(nn.Module):
    """Encodes the observation into a latent representation."""
    def __init__(self, obs_dim: int, hidden_dim: int, activation=nn.ELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim) 
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM).
    This model learns the dynamics of the environment in a latent space.
    """
    def __init__(self, config: Dict, action_dim: int, encoder_output_dim: int):
        super().__init__()
        self.stoch_dim = config['stochastic_state_dim']
        self.deter_dim = config['deterministic_state_dim']
        self.hidden_dim = config['hidden_units']
        self.activation = getattr(nn, config['activation'])()

        # Recurrent model (deterministic path)
        self.rnn = nn.GRUCell(self.hidden_dim, self.deter_dim)
        
        # Combined state and action to feed into posterior/prior
        self.action_stoch_input = nn.Linear(action_dim + self.stoch_dim, self.hidden_dim)

        # Posterior (for training with real data)
        self.posterior_net = nn.Sequential(
            nn.Linear(self.deter_dim + encoder_output_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, 2 * self.stoch_dim) # mean and std
        )
        
        # Prior (for imagination)
        self.prior_net = nn.Sequential(
            nn.Linear(self.deter_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, 2 * self.stoch_dim) # mean and std
        )

    def initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the initial zero state for the RSSM."""
        return (torch.zeros(batch_size, self.stoch_dim, device=device),
                torch.zeros(batch_size, self.deter_dim, device=device))

    def observe(self, embedded_obs: torch.Tensor, actions: torch.Tensor, is_first: torch.Tensor,
                initial_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a sequence of observations and actions to compute the posterior states.

        Args:
            embedded_obs: A sequence of encoded observations (batch, seq_len, embed_dim).
            actions: A sequence of actions (batch, seq_len, action_dim).
            is_first: A tensor indicating the start of a new episode.
            initial_state: The initial state of the RSSM.

        Returns:
            A tuple of posterior states (stochastic and deterministic) for the sequence.
        """
        # Transpose to (seq_len, batch, dim) for iterating over the sequence
        embedded_obs = embedded_obs.transpose(0, 1)
        actions = actions.transpose(0, 1)
        
        post_states = []
        prior_states = []
        
        stoch_state, deter_state = initial_state
        
        for i in range(len(embedded_obs)):
            # On the first step of an episode, reset the state
            stoch_state = stoch_state * (1.0 - is_first[:, i]).unsqueeze(-1)
            deter_state = deter_state * (1.0 - is_first[:, i]).unsqueeze(-1)

            # 1. Compute prior state from previous state and action
            action_stoch = self.action_stoch_input(torch.cat([actions[i], stoch_state], dim=-1))
            deter_state = self.rnn(action_stoch, deter_state)
            
            prior_logits = self.prior_net(deter_state)
            prior_dist = D.Normal(prior_logits[..., :self.stoch_dim], torch.nn.functional.softplus(prior_logits[..., self.stoch_dim:]))
            
            # 2. Compute posterior state from prior and current observation
            posterior_input = torch.cat([deter_state, embedded_obs[i]], dim=-1)
            posterior_logits = self.posterior_net(posterior_input)
            posterior_dist = D.Normal(posterior_logits[..., :self.stoch_dim], torch.nn.functional.softplus(posterior_logits[..., self.stoch_dim:]))

            # Sample from posterior for the next step's input
            stoch_state = posterior_dist.rsample()

            post_states.append((stoch_state, deter_state))
            prior_states.append((prior_dist.mean, prior_dist.scale))

        # Stack and transpose back to (batch, seq_len, dim)
        post_stoch = torch.stack([s[0] for s in post_states], dim=0).transpose(0, 1)
        post_deter = torch.stack([s[1] for s in post_states], dim=0).transpose(0, 1)
        prior_mean = torch.stack([s[0] for s in prior_states], dim=0).transpose(0, 1)
        prior_std = torch.stack([s[1] for s in prior_states], dim=0).transpose(0, 1)
        
        return (post_stoch, post_deter), (prior_mean, prior_std), (posterior_dist.mean, posterior_dist.scale)

    def imagine(self, initial_stoch_state: torch.Tensor, initial_deter_state: torch.Tensor,
                actor: nn.Module, horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Imagines future trajectories from a starting state.

        Args:
            initial_stoch_state: The starting stochastic state.
            initial_deter_state: The starting deterministic state.
            actor: The policy network to generate actions.
            horizon: The number of steps to imagine into the future.

        Returns:
            A tuple containing imagined stochastic states and their corresponding deterministic states.
        """
        stoch_state = initial_stoch_state
        deter_state = initial_deter_state
        
        imagined_stoch_states = []
        imagined_deter_states = []

        for _ in range(horizon):
            # 1. Get action from the actor using the current latent state
            combined_state = torch.cat([stoch_state, deter_state], dim=-1)
            action = actor(combined_state).sample() # Assumes actor returns a distribution

            # 2. Predict the next state using the world model's prior
            action_stoch = self.action_stoch_input(torch.cat([action, stoch_state], dim=-1))
            deter_state = self.rnn(action_stoch, deter_state)
            
            prior_logits = self.prior_net(deter_state)
            prior_dist = D.Normal(prior_logits[..., :self.stoch_dim], torch.nn.functional.softplus(prior_logits[..., self.stoch_dim:]))
            stoch_state = prior_dist.sample()

            imagined_stoch_states.append(stoch_state)
            imagined_deter_states.append(deter_state)
            
        return (torch.stack(imagined_stoch_states, dim=0).transpose(0, 1),
                torch.stack(imagined_deter_states, dim=0).transpose(0, 1))

class Decoder(nn.Module):
    """Decodes a latent state into an observation or reward."""
    def __init__(self, stoch_dim: int, deter_dim: int, output_dim: int, hidden_dim: int, activation=nn.ELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stoch_dim + deter_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, stoch_state: torch.Tensor, deter_state: torch.Tensor) -> torch.Tensor:
        combined_state = torch.cat([stoch_state, deter_state], dim=-1)
        return self.net(combined_state)

    def compute_loss(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Computes the total loss for the world model.
        """
        batch_size, seq_len, _ = obs.shape
        device = obs.device

        # Encode all observations in the sequence
        embedded_obs = self.encoder(obs)
        
        # Initialize RSSM state
        initial_state = self.rssm.initial_state(batch_size, device)
        is_first = torch.zeros(batch_size, seq_len, device=device) # Assume no resets within a chunk for now
        is_first[:, 0] = 1

        # Process the sequence to get latent states
        (post_stoch, post_deter), (prior_mean, prior_std), (post_mean, post_std) = self.rssm.observe(
            embedded_obs, actions, is_first, initial_state
        )
        
        # 1. Reconstruction Loss (Observation and Reward)
        recon_obs = self.obs_decoder(post_stoch, post_deter)
        recon_reward = self.reward_decoder(post_stoch, post_deter).squeeze(-1)
        
        recon_obs_loss = torch.nn.functional.mse_loss(recon_obs, obs, reduction='none').mean([0, 1])
        recon_reward_loss = torch.nn.functional.mse_loss(recon_reward, rewards, reduction='none').mean([0, 1])
        
        # 2. KL Divergence Loss
        prior_dist = D.Normal(prior_mean, prior_std)
        post_dist = D.Normal(post_mean, post_std)
        kl_loss = torch.max(
            D.kl_divergence(post_dist, prior_dist).sum(-1),
            # Free bits - a minimum KL value to prevent posterior collapse
            torch.tensor(1.0, device=device) 
        ).mean([0, 1])

        # Total Loss
        total_loss = recon_obs_loss + recon_reward_loss + kl_loss

        return total_loss, recon_obs_loss, recon_reward_loss, kl_loss

class WorldModel(nn.Module):
    """
    The main World Model class that combines the Encoder, RSSM, and Decoders.
    """
    def __init__(self, config: Dict, obs_dim: int, action_dim: int):
        super().__init__()
        self.config = config
        wm_config = config['world_model']
        
        self.encoder = Encoder(obs_dim, wm_config['hidden_units'])
        self.rssm = RSSM(wm_config, action_dim, wm_config['hidden_units'])
        
        # Observation decoder reconstructs the original observation
        self.obs_decoder = Decoder(
            wm_config['stochastic_state_dim'], 
            wm_config['deterministic_state_dim'], 
            obs_dim, 
            wm_config['hidden_units']
        )
        
        # Reward decoder predicts the reward
        self.reward_decoder = Decoder(
            wm_config['stochastic_state_dim'], 
            wm_config['deterministic_state_dim'], 
            1, # Reward is a single value
            wm_config['hidden_units']
        )

    def compute_loss(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Computes the total loss for the world model.
        """
        batch_size, seq_len, _ = obs.shape
        device = obs.device

        # Encode all observations in the sequence
        embedded_obs = self.encoder(obs)
        
        # Initialize RSSM state
        initial_state = self.rssm.initial_state(batch_size, device)
        is_first = torch.zeros(batch_size, seq_len, device=device) # Assume no resets within a chunk for now
        is_first[:, 0] = 1

        # Process the sequence to get latent states
        (post_stoch, post_deter), (prior_mean, prior_std), (post_mean, post_std) = self.rssm.observe(
            embedded_obs, actions, is_first, initial_state
        )
        
        # 1. Reconstruction Loss (Observation and Reward)
        recon_obs = self.obs_decoder(post_stoch, post_deter)
        recon_reward = self.reward_decoder(post_stoch, post_deter).squeeze(-1)
        
        recon_obs_loss = torch.nn.functional.mse_loss(recon_obs, obs, reduction='none').mean([0, 1])
        recon_reward_loss = torch.nn.functional.mse_loss(recon_reward, rewards, reduction='none').mean([0, 1])
        
        # 2. KL Divergence Loss
        prior_dist = D.Normal(prior_mean, prior_std)
        post_dist = D.Normal(post_mean, post_std)
        kl_loss = torch.max(
            D.kl_divergence(post_dist, prior_dist).sum(-1),
            # Free bits - a minimum KL value to prevent posterior collapse
            torch.tensor(1.0, device=device) 
        ).mean([0, 1])

        # Total Loss
        total_loss = recon_obs_loss + recon_reward_loss + kl_loss

        return total_loss, recon_obs_loss, recon_reward_loss, kl_loss

    def forward(self, obs, actions, rewards):
        # The forward pass is now essentially the loss computation
        return self.compute_loss(obs, actions, rewards)
