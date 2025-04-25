import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os
from typing import Optional
from copy import deepcopy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
            
def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / (mask.sum(axis=dim) + 1e-8)


class GRPOLoss(nn.Module):
    """
    GRPO Loss: Computes the policy loss for GRPO using a clipped surrogate objective with KL penalty.
    - Advantage = (discounted_returns - mean) / std  (no critic, no GAE)
    - Clipped Surrogate Objective
    - Optional entropy bonus
    """
    def __init__(self, clip_eps=0.2, kl_weight=0.1, discount_factor=0.999, entropy_coeff=0.01):
        super().__init__()
        """
        Args:
            clip_eps      : PPO-style clipping 범위 (ex: 0.2)
            kl_weight     : (사용하지 않는다면 0 또는 코드 제거)
            discount_factor: 감가율 gamma
            entropy_coeff : 엔트로피 항 가중치 (ex: 0.01)
        """
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight
        self.discount_factor = discount_factor
        self.entropy_coeff = entropy_coeff # Add entropy coefficient

    def forward(self, policy, trajectories, device, epsilon=1e-8):
        # --- 데이터 텐서화 ---
        # ... (obs_tensor, action_tensor, old_log_tensor 생성 부분은 동일)
        obs_tensor = torch.tensor(np.concatenate([traj['states'] for traj in trajectories]),
                                  dtype=torch.float32).to(device)
        action_tensor = torch.tensor(np.concatenate([traj['actions'] for traj in trajectories]),
                                     dtype=torch.float32).to(device)
        old_log_tensor = torch.tensor(np.concatenate([traj['log_probs'] for traj in trajectories]),
                                      dtype=torch.float32).to(device)
        rewards_tensor = torch.tensor(np.stack([traj['rewards'] for traj in trajectories]),
                                      dtype=torch.float32).to(device)
        num_traj, T = rewards_tensor.shape

        # --- 어드밴티지 계산 (이미지 공식 2: 정규화된 미래 보상 합) ---
        # 1. 모든 개별 보상의 평균/표준편차 계산
        mean_rewards = rewards_tensor.mean()
        std_rewards = rewards_tensor.std(unbiased=False) + epsilon

        # 2. 개별 보상 정규화
        normalized_rewards = (rewards_tensor - mean_rewards) / std_rewards

        # 3. 미래 정규화된 보상의 합 계산 (할인 없음)
        # 뒤집어서 cumsum 계산 후 다시 뒤집기
        advantages = torch.flip(torch.cumsum(torch.flip(normalized_rewards, dims=[1]), dim=1), dims=[1])
        advantages_tensor = advantages.reshape(-1) # [num_traj * T]

        # --- 정책 네트워크를 통한 새 Log Prob 및 엔트로피 계산 ---
        mean, std = policy(obs_tensor)
        std = std.clamp(min=epsilon)
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(action_tensor).sum(dim=-1)  # 액션 차원에 대해 합산
        #entropy = dist.entropy().mean() # 배치 및 액션 차원에 대한 평균 엔트로피

        # --- PPO Surrogate Objective 계산 ---
        ratios = torch.exp(new_log_probs - old_log_tensor)
        clipped_ratios = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps)
        # 어드밴티지 텐서 사용
        surr_loss = -torch.min(ratios * advantages_tensor, clipped_ratios * advantages_tensor).mean()

        # --- 최종 손실 (엔트로피 보너스 포함) ---
        # KL 발산 항은 주석 처리
        # kl_div = self.calculate_kl_divergence(policy, old_policy, obs_tensor, device) # 예시, 실제 구현 필요
        # total_loss = surr_loss + self.kl_weight * kl_div - self.entropy_coeff * entropy
        total_loss = surr_loss # - self.entropy_coeff * entropy

        return total_loss
        
class GRPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, action_space=None):
        super(GRPOPolicy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 네트워크 레이어 정의
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            in_dim = h
        self.feature_extractor = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(in_dim, action_dim)
        self.log_std_layer = nn.Linear(in_dim, action_dim)
        
        # 상수들
        self.LOG_SIG_MIN = LOG_SIG_MIN
        self.LOG_SIG_MAX = LOG_SIG_MAX
        self.epsilon = epsilon
        
        # 가중치 초기화
        self.apply(weights_init_)
        # Action scaling & bias
        if action_space is None:
            self.action_scale = torch.tensor(1.).to(self.device)
            self.action_bias = torch.tensor(0.).to(self.device)
        else:
            if isinstance(action_space, list):
                if len(action_space) != 2:
                    raise ValueError("action_space list must have two elements [min, max]")
                low, high = action_space
                low = torch.tensor(low, dtype=torch.float32).to(self.device)
                high = torch.tensor(high, dtype=torch.float32).to(self.device)
            else:
                low = torch.tensor(action_space.low, dtype=torch.float32).to(self.device)
                high = torch.tensor(action_space.high, dtype=torch.float32).to(self.device)
            self.action_scale = (high - low) / 2.0
            self.action_bias = (high + low) / 2.0
            print(f"Action Scale: {self.action_scale}")
            print(f"Action Bias: {self.action_bias}")

    def forward(self, state):
        x = self.feature_extractor(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(self.LOG_SIG_MIN, self.LOG_SIG_MAX)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        if torch.isnan(mean).any() or torch.isnan(std).any():
            mean = torch.nan_to_num(mean, nan=0.0)
            std = torch.nan_to_num(std, nan=1.0).clamp(min=self.epsilon)
        else:
            std = std.clamp(min=self.epsilon)
        dist = Normal(mean, std)
        x_t = dist.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t) # [-1, 1]
        action = y_t * self.action_scale + self.action_bias

        # log_prob = dist.log_prob(x_t)
        # # Enforce action bounds via tanh transformation
        # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        # log_prob = log_prob.sum(dim=-1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias

        log_p_x = dist.log_prob(x_t)                      # (batch, action_dim)
        log_p_x = log_p_x.sum(dim=-1, keepdim=True)       # (batch, 1)

        eps = 1e-6
        log_j = torch.log(self.action_scale) + torch.log(torch.clamp(1 - y_t.pow(2), min=eps))
        log_j = log_j.sum(dim=-1, keepdim=True)           # (batch, 1)

        log_prob = log_p_x - log_j
        return action, log_prob, mean

class GRPOAgent:
    def __init__(self, 
                 state_dim,
                 action_dim,
                 hidden_sizes, 
                 action_space, 
                 lr=3e-4, 
                 clip_epsilon=0.2, 
                 group_size=5, 
                 kl_weight=0.1):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = GRPOPolicy(state_dim, action_dim, hidden_sizes, action_space).to(self.device)
        self.old_policy = GRPOPolicy(state_dim, action_dim, hidden_sizes, action_space).to(self.device)
        self.old_policy.load_state_dict(self.old_policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.group_size = group_size  # trajectories per update
        self.kl_weight = kl_weight
        self.loss_fn = GRPOLoss(self.clip_epsilon, self.kl_weight)

    def select_action(self, state, evaluate=False):
        # Check if state is already a tensor, otherwise convert
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
        # Ensure state is on the correct device
        state = state.to(self.device)
        # If state doesn't have a batch dimension, add one
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                # 평가 시에는 deterministic하게 mean을 사용하는 게 일반적
                mean, std = self.policy(state)
                # mean에 tanh 변환과 스케일링 적용하여 액션 공간으로 변환
                deterministic_action = torch.tanh(mean) * self.policy.action_scale + self.policy.action_bias
                return deterministic_action.cpu().detach().numpy()[0]  # 변환된 deterministic action 반환
            else:
                # 학습 시 trajectory는 old_policy에서 수집
                action, log_prob, _ = self.old_policy.sample(state)
                return action.cpu().detach().numpy()[0], log_prob.cpu().detach().item()

    def collect_trajectory(self, env, max_steps):
        self.old_policy.eval()  # Set old_policy to evaluation mode for data collection
        states, actions, log_probs, rewards = [], [], [], []
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.select_action(state_tensor, evaluate=False) # Still use old_policy logic internally
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            steps += 1

        # Note: We keep old_policy in eval mode here.
        # It should be set back to train mode if/when it's trained,
        # or kept in eval if it's only used for reference/sampling.
        # The policy being updated (self.policy) will be set to train mode in grpo_update.
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'rewards': np.array(rewards)
        }

    def grpo_update(self, group_trajectories, n_iterations=10):
        self.policy.train() # Set the policy being updated to train mode

        #all_states = np.concatenate([traj['states'] for traj in group_trajectories])
        #all_actions = np.concatenate([traj['actions'] for traj in group_trajectories])
        
        loss = 0

        for _ in range(n_iterations):
            loss = self.loss_fn(self.policy, group_trajectories, self.device)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.old_policy.load_state_dict(self.policy.state_dict())
        return loss.item()
    
    def save_model(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(self.policy.state_dict(), os.path.join(dir, f"{name}_grpo_policy.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(dir, f"{name}_grpo_optimizer.pth"))
    
    def load_model(self, dir, name):
        self.policy.load_state_dict(torch.load(os.path.join(dir, f"{name}_grpo_policy.pth"), map_location=self.device))
        self.optimizer.load_state_dict(torch.load(os.path.join(dir, f"{name}_grpo_optimizer.pth"), map_location=self.device))
        self.policy.eval()



# def approx_kl_divergence(
#     log_probs: torch.Tensor,
#     log_probs_ref: torch.Tensor,
#     action_mask: Optional[torch.Tensor],
# ) -> torch.Tensor:
#     log_ratio = log_probs_ref.float() - log_probs.float()
#     if action_mask is not None:
#         log_ratio = log_ratio * action_mask
#     return torch.exp(log_ratio-1) - log_ratio


# class GRPOLoss(nn.Module):
#     """
#     GRPO Loss: Computes the policy loss for GRPO using a clipped surrogate objective with KL penalty.
#     """
#     def __init__(self, clip_eps: float, kl_weight: float = 0.1) -> None:
#         super(GRPOLoss, self).__init__()
#         self.clip_eps = clip_eps
#         self.kl_weight = kl_weight
#         self.discount_factor = 1 # dicount factor
#     def forward(self, 
#                 policy, 
#                 trajectories,
#                 device,
#                 action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # Batch processing of trajectories
        
#         # 모든 데이터 tensor로 변환 (batch-level 처리)
#         obs_tensor = torch.tensor(np.concatenate([traj['states'] for traj in trajectories]), dtype=torch.float32).to(device)
#         action_tensor = torch.tensor(np.concatenate([traj['actions'] for traj in trajectories]), dtype=torch.float32).to(device)
#         old_log_tensor = torch.tensor(np.concatenate([traj['log_probs'] for traj in trajectories]), dtype=torch.float32).to(device)
#         rewards_tensor = torch.tensor(np.stack([traj['rewards'] for traj in trajectories]), dtype=torch.float32).to(device)  # (num_traj, T)
#         num_traj, T = rewards_tensor.shape
#         # Advantage 계산 (batch-level)
#         mean_reward = rewards_tensor.mean()
#         std_reward = rewards_tensor.std(unbiased=False) + 1e-8
#         normalized_rewards = (rewards_tensor - mean_reward) / std_reward
#         #advantages_tensor = torch.flip(torch.cumsum(torch.flip(normalized_rewards, dims=[1]), dim=1), dims=[1]).reshape(-1).to(device)
#         # discount factor 적용 (finance)
#         discount_factors = self.discount_factor ** torch.arange(T).float().to(device)
#         # 뒤집어서 discounted cumulative sum 적용 후 다시 뒤집음
#         discounted_cumsum = torch.flip(torch.cumsum(torch.flip(normalized_rewards * discount_factors, dims=[1]), dim=1), dims=[1])
        
#         advantages_tensor = discounted_cumsum.reshape(-1).to(device)

#         # # Advantage 계산: Discount factor와 Sharpe ratio 기반 계산
#         # gamma = 0.99  # Discounting factor
#         # window = 100  # Recent window size for volatility calculation
#         # rf_daily = 0.01 / 252  # Risk-free rate annualized to daily
#         # rewards_np = rewards_tensor.cpu().numpy()
#         # num_traj, T = rewards_np.shape
#         # advantages = np.zeros_like(rewards_np)

#         # for i in range(num_traj):
#         #     for t in range(T):
#         #         discounted_rewards = np.array([
#         #             (gamma ** (t_prime - t)) * (rewards_np[i, t_prime] - rf_daily)
#         #             for t_prime in range(t, T)
#         #         ])
                
#         #         if t >= window:
#         #             volatility = np.std(rewards_np[i, t-window:t]) + 1e-8
#         #         else:
#         #             volatility = np.std(rewards_np[i, :t+1]) + 1e-8
                
#         #         advantages[i, t] = discounted_rewards.sum() / volatility

#         # advantages_tensor = torch.tensor(advantages.flatten(), dtype=torch.float32).to(device)


#         # Forward pass (정책 네트워크의 병렬처리)
#         mean, std = policy(obs_tensor)
#         std = std.clamp(min=epsilon)
#         dist = Normal(mean, std)
#         new_log_probs = dist.log_prob(action_tensor).sum(dim=-1)

#         # PPO-style surrogate objective 계산 (vectorized)
#         ratios = torch.exp(new_log_probs - old_log_tensor)
#         clipped_ratios = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps)
#         surr_losses = -torch.min(ratios * advantages_tensor, clipped_ratios * advantages_tensor)


#         # Extract observations, actions, old_log_probs from all trajectories
#         # all_observations, all_actions, all_old_log_probs, all_advantages = [], [], [], []
        
#         # # Compute normalized advantages from trajectories' total rewards
#         # all_rewards = np.concatenate([traj['rewards'] for traj in trajectories])
#         # mean_reward = np.mean(all_rewards)
#         # std_reward = np.std(all_rewards) + 1e-8
        
#         # # Process all trajectories and prepare batched data
#         # for traj in trajectories:
#         #     rewards = traj['rewards']
#         #     # 보상 정규화
#         #     normalized_rewards = (np.array(rewards) - mean_reward) / std_reward
#         #     # 각 step마다 advantage 누적 (역방향)
#         #     advantages = np.array([np.sum(normalized_rewards[t:]) for t in range(len(normalized_rewards))])

#         #     for obs, action, old_log, adv in zip(traj['states'], traj['actions'], traj['log_probs'], advantages):
#         #         all_observations.append(obs)
#         #         all_actions.append(action)
#         #         all_old_log_probs.append(old_log)
#         #         all_advantages.append(adv)
        
#         # # Convert lists to tensors
#         # obs_tensor = torch.tensor(np.array(all_observations), dtype=torch.float32).to(device)
#         # action_tensor = torch.tensor(np.array(all_actions), dtype=torch.float32).to(device)
#         # old_log_tensor = torch.tensor(np.array(all_old_log_probs), dtype=torch.float32).to(device)
#         # adv_tensor = torch.tensor(np.array(all_advantages), dtype=torch.float32).to(device)
        
#         # # Forward pass through policy network
#         # mean, std = policy(obs_tensor)
#         # if torch.isnan(mean).any() or torch.isnan(std).any():
#         #     mean = torch.nan_to_num(mean, nan=0.0)
#         #     std = torch.nan_to_num(std, nan=1.0).clamp(min=epsilon)
#         # else:
#         #     std = std.clamp(min=epsilon)
#         # dist = Normal(mean, std)
#         # new_log_probs = dist.log_prob(action_tensor).sum(dim=-1)
        
#         # # Compute policy ratio and clipped surrogate objective
#         # ratios = torch.exp(new_log_probs - old_log_tensor)
#         # clipped_ratios = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps)
#         # surr_losses = -torch.min(ratios * adv_tensor, clipped_ratios * adv_tensor)
        
#         #! important: We do not need to compute the KL divergence here
#         # Calculate KL divergence
#         ##kl = approx_kl_divergence(new_log_probs, old_log_tensor, action_mask)
        
#         # Combine surrogate loss and KL penalty
#         losses = surr_losses ## + self.kl_weight * kl
        
#         # Apply masking if needed
#         if action_mask is not None:
#             return masked_mean(losses, action_mask)
#         else:
#             return losses.mean()