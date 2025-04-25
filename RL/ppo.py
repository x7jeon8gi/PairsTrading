import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# 하이퍼파라미터 상수들
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Actor-Critic 네트워크 (공유 피처 추출기와 두 개의 헤드: policy, value)
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        # 공통 피처 추출기
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # Policy head: 평균을 출력, 로그표준편차는 파라미터로 학습
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        # Value head: 상태의 가치 평가
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        features = self.shared(state)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std.expand_as(mean).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        value = self.critic(features)
        return mean, std, value



class PPOAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_size=64,
                 lr=3e-4,
                 gamma=0.99,
                 lam=0.95,
                 clip_epsilon=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = ActorCritic(state_dim, action_dim, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state, evaluate=False):
        # state is a numpy array
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std, value = self.ac(state)
        dist = Normal(mean, std)
        if evaluate:
            action = mean
            log_prob = dist.log_prob(mean).sum(dim=-1)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().detach().numpy()[0], log_prob.cpu().detach().numpy()[0], value.cpu().detach().numpy()[0]

    def evaluate_actions(self, states, actions):
        mean, std, values = self.ac(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_probs, entropy, values

    def compute_gae(self, rewards, masks, values):
        """
        Compute Generalized Advantage Estimation (GAE).
        rewards: list of rewards per timestep
        masks: list where 1 indicates non-terminal, 0 terminal
        values: list of critic value estimates
        """
        gae = 0
        returns = []
        # values 리스트의 마지막 값에 0을 추가
        values = values + [0]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.lam * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def collect_trajectory(self, env, max_steps):
        """
        환경과 상호작용하여 단일 궤적(trajectory)을 수집합니다.
        returns: 상태, 행동, 로그 확률, 보상, 마스크 및 가치 추정값을 포함하는 딕셔너리
        """
        states, actions, log_probs, rewards, masks, values = [], [], [], [], [], []
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action, log_prob, value = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # 데이터 저장
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            masks.append(1.0 - float(done))
            values.append(value)
            
            state = next_state
            steps += 1
            
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'rewards': np.array(rewards),
            'masks': np.array(masks),
            'values': np.array(values)
        }

    def train(self, env, max_steps_per_episode=1000, n_episodes=10, update_epochs=10, batch_size=64):
        """
        PPO 에이전트 학습을 위한 메인 훈련 루프
        """
        total_rewards = []
        
        for episode in range(n_episodes):
            # 궤적 수집
            trajectory = self.collect_trajectory(env, max_steps_per_episode)
            episode_reward = np.sum(trajectory['rewards'])
            total_rewards.append(episode_reward)
            
            # GAE 및 리턴 계산
            returns = self.compute_gae(
                trajectory['rewards'], 
                trajectory['masks'], 
                trajectory['values']
            )
            
            # 어드밴티지 계산
            advantages = np.array(returns) - trajectory['values']
            
            # PPO 업데이트 수행
            self.ppo_update(
                trajectory['states'], 
                trajectory['actions'], 
                trajectory['log_probs'], 
                returns, 
                advantages, 
                epochs=update_epochs, 
                batch_size=batch_size
            )
            
            print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")
            
        return total_rewards

    def ppo_update(self, states, actions, log_probs_old, returns, advantages, epochs=10, batch_size=64):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        dataset_size = states.size(0)
        
        for _ in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                # Normalize advantages
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
                new_log_probs, entropy, values = self.evaluate_actions(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_log_probs_old.unsqueeze(1))
                surr1 = ratio * batch_advantages.unsqueeze(1)
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages.unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns.unsqueeze(1))
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save_model(self, dir, name):
        import os
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(self.ac.state_dict(), f"{dir}/{name}_ac.pth")
        torch.save(self.optimizer.state_dict(), f"{dir}/{name}_optim.pth")

    def load_model(self, dir, name):
        self.ac.load_state_dict(torch.load(f"{dir}/{name}_ac.pth", map_location=self.device))
        self.optimizer.load_state_dict(torch.load(f"{dir}/{name}_optim.pth", map_location=self.device))
        self.ac.eval()
