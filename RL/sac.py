import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import GaussianPolicy, QNetwork
from gym import spaces
import numpy as np

class SAC(object):

    def __init__(self, 
                 num_inputs,
                 num_action, 
                 action_space, 
                 gamma,
                 tau, 
                 alpha, 
                 target_update_interval,
                 hidden_size,
                 agent_type,
                 lr,
                 **kwargs):  # 추가 매개변수 허용
            
            self.gamma = gamma
            self.tau = tau
            self.alpha_value = alpha  # Keep original alpha for cases where automatic tuning is off
            self.agent_type = agent_type
            self.action_space = spaces.Box(low= np.array(action_space[0]), high= np.array(action_space[1]), dtype=np.float32)
            self.target_update_interval = target_update_interval
            self.num_inputs = num_inputs
            self.num_action = num_action

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
            self.critic = QNetwork(num_inputs, num_action, hidden_size).to(self.device)
            self.critic_optim = Adam(self.critic.parameters(), lr=lr)
    
            self.critic_target = QNetwork(num_inputs, num_action, hidden_size).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict()) # copy parameters
    
            self.actor = GaussianPolicy(num_inputs, num_action, hidden_size, self.action_space).to(self.device)
            self.policy_optim = Adam(self.actor.parameters(), lr=lr) 
    
            # Automatic Entropy Tunning
            self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp() # Alpha is a learned parameter
    
            self.critic_target.eval()
            self.value_criterion = F.mse_loss
            
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, log_prob, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()[0], log_prob.detach().cpu().numpy()[0]
        else:
            _, _, action_mean = self.actor.sample(state)
            return action_mean.detach().cpu().numpy()[0]
    
    def update_parameters(self, memory, batch_size, updates):
        # sample from prioritized replay buffer
        (state, action, reward, next_state, mask), idxs, is_weights = memory.sample(batch_size=batch_size)
        
        state, action, reward, next_state, mask, is_weights = map(
            lambda x: torch.FloatTensor(x).to(self.device),
            [state, action, reward, next_state, mask, is_weights]
        )
        reward, mask, is_weights = reward.unsqueeze(-1), mask.unsqueeze(-1), is_weights.unsqueeze(-1)


        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            
            # 🔧 Q-value 안정화 (클리핑)
            min_qf_next_target = torch.clamp(min_qf_next_target, -100.0, 100.0)
            reward = torch.clamp(reward, -10.0, 10.0)  # Reward clipping
            
            next_q_value = reward + mask * self.gamma * min_qf_next_target
        
        # Critics update
        qf1, qf2 = self.critic(state, action)
        
        # Calculate TD errors for PER priority update
        td_error1 = (qf1 - next_q_value).abs()
        td_error2 = (qf2 - next_q_value).abs()
        # Use the mean of the two TD errors for priority
        priorities = ((td_error1 + td_error2) / 2.0).detach().cpu().numpy()

        # Update priorities in the replay buffer
        memory.update_priorities(idxs, priorities.flatten())

        qf1_loss = (is_weights * F.mse_loss(qf1, next_q_value, reduction='none')).mean()
        qf2_loss = (is_weights * F.mse_loss(qf2, next_q_value, reduction='none')).mean()
        critic_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        # 🔧 Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        # Policy update
        new_action, log_pi, _ = self.actor.sample(state)

        qf1_pi, qf2_pi = self.critic(state, new_action)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        # 🔧 Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.policy_optim.step()

        # Alpha (Entropy term) update
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        # 🔧 Gradient clipping for alpha
        torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
        self.alpha_optim.step()
        
        # 🔧 Alpha 값 안정화 (너무 큰 값 방지)
        self.alpha = torch.clamp(self.log_alpha.exp(), min=1e-8, max=100.0)

        # update target networks
        if updates % self.target_update_interval == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()
    
    def save_model(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (dir, name))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (dir, name))
        torch.save(self.log_alpha, '%s/%s_log_alpha.pth' % (dir, name))
        torch.save(self.alpha_optim.state_dict(), '%s/%s_alpha_optim.pth' % (dir, name))

    def load_model(self, dir, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (dir, name),  map_location= self.device))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (dir, name), map_location= self.device))
        
        # Load log_alpha and its optimizer if they exist
        log_alpha_path = '%s/%s_log_alpha.pth' % (dir, name)
        if os.path.exists(log_alpha_path):
            self.log_alpha = torch.load(log_alpha_path, map_location=self.device)
        
        alpha_optim_path = '%s/%s_alpha_optim.pth' % (dir, name)
        if os.path.exists(alpha_optim_path):
            self.alpha_optim.load_state_dict(torch.load(alpha_optim_path, map_location=self.device))

        self.actor.eval()
        self.critic.eval()
