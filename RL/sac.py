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
            self.alpha = alpha
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
        # sample from replay buffer
        state, action, reward, next_state, mask = memory.sample(batch_size=batch_size)
        state, action, reward, next_state, mask = map(lambda x: torch.FloatTensor(x).to(self.device), 
                                                      [state, action, reward, next_state, mask])
        reward, mask = reward.unsqueeze(-1), mask.unsqueeze(-1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward + mask * self.gamma * (min_qf_next_target) # Q(s', a') : TD target
        
        # Critics update
        qf1, qf2 = self.critic(state, action) # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = self.value_criterion(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = self.value_criterion(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        critic_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Policy update
        new_action, log_pi, _ = self.actor.sample(state)

        qf1_pi, qf2_pi = self.critic(state, new_action)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # update target networks
        if updates % self.target_update_interval == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()
    
    def save_model(self, dir, name):
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (dir, name))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (dir, name))
        torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pth' % (dir, name))
        torch.save(self.policy_optim.state_dict(), '%s/%s_policy_optim.pth' % (dir, name))
        torch.save(self.critic_optim.state_dict(), '%s/%s_critic_optim.pth' % (dir, name))
    
    def load_model(self, dir, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (dir, name),  map_location= self.device))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (dir, name), map_location= self.device))
        # self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pth' % (dir, name)))
        # self.policy_optim.load_state_dict(torch.load('%s/%s_policy_optim.pth' % (dir, name)))
        # self.critic_optim.load_state_dict(torch.load('%s/%s_critic_optim.pth' % (dir, name)))
        self.actor.eval()
        self.critic.eval()
        # self.critic_target.eval()  
        # self.actor.to(self.device)
        # self.critic.to(self.device)
        # self.critic_target.to(self.device)
        # self.policy_optim.to(self.device)
        # self.critic_optim.to(self.device)
