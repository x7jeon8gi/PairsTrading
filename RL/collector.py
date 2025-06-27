import torch
import torch.multiprocessing as mp
import time
from env import TradingEnvironment
from sac import SAC
import numpy as np

def setup_collector_process(env_args, agent_args, seed):
    """
    Helper function to initialize environment and agent for a collector process.
    """
    # 각 프로세스에 대해 다른 시드를 설정하여 다양한 탐색을 유도
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = TradingEnvironment(**env_args)
    # Collector는 정책 네트워크만 필요로 할 수 있지만, 
    # action 선택을 위해 전체 agent를 초기화하는 것이 간단할 수 있음
    agent = SAC(**agent_args) 
    return env, agent

class DataCollector(mp.Process):
    """
    환경과 상호작용하여 경험 데이터를 수집하고 중앙 큐로 보내는 데이터 수집기 프로세스.
    """
    def __init__(self, env_args, agent_args, experience_queue, policy_queue, seed, collector_id=0):
        super(DataCollector, self).__init__()
        self.env_args = env_args
        self.agent_args = agent_args
        self.experience_queue = experience_queue
        self.policy_queue = policy_queue
        self.seed = seed
        self.collector_id = collector_id
        self.daemon = True # 메인 프로세스 종료 시 함께 종료

    def run(self):
        """프로세스 실행 시 호출되는 메인 메소드."""
        print(f"[Collector {self.collector_id}] Starting...")
        env, agent = setup_collector_process(self.env_args, self.agent_args, self.seed)
        
        state = env.reset()
        episode_reward = 0
        episode_steps = 0

        while True:
            # 1. 최신 정책 확인 및 업데이트
            if not self.policy_queue.empty():
                new_policy_state_dict = self.policy_queue.get()
                agent.actor.load_state_dict(new_policy_state_dict)
            
            # 2. 환경과 상호작용
            action, _ = agent.select_action(state, evaluate=False)
            next_state, reward, done = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # 3. 경험 데이터를 큐에 추가
            # 텐서가 아닌 numpy 배열로 보내는 것이 일반적으로 더 효율적
            experience = (state, action, reward, next_state, float(done))
            self.experience_queue.put(experience)
            
            # 4. 에피소드 종료 시 리셋
            if done:
                print(f"[Collector {self.collector_id}] Episode finished. Reward: {episode_reward:.2f}, Steps: {episode_steps}")
                state = env.reset()
                episode_reward = 0
                episode_steps = 0
            else:
                state = next_state


class TrajectoryCollector(mp.Process):
    """
    GRPO용 trajectory 수집기 - 전체 에피소드를 수집하여 큐로 보냄
    """
    def __init__(self, env_args, agent_args, trajectory_queue, policy_queue, seed, collector_id=0, max_steps=200):
        super(TrajectoryCollector, self).__init__()
        self.env_args = env_args
        self.agent_args = agent_args
        self.trajectory_queue = trajectory_queue
        self.policy_queue = policy_queue
        self.seed = seed
        self.collector_id = collector_id
        self.max_steps = max_steps
        self.daemon = True

    def run(self):
        """프로세스 실행 시 호출되는 메인 메소드."""
        print(f"[TrajectoryCollector {self.collector_id}] Starting...")
        
        # GRPO agent 초기화
        from grpo import GRPOAgent
        
        # 환경 초기화
        env, _ = setup_collector_process(self.env_args, self.agent_args, self.seed)
        
        # GRPO 에이전트만 초기화 (policy만 필요)
        state_dim = self.agent_args['num_inputs']
        action_dim = self.agent_args['num_action']
        hidden_sizes = self.agent_args['hidden_size']
        action_space = self.agent_args['action_space']
        lr = self.agent_args.get("lr", 0.0003)
        clip_epsilon = self.agent_args.get("clip_epsilon", 0.2)
        group_size = self.agent_args.get("group_size", 5)
        kl_weight = self.agent_args.get("kl_weight", 0.1)
        
        agent = GRPOAgent(state_dim, action_dim, hidden_sizes, action_space, 
                         lr=lr, clip_epsilon=clip_epsilon, group_size=group_size, kl_weight=kl_weight)

        while True:
            # 1. 최신 정책 확인 및 업데이트
            if not self.policy_queue.empty():
                new_policy_state_dict = self.policy_queue.get()
                agent.old_policy.load_state_dict(new_policy_state_dict)
                # 샘플링용 정책도 업데이트
                agent.policy.load_state_dict(new_policy_state_dict)

            # 2. 전체 trajectory 수집
            states, actions, log_probs, rewards = [], [], [], []
            state = env.reset()
            done = False
            steps = 0
            
            while not done and steps < self.max_steps:
                action, log_prob = agent.select_action(state, evaluate=False)
                next_state, reward, done = env.step(action)

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)

                state = next_state
                steps += 1

            # 3. trajectory를 큐에 추가
            trajectory = {
                'states': np.array(states),
                'actions': np.array(actions),
                'log_probs': np.array(log_probs),
                'rewards': np.array(rewards)
            }
            
            episode_reward = sum(rewards)
            print(f"[TrajectoryCollector {self.collector_id}] Episode finished. Reward: {episode_reward:.2f}, Steps: {steps}")
            
            try:
                self.trajectory_queue.put(trajectory, timeout=1)
            except:
                # 큐가 가득 찬 경우 패스
                pass
