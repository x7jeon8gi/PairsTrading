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
