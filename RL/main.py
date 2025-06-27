# 경로 호환성 확보할 것

from env import TradingEnvironment
from sac import SAC
from replay_memory import ReplayMemory
import torch
import argparse
import datetime
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import yaml
import os
from tqdm import tqdm
import wandb
import sys
import json
import time
from grpo import GRPOAgent
from ppo import PPOAgent
from utils.seed import seed_everything
from utils import calculate_metrics
from utils.metrics import calculate_financial_metrics_monthly
import logging
from collector import DataCollector
import torch.multiprocessing as mp

# Set the start method for multiprocessing
# This must be done at the very beginning, right after imports
# and before any other multiprocessing-related code is executed.
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass


def override_config(config, args):
    args_dict = vars(args)
    for key, value in args_dict.items():
        if value is not None:
            if key.startswith('env_args_'):
                # env_args 관련 인자
                sub_key = key.replace('env_args_', '')
                config['env_args'][sub_key] = value
            elif key.startswith('agent_args_'):
                # agent_args 관련 인자
                sub_key = key.replace('agent_args_', '')
                if sub_key == 'action_space':
                    # 문자열을 리스트로 변환
                    config['agent_args'][sub_key] = [float(x) for x in value.split(',')]
                else:
                    config['agent_args'][sub_key] = value
            else:
                # 최상위 인자
                config[key] = value
    return config

def set_logger(seed, run_name, save_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    if not os.path.exists(f'{save_dir}/log'):
        os.makedirs(f'{save_dir}/log')
    fh = logging.FileHandler(filename=f'{save_dir}/log/logging_{seed}_{run_name}.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

class Trainer(object):
    def __init__(self,
                 batch_size,
                 start_steps, # Number of steps for which the model picks random actions
                 n_steps,     # Maximum number of steps
                 save_name,
                 device,
                 seed,
                 env_args,   # returns_data, clusters_dir, start_month, end_month, reward_scale, outlier_filter
                 agent_args, # num_inputs, action_space, gamma, tau, alpha, target_update_interval, hidden_size, lr
                 replay_size,
                 update_per_step,
                 logger,
                 use_per=False,
                 per_alpha=0.6,
                 per_beta=0.4,
                 eval_interval=10,
                 eval_episodes=10,
                 num_collectors=2):

        #wandb.init(project='SAC_PairsTrading', name=save_name, config=config)

        self.batch_size = batch_size
        self.start_steps = start_steps
        self.n_steps = n_steps 
        self.save_name = save_name
        self.device = device
        self.seed = seed
        self.update_per_step = update_per_step
        self.env_args = env_args
        self.agent_args = agent_args
        self.num_actions = agent_args['num_action']
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.use_per = use_per
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.logger = logger
        self.num_collectors = num_collectors

        # ! 환경 초기화 -> Collector로 이동
        # self.env    = TradingEnvironment(**env_args)

        # ! 에이전트 초기화 (Learner)
        agent_type = agent_args.get("agent_type", "SAC")
        if agent_type == "SAC":
            from sac import SAC
            self.agent = SAC(**agent_args)
        elif agent_type == "PPO":
            from ppo import PPOAgent
            self.agent = PPOAgent(**agent_args)
        elif agent_type == "GRPO":
            # GRPOAgent expects state_dim, action_dim, hidden_sizes, etc.
            state_dim = agent_args['num_inputs']
            action_dim = agent_args['num_action']
            hidden_sizes = agent_args['hidden_size']  # should be a list, e.g., [64,64,64]
            action_space = agent_args['action_space']
            # 다른 하이퍼파라미터들도 agent_args에서 가져옴
            lr = agent_args.get("lr", 0.0003)
            clip_epsilon = agent_args.get("clip_epsilon", 0.2)
            group_size = agent_args.get("group_size", 5)
            kl_weight = agent_args.get("kl_weight", 0.1)
            self.agent = GRPOAgent(state_dim, 
                                   action_dim, 
                                   hidden_sizes, 
                                   action_space, 
                                   lr=lr, 
                                   clip_epsilon=clip_epsilon, 
                                   group_size=group_size,
                                   kl_weight = kl_weight)
        else:
            raise ValueError("Unknown agent_type specified in agent_args")

        if self.use_per:
            from replay_memory import PrioritizedReplayMemory
            self.memory = PrioritizedReplayMemory(replay_size, seed, alpha=per_alpha, beta=per_beta)
            self.logger.info(f"🔥 Using Prioritized Experience Replay (alpha={per_alpha}, beta={per_beta})")
            print(f"🔥 PER ENABLED: Memory type = {type(self.memory).__name__}")
        else:
            from replay_memory import ReplayMemory
            self.memory = ReplayMemory(replay_size, seed)
            self.logger.info("📝 Using standard Replay Memory")
            print(f"📝 STANDARD MEMORY: Memory type = {type(self.memory).__name__}")
            
        self.max_avg_reward = -float('inf')

        # Create model directory if it doesn't exist
        os.makedirs('./res/RL/models', exist_ok=True)
        os.makedirs('./res/RL/rewards', exist_ok=True)
        
        # --- 비동기 파이프라인 설정 ---
        self.experience_queue = mp.Queue(maxsize=1000)
        self.policy_queue = mp.Queue(maxsize=self.num_collectors)

        self.collectors = []
        for i in range(self.num_collectors):
            collector_seed = self.seed + i
            collector = DataCollector(
                self.env_args,
                self.agent_args,
                self.experience_queue,
                self.policy_queue,
                collector_seed,
                collector_id=i
            )
            collector.start()
            self.collectors.append(collector)
            self.logger.info(f"DataCollector {i} started with seed {collector_seed}.")


    def train_offline(self):
        """Offline training loop for SAC (replay memory based) using async data collection."""
        total_numsteps = 0
        updates = 0
        rewards = []
        progress_bar = tqdm(total=self.n_steps, desc="Training Progress")

        # 초기 정책을 Collector들에게 한 번 브로드캐스트
        policy_state_dict = {k: v.cpu() for k, v in self.agent.actor.state_dict().items()}
        for _ in range(self.num_collectors):
            self.policy_queue.put(policy_state_dict)

        while updates < self.n_steps:
            # 1. 경험 데이터 수집 및 리플레이 버퍼 채우기
            try:
                state, action, reward, next_state, done = self.experience_queue.get(timeout=1)
                self.memory.push(state, action, reward, next_state, done)
                total_numsteps += 1
                progress_bar.update(1)
            except Exception as e:
                # 큐가 비어있으면 계속 학습 진행
                pass

            # 2. 리플레이 버퍼가 충분히 차면 학습 시작
            if len(self.memory) > self.batch_size:
                for _ in range(self.update_per_step):
                    critic_1_loss, critic_2_loss, policy_loss, alpha_loss = self.agent.update_parameters(self.memory, self.batch_size, updates)
                    updates += 1

                    # 주기적으로 Collector들에게 최신 정책 전송
                    if updates % 200 == 0: 
                        policy_state_dict = {k: v.cpu() for k, v in self.agent.actor.state_dict().items()}
                        # 큐가 비어있을 때만 넣어서 너무 많이 쌓이지 않도록 함
                        if self.policy_queue.empty():
                            for _ in range(self.num_collectors):
                                self.policy_queue.put(policy_state_dict)

                    if updates % 1000 == 0:  # 1000번마다 출력으로 변경
                        log_msg = (f"Updates: {updates} | C1 Loss: {critic_1_loss:.4f} | "
                                   f"C2 Loss: {critic_2_loss:.4f} | Policy Loss: {policy_loss:.4f} | "
                                   f"Alpha Loss: {alpha_loss:.4f} | Alpha: {self.agent.alpha.item():.4f}")
                        self.logger.info(log_msg)
                        print(f"📈 Training Progress - Updates: {updates} | C1: {critic_1_loss:.2f} | C2: {critic_2_loss:.2f} | Policy: {policy_loss:.2f}")

            # 3. 주기적인 평가
            if updates > 0 and updates % self.eval_interval == 0: # 수정: 1000 곱하기 제거
                avg_reward = self._evaluate_agent(self.eval_episodes)
                rewards.append(avg_reward)
                self.logger.info(f'Evaluation at Updates: {updates} | Total Steps: {total_numsteps} | Average Reward: {avg_reward:.4f}')

                if avg_reward > self.max_avg_reward:
                    self.max_avg_reward = avg_reward
                    self.agent.save_model('./res/RL/models', self.save_name)
                    self.logger.info(f'********** New best model saved with avg reward: {self.max_avg_reward:.4f} **********')

            if updates >= self.n_steps:
                break

        progress_bar.close()
        self._save_reward_plot(rewards, xlabel=f'Evaluation (every {self.eval_interval} updates)')
        self.logger.info("Offline training completed.")

        # 학습 종료 후 Collector 프로세스 정리
        for collector in self.collectors:
            collector.terminate()
            collector.join()
        self.logger.info("All DataCollectors terminated.")


    def train_online(self):
        """Online training loop for PPO/GRPO (group-based updates)"""
        total_numsteps = 0
        rewards = []
        group_trajectories = []
        episode_num = 0
        group_size = self.agent_args.get("group_size", 5)
        progress_bar = tqdm(total=self.n_steps, desc="Training Progress")
        episode_rewards_history = [] # List to store episode rewards

        while total_numsteps < self.n_steps:
            # collect trajectories from old policy
            trajectory = self.agent.collect_trajectory(self.env, max_steps=200) # states, actions, log_porbs, rewards
            group_trajectories.append(trajectory)

            episode_reward = sum(trajectory['rewards'])
            episode_rewards_history.append(episode_reward) # Store episode reward
            total_numsteps += len(trajectory['states'])
            episode_num += 1

            if len(group_trajectories) >= group_size:
                update_loss = self.agent.grpo_update(group_trajectories, n_iterations=20) # n_iterations: 국밥
                self.logger.info(f'Online Update at Episode: {episode_num} | Loss: {update_loss:.4f}')
                group_trajectories = []  # reset group

            self.logger.info(f'Episode: {episode_num} | Total Steps: {total_numsteps} | Episode Reward: {episode_reward:.4f}')

            if episode_num % self.eval_interval == 0:
                avg_reward = self._evaluate_agent(self.eval_episodes)
                rewards.append(avg_reward)
                self.logger.info(f'Evaluation at Episode: {episode_num} | Total Steps: {total_numsteps} | Average Reward: {avg_reward:.4f}')

                if avg_reward > self.max_avg_reward:
                    self.max_avg_reward = avg_reward
                    self.agent.save_model('./res/RL/models', self.save_name)
                    self.logger.info(f'********** New best model saved with avg reward: {self.max_avg_reward:.4f} **********')

            progress_bar.update(len(trajectory['states']))
            
            # Break loop if n_steps reached
            if total_numsteps >= self.n_steps:
                break

        progress_bar.close()
        if group_trajectories: # Process remaining trajectories if any
            update_loss = self.agent.grpo_update(group_trajectories, n_iterations=20)
            self.logger.info(f'Final Online Update | Loss: {update_loss:.4f}')

        # Save the plot of episode rewards
        self._save_reward_plot(episode_rewards_history, xlabel='Episode')
        self.logger.info("Online training completed.")

    def train(self):
        """Main train method selecting offline or online training based on agent type."""
        agent_type = self.agent_args.get("agent_type", "SAC")
        if agent_type == "SAC":
            self.train_offline()
        else:  # For PPO and GRPO
            self.train_online()

    @torch.no_grad()
    def _evaluate_agent(self, episodes):
        """Evaluate the agent's performance without affecting training - optimized version"""

        # 평가용 환경이 없으면 생성, 있으면 재사용
        if not hasattr(self, 'eval_env'):
            self.eval_env = TradingEnvironment(**self.env_args)
        
        episode_rewards = []
        
        # Use evaluation mode for the agent (no exploration)
        for ep in range(episodes):
            state = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, done  = self.eval_env.step(action)
                episode_reward += reward
                state = next_state
            
            episode_rewards.append(episode_reward)
            
            # Early stopping: 충분한 에피소드(최소 10개)를 실행한 후 
            # 분산이 작으면 조기 종료하여 평가 시간 단축
            if ep >= 9 and ep % 5 == 4:  # 10, 15, 20, ... 에피소드마다 체크
                rewards_array = np.array(episode_rewards)
                std_error = np.std(rewards_array) / np.sqrt(len(rewards_array))
                if std_error < 0.1:  # 표준 오차가 0.1 미만이면 충분히 안정적
                    break
    
        # Compute average reward
        avg_reward = np.mean(episode_rewards)
        return avg_reward

    def _save_reward_plot(self, rewards, xlabel=None):
        """Save a plot of the rewards"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.plot(rewards)
        # Use provided xlabel if available, otherwise use default based on eval_interval
        if xlabel is None:
            xlabel = 'Evaluation (every {} episodes)'.format(self.eval_interval)
        plt.xlabel(xlabel)
        plt.ylabel('Average Reward' if xlabel.startswith('Evaluation') else 'Episode Reward') # Adjust ylabel based on xlabel
        plt.title('Training Progress')
        plt.savefig(f'./res/RL/rewards/{self.save_name}.png')
        plt.close()

    def inference(self, model_dir, model_name):
        """
        강화학습 모델을 로드하고 전체 기간에 대해 추론을 실행하여 결과를 반환합니다.
        
        Args:
            model_dir: 모델이 저장된 디렉토리 경로
            model_name: 모델 파일 이름
            
        Returns:
            tuple: 총 보상, 스텝별 보상, 액션 리스트, 포트폴리오 가치 리스트, 상태 정보
        """
        # 모델 로드
        try:
            self.agent.load_model(model_dir, model_name)
            self.logger.info(f"모델을 성공적으로 로드했습니다: {model_dir}/{model_name}")
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise e
            
        # 환경 설정 업데이트
        self.env_args['start_month'] = '2006-01'
        self.env_args['end_month'] = '2023-12'
        self.env_args['new_data'] = True
        self.env = TradingEnvironment(**self.env_args, sp500_data='./data/sp500.csv')
        
        self.logger.info(f"추론 시작: 기간 {self.env_args['start_month']}~{self.env_args['end_month']}")

        # 환경 초기화
        state = self.env.reset()
        done = False
        total_reward = 0
        
        # 결과 저장 리스트
        actions = []
        rewards = []
        portfolio_values = []
        states = []
        action_info = []  # 액션의 상세 정보 저장
        month_dates = []  # 월별 날짜 정보
        
        # 추론 실행
        step_count = 0
        while not done:
            # 현재 날짜 저장
            if hasattr(self.env, 'current_month'):
                month_dates.append(self.env.current_month)
            
            # 액션 선택 및 실행
            action = self.agent.select_action(state, evaluate=True)
            next_state, reward, done = self.env.step(action)
            
            # 결과 저장
            actions.append(action)
            rewards.append(reward)
            portfolio_values.append(self.env.current_portfolio_value)
            states.append(state)
            
            # 액션 상세 정보 기록
            action_detail = {
                'step': step_count,
                'threshold': float(action[0]),
                'outlier_filter': float(action[1]),
                'reward': float(reward),
                'portfolio_value': float(self.env.current_portfolio_value)
            }
            
            if hasattr(self.env, 'current_month'):
                action_detail['month'] = self.env.current_month
                
            action_info.append(action_detail)
            
            # 상태 업데이트
            total_reward += reward
            state = next_state
            step_count += 1
            
            # 로깅
            if step_count % 10 == 0:
                self.logger.info(f"스텝 {step_count}: 보상 {reward:.4f}, 포트폴리오 가치: {self.env.current_portfolio_value:.4f}")
        
        # 최종 결과 요약
        self.logger.info(f"추론 완료: 총 {step_count}개 스텝, 총 보상: {total_reward:.4f}, 최종 포트폴리오 가치: {self.env.current_portfolio_value:.4f}")
        
        # 결과 시각화 및 분석
        # self._visualize_results(portfolio_values, rewards, actions, month_dates)
        
        # 액션 정보를 DataFrame으로 변환하여 저장
        action_df = pd.DataFrame(action_info)
        if 'month' in action_df.columns:
            action_df.set_index('month', inplace=True)
        
        os.makedirs('./res/RL/results', exist_ok=True)
        action_df.to_csv(f'./res/RL/results/action_details_{model_name}.csv')
        self.logger.info(f"액션 상세 정보 저장 완료: ./res/RL/results/action_details_{model_name}.csv")
        
        # Convert RL agent's portfolio values (cumulative log returns) into a series
        cum_log_returns_rl = pd.Series(portfolio_values)
        
        # 날짜 인덱스가 존재하면 이를 활용
        if month_dates:
            cum_log_returns_rl.index = month_dates
            
        # Compute the per-step log returns for RL agent
        log_returns_rl = cum_log_returns_rl.diff().fillna(0)  # Use 0 for the first step's diff
        
        # 최종 누적 수익률 계산
        cumulative_return_rl = np.expm1(portfolio_values[-1]) if portfolio_values else 0
        self.logger.info(f"RL 모델 최종 누적 수익률: {cumulative_return_rl:.4f} ({cumulative_return_rl*100:.2f}%)")
        
        # 반환 값 추가: states를 포함하여 더 자세한 분석 가능하도록 함
        return total_reward, rewards, actions, portfolio_values, states
        
    def _visualize_results(self, portfolio_values, rewards, actions, dates=None):
        """
        RL 모델의 결과를 시각화합니다.
        
        Args:
            portfolio_values: 포트폴리오 가치 리스트
            rewards: 보상 리스트
            actions: 액션 리스트
            dates: 날짜 리스트 (선택 사항)
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import numpy as np
        from datetime import datetime
        
        # 저장 디렉토리 생성
        os.makedirs('./res/RL/plots', exist_ok=True)
        
        # 1. 포트폴리오 가치 (누적 수익률) 그래프
        plt.figure(figsize=(12, 6))
        
        if dates and len(dates) == len(portfolio_values):
            # 날짜 형식으로 변환
            date_objects = [datetime.strptime(date, '%Y-%m') for date in dates]
            plt.plot(date_objects, portfolio_values, 'b-', linewidth=2)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        else:
            plt.plot(portfolio_values, 'b-', linewidth=2)
        
        plt.title('RL 모델의 포트폴리오 누적 로그 수익률')
        plt.xlabel('월')
        plt.ylabel('누적 로그 수익률')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'./res/RL/plots/{self.save_name}_portfolio_value.png')
        plt.close()
        
        # 2. 월별 수익률 그래프 (로그 수익률의 차분)
        monthly_returns = np.diff(portfolio_values, prepend=0)
        
        plt.figure(figsize=(12, 6))
        
        if dates and len(dates) == len(monthly_returns):
            date_objects = [datetime.strptime(date, '%Y-%m') for date in dates]
            # 플러스/마이너스에 따라 막대 색상 변경
            colors = ['red' if x < 0 else 'green' for x in monthly_returns]
            plt.bar(date_objects, monthly_returns, color=colors, width=20)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        else:
            colors = ['red' if x < 0 else 'green' for x in monthly_returns]
            plt.bar(range(len(monthly_returns)), monthly_returns, color=colors)
        
        plt.title('RL 모델의 월별 로그 수익률')
        plt.xlabel('월')
        plt.ylabel('월별 로그 수익률')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f'./res/RL/plots/{self.save_name}_monthly_returns.png')
        plt.close()
        
        # 3. 액션 분포 분석
        if actions and len(actions) > 0:
            # 액션 추출
            actions_array = np.array(actions)
            thresholds = actions_array[:, 0]
            outlier_filters = actions_array[:, 1]
            
            # 3-1. 임계값(threshold) 분포
            plt.figure(figsize=(10, 5))
            plt.plot(thresholds, 'r-', label='Threshold')
            plt.title('RL 모델의 임계값(threshold) 변화')
            plt.xlabel('스텝')
            plt.ylabel('임계값')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'./res/RL/plots/{self.save_name}_thresholds.png')
            plt.close()
            
            # 3-2. 아웃라이어 필터 분포
            plt.figure(figsize=(10, 5))
            plt.plot(outlier_filters, 'g-', label='Outlier Filter')
            plt.title('RL 모델의 아웃라이어 필터 변화')
            plt.xlabel('스텝')
            plt.ylabel('아웃라이어 필터 값')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'./res/RL/plots/{self.save_name}_outlier_filters.png')
            plt.close()
            
            # 3-3. 임계값과 아웃라이어 필터의 관계 (산점도)
            plt.figure(figsize=(8, 8))
            plt.scatter(thresholds, outlier_filters, alpha=0.6, c=range(len(thresholds)), cmap='viridis')
            plt.colorbar(label='Step')
            plt.title('RL 모델의 임계값 vs 아웃라이어 필터')
            plt.xlabel('임계값(threshold)')
            plt.ylabel('아웃라이어 필터')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'./res/RL/plots/{self.save_name}_action_scatter.png')
            plt.close()
        
        # 4. 월별 보상 그래프
        plt.figure(figsize=(12, 6))
        
        if dates and len(dates) == len(rewards):
            date_objects = [datetime.strptime(date, '%Y-%m') for date in dates]
            colors = ['red' if x < 0 else 'blue' for x in rewards]
            plt.bar(date_objects, rewards, color=colors, width=20)
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        else:
            colors = ['red' if x < 0 else 'blue' for x in rewards]
            plt.bar(range(len(rewards)), rewards, color=colors)
        
        plt.title('RL 모델의 월별 보상')
        plt.xlabel('월')
        plt.ylabel('보상')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f'./res/RL/plots/{self.save_name}_rewards.png')
        plt.close()
        
        self.logger.info(f"결과 시각화 완료: ./res/RL/plots/{self.save_name}_*.png")


if __name__ == "__main__":
    # * Config
    with open('./RL/SAC_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='G',
                        help='learning rate (default: 0.0001)')
    # parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
    #                     help='Temperature parameter α determines the relative importance of the entropy\
    #                             term against the reward (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='random seed (default: 42)')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_collectors', type=int, default=4, metavar='N',
                        help='number of parallel data collectors (default: 4)')
   

    # * Arguments
    # 기본적으로 Yaml을 따르나, argparse로 받은 인자가 있다면 해당 인자로 덮어씌움
    args = parser.parse_args()
    config = override_config(config, args)
    
    batch_size = config['batch_size']
    start_steps = config['start_steps']
    n_steps = config['n_steps']
    env_args = config['env_args']
    agent_args = config['agent_args']

    reward_method =  env_args['hard_reward']
    reward_scale = env_args['reward_scale']
    dynamic_gamma = env_args['dynamic_gamma']
    num_cluster = config['num_cluster']
    hidden = agent_args['hidden_size'][0]
    agent_type = agent_args['agent_type']
    kl_weight = agent_args['kl_weight']
    group_size = agent_args['group_size']
    num_layers = len(agent_args['hidden_size'])
    lr = agent_args['lr']
    
    # Generate a unique name for the run
    base_save_name = (
        f'{agent_type}{num_cluster}_{reward_method}_scale{reward_scale}_'
        f'dim{hidden}_g{dynamic_gamma}_layers{num_layers}_'
        f'gsize{group_size}_lr{lr}'
    )
    
    if config['save_name'] == 'automatic':
        save_name = base_save_name
    else:
        prefix = config['save_name']
        save_name = f'{prefix}_{base_save_name}'
    
    # env_args's cluster_dir fix
    env_args['clusters_dir'] = env_args['clusters_dir'] + f'{num_cluster}'
    seed = config['seed']
    replay_size = config['replay_size']
    update_per_step = config['update_per_step']

    logger = set_logger(seed=seed, run_name=save_name, save_dir='./res/RL')
    logger.info("Configuration is:\n" + json.dumps(config, indent=4))

    seed_everything(seed)
    trainer = Trainer(
                    batch_size=batch_size,
                    start_steps=start_steps,
                    n_steps=n_steps,
                    save_name=save_name,
                    device='cuda',
                    seed=seed,
                    env_args=env_args,
                    agent_args=agent_args,
                    update_per_step= update_per_step,
                    replay_size=replay_size,
                    logger=logger,
                    use_per=config.get('use_per', False),
                    per_alpha=config.get('per_alpha', 0.6),
                    per_beta=config.get('per_beta', 0.4),
                    eval_interval=config.get('eval_interval', 10),  # YAML에서 읽어오기
                    eval_episodes=config.get('eval_episodes', 10), # YAML에서 읽어오기
                    num_collectors=config.get('num_collectors', 4)
                    )
    
    trainer.train()
    total_reward, rewards, actions, portfolio_values, states = trainer.inference('./res/RL/models', save_name)
    # final logging for its name
    # save actions
    action_df = pd.DataFrame(actions)
    action_df.to_csv(f'./res/RL/results/action_details_{save_name}.csv')
    
    logger.info(f"Name of this run: {save_name}")