import numpy as np
import pandas as pd
import os
import sys
from glob import glob
from scipy.stats.mstats import winsorize
import logging
import copy
from datetime import datetime, timedelta
import gym
from gym.spaces import Box
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.trading_logic import calculate_positions, calculate_portfolio_returns, LONG_NAN_RETURN, SHORT_NAN_RETURN


long_nan_return = -0.25
short_nan_return = 0.00

class TradingEnvironment:
    """
    Designed for Soft Actor Critic (SAC) algorithm.
    Implements a pairs trading strategy with customizable reward functions.
    """
    def __init__(self, 
                 returns_data, 
                 clusters_dir, 
                 start_month='1990-01',
                 end_month = '1999-12', 
                 reward_scale=10,
                 new_data = None,
                 sp500_data = './data/sp500.csv',
                 hard_reward=False,
                 num_inputs = 10,
                 dynamic_gamma= 0.5,
                 use_winsorize=False
                 ):
        """
        Initialize the trading environment.
        
        Args:
            returns_data (str): Path to CSV file containing returns data.
            clusters_dir (str): Directory containing clustering results.
            start_month (str): Start month in format 'YYYY-MM'.
            end_month (str): End month in format 'YYYY-MM'.
            reward_scale (float): Scaling factor for rewards.
            new_data (str, optional): Path to new data if available.
            sp500_data (str): Path to S&P500 benchmark data.
            hard_reward (bool or str): Reward function type ('Markowitz', 'CVaR', 'Sharpe' or boolean).
            num_inputs (int): Number of state inputs for the RL agent.
        """

        # Define action and observation spaces
        self.action_space = Box(low=np.array([0.0, 0.0]), high=np.array([2.0, 1.0]), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        self._setup_reward_function(hard_reward, reward_scale)
        self._display_initialization_info(new_data, start_month, end_month)
        
        self.num_inputs = num_inputs
        self.use_winsorize = use_winsorize
        # for CVaR
        self.alpha = 0.05
        self.lambda_cvar = 1.0

        self.dynamic_gamma = dynamic_gamma

        # Load and preprocess returns data
        self.returns_data = self._load_returns_data(returns_data, start_month, end_month)
        
        # Load S&P500 data
        self.index_data = pd.read_csv(sp500_data, index_col=0)

        # Load clustering and probability data
        self._load_cluster_data(clusters_dir, start_month, end_month)
        
        self.current_step = 0
        self.consecutive_non_negative_steps = 0
    def _setup_reward_function(self, hard_reward, reward_scale):
        """Set up reward function parameters based on configuration."""
        self.hard_reward = hard_reward
        
        if isinstance(hard_reward, bool) and hard_reward:
            print(f"Hard reward is used. {hard_reward}")
            self.reward_scale = 1
            self.negative_reward = - np.abs(reward_scale)
        else:
            self.reward_scale = reward_scale
            
        print(f"** Reward Scale: {self.reward_scale} **")
    
    def _display_initialization_info(self, new_data, start_month, end_month):
        """Display initialization information."""
        if new_data is not None:
            print("New data is provided.")
            print(f"Start month: {start_month}, End month: {end_month}")

    def _load_returns_data(self, returns_data_path, start_month, end_month):
        """Load and preprocess returns data."""
        
        # 데이터가 pkl인지 csv인지 확인
        if returns_data_path.endswith('.pkl'):
            returns_data = pd.read_pickle(returns_data_path)
        else:
            returns_data = pd.read_csv(returns_data_path, index_col=0)
        # return_data 의 index를 int 로 변환
        returns_data.index = returns_data.index.astype(int)

        # Calculate end_month + 1 (for next month data)
        end_month_dt = datetime.strptime(end_month, '%Y-%m')
        end_month_1_dt = end_month_dt.replace(year=end_month_dt.year + 1 if end_month_dt.month == 12 else end_month_dt.year,
                                              month=1 if end_month_dt.month == 12 else end_month_dt.month + 1)
        end_month_1 = end_month_1_dt.strftime('%Y-%m')
        
        # Filter data from start_month to end_month + 1
        returns_data = returns_data[[col for col in returns_data.columns if start_month <= col <= end_month_1]]
        
        # Winsorize returns data
        if self.use_winsorize:
            filled_zeros = returns_data.fillna(0)
            winsorized_data = filled_zeros.apply(lambda x: x.clip(lower=-0.95, upper=1.5)) # 윈저화 값은 모든 실험에서 동일하게 유지됨.
            returns_data = winsorized_data.where(returns_data.notna())
        

        return returns_data

    def _load_cluster_data(self, clusters_dir, start_month, end_month):
        """Load clustering and probability data."""

        if clusters_dir is None:
            raise ValueError("Please provide a directory containing the monthly clustering results.")
        
        self.clusters_dir = clusters_dir
        self.prob_dir = clusters_dir.replace('predictions', 'prob')
        
        # Calculate end_month + 1 for filtering
        end_month_dt = datetime.strptime(end_month, '%Y-%m')
        end_month_1_dt = end_month_dt.replace(year=end_month_dt.year + 1 if end_month_dt.month == 12 else end_month_dt.year,
                                              month=1 if end_month_dt.month == 12 else end_month_dt.month + 1)
        end_month_1 = end_month_1_dt.strftime('%Y-%m')
        
        # Get sorted file lists
        self.prob_files = sorted(glob(self.prob_dir + '/*.csv'))
        self.cluster_files = sorted(glob(clusters_dir + '/*.csv'))
        
        # Filter files by date range
        self.prob_files = [file for file in self.prob_files 
                           if start_month <= file.split('/')[-1].split('.')[0] <= end_month_1]
        self.cluster_files = [file for file in self.cluster_files 
                              if start_month <= file.split('/')[-1].split('.')[0] <= end_month_1]
        
        # Preload all data for efficiency
        self._preload_all_data()

    def _preload_all_data(self):
        # 모든 클러스터와 확률 데이터를 미리 로드
        self.all_cluster_data = {}
        self.all_prob_data = {}
        for file_path in self.cluster_files:
            month = file_path.split('/')[-1].split('.')[0]
            self.all_cluster_data[month] = pd.read_csv(file_path, index_col=0)
        
        for file_path in self.prob_files:
            month = file_path.split('/')[-1].split('.')[0]
            self.all_prob_data[month] = pd.read_csv(file_path, index_col='firms')
            
    def reset(self):
        self.rolling_returns = []
        self.rolling_volatility = 0.0
        self.window_size = 6
        self.current_step = 0
        self.total_reward = 0
        self.current_portfolio_value = 1.0
        self.max_portfolio_value = 1.0
        self.prev_portfolio_value = self.current_portfolio_value
        self.consecutive_non_negative_steps = 0
        
        # 혁신적인 보상 함수를 위한 초기화
        self.rolling_spread_returns = []
        self.rolling_portfolio_returns = []
        self.rolling_sharpe_window = []
        self.risk_aversion = 0.5
        self.trade_efficiency_memory = []
        self.convergence_memory = []
        self.market_regime_memory = []
        
        return self._get_state()
    
    def _get_state(self):
        current_cluster_file = self.cluster_files[self.current_step]
        self.current_month = current_cluster_file.split('/')[-1].split('.')[0] 
        self.current_data = self.all_cluster_data[self.current_month]
        self.current_data = self.current_data.sort_values(by='MOM1', ascending=False)
        if self.current_step + 1 < len(self.cluster_files):
            self.next_month = self.cluster_files[self.current_step + 1].split('/')[-1].split('.')[0]

        sp500 = self.index_data.loc[self.current_month]
        num_assets = sp500['Number of Assets'] if not pd.isna(sp500['Number of Assets']) else 0
        sp_return = sp500['Monthly Returns'] if not pd.isna(sp500['Monthly Returns']) else 0
        sp_volatility = sp500['Monthly Volatility'] if not pd.isna(sp500['Monthly Volatility']) else 0
        
        # 안전한 로그 계산
        mom1_data = self.current_data['MOM1'].fillna(0)  # NaN을 0으로 처리
        log_returns = np.log1p(mom1_data.clip(lower=-0.99))  # -1 미만 값 방지
        
        avg_asset_return = log_returns.mean() if len(log_returns) > 0 else 0
        top_asset_return = log_returns.quantile(0.75) if len(log_returns) > 0 else 0
        bottom_asset_return = log_returns.quantile(0.25) if len(log_returns) > 0 else 0
        volatility = log_returns.std() if len(log_returns) > 0 else 0

        clusters = self.current_data['clusters']
        total_firms = len(clusters)
        cluster_counts = clusters.value_counts()
        n_clusters = len(cluster_counts)
        max_cluster_ratio = cluster_counts.max() / total_firms if total_firms > 0 else 0
        cluster_ratios = cluster_counts / total_firms if total_firms > 0 else pd.Series(dtype=float)
        
        # 안전한 엔트로피 계산
        if len(cluster_ratios) > 0:
            entropy = -(cluster_ratios * np.log(cluster_ratios + 1e-8)).sum()
        else:
            entropy = 0
            
        n_clusters_norm = min(n_clusters / 50.0, 1.0)  # 상한 설정
        entropy_norm = min(entropy / 50.0, 1.0)  # 상한 설정

        drawdown = (self.max_portfolio_value - self.current_portfolio_value) / self.max_portfolio_value
        
        state_vector = np.array([
            num_assets, sp_return, sp_volatility,
            avg_asset_return, top_asset_return, bottom_asset_return, volatility,
            n_clusters_norm, max_cluster_ratio, entropy_norm,
            drawdown,
        ], dtype=np.float32)

        # NaN 값을 완전히 제거하고 무한대 값도 처리
        state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 추가 안전장치: 값의 범위 제한
        state_vector = np.clip(state_vector, -100.0, 100.0)
        
        return state_vector

    def _take_action(self, action, stoploss=0.3):
        threshold = action[0].item()
        outlier_filter = action[1].item()

        current_month = self.prob_files[self.current_step].split('/')[-1].split('.')[0]
        prob_data = self.all_prob_data[current_month]
        
        cluster_data = self.current_data.reset_index()
        
        long_firms, short_firms, _ = calculate_positions(
            cluster_data, prob_data, outlier_filter, threshold
        )
        long_firms.set_index('firms', inplace=True)
        short_firms.set_index('firms', inplace=True)

        next_month_returns = self.returns_data[self.next_month].copy()
        
        log_earning, _, _, _, _, \
        long_firm_returns, short_firm_returns = calculate_portfolio_returns(
            long_firms=long_firms, 
            short_firms=short_firms, 
            next_month_returns=next_month_returns, 
            stoploss=stoploss
        )
        return log_earning, long_firm_returns, short_firm_returns
    

    def _calculate_reward(self, log_earning, long_firm_returns, short_firm_returns):
        """
        보상 함수: 다양한 보상 전략을 지원합니다.
        
        Args:
            log_earning (float): 포트폴리오의 로그 수익률
            long_firm_returns (pd.Series): 롱 포지션 수익률
            short_firm_returns (pd.Series): 숏 포지션 수익률
            
        Returns:
            float: 계산된 보상 값
        """
        # 다이나믹 감마 보너스 계산 (모든 보상 함수에 공통 적용)
        gamma_dynamics = self.dynamic_gamma
        delta_portfolio = self.current_portfolio_value - self.prev_portfolio_value
        dynamic_bonus = delta_portfolio * gamma_dynamics
        
        # 안전장치: NaN 처리를 위한 롱/숏 수익률 전처리
        if long_firm_returns is None or len(long_firm_returns) == 0:
            long_avg = 0.0
        else:
            long_avg = float(long_firm_returns.fillna(0).mean())
            
        if short_firm_returns is None or len(short_firm_returns) == 0:
            short_avg = 0.0
        else:
            short_avg = float(short_firm_returns.fillna(0).mean())
        
        # NaN 체크
        if np.isnan(long_avg): long_avg = 0.0
        if np.isnan(short_avg): short_avg = 0.0
        
        # 스프레드 수익률 계산 (모든 보상 함수에서 사용)
        spread_return = long_avg - short_avg
        
        # 롤링 윈도우 업데이트
        self.rolling_returns.append(log_earning)
        if len(self.rolling_returns) > self.window_size:
            self.rolling_returns.pop(0)
        
        # 스프레드 롤링 윈도우 업데이트
        self.rolling_spread_returns.append(spread_return)
        if len(self.rolling_spread_returns) > self.window_size:
            self.rolling_spread_returns.pop(0)
        
        # ===================================================================
        # 보상 함수 선택 (env_args의 hard_reward 값에 따라)
        # ===================================================================
        
        # 1. 페어 트레이딩 전용 보상 함수 (간단하고 효과적)
        if self.hard_reward == "PairsTrading":
            # 거래 효율성: 포지션 수 대비 수익
            total_positions = (len(long_firm_returns) if long_firm_returns is not None else 0) + \
                            (len(short_firm_returns) if short_firm_returns is not None else 0)
            
            if total_positions > 0:
                # 포지션당 효율성
                efficiency = abs(spread_return) / (total_positions / 50.0)  # 50개 기준 정규화
                efficiency_bonus = min(efficiency, 1.0)  # 상한 1.0
            else:
                efficiency_bonus = -0.1  # 거래 안하면 작은 페널티
            
            # 안정성: 극단적 손실 방지
            if spread_return < -0.08:  # 8% 이상 손실
                stability_penalty = abs(spread_return) * 3.0
            else:
                stability_penalty = 0.0
            
            # 최종 보상: 단순한 가중합
            reward = (
                spread_return * 3.0 +           # 스프레드 수익 (주요)
                efficiency_bonus * 0.5 -        # 효율성 보너스
                stability_penalty               # 안정성 페널티
            )
        
        # 2. 평균-분산 최적화 보상 함수 (True로 설정한 경우)
        elif self.hard_reward == True:
            # 리스크(분산) 계산
            variance_spread = np.var(self.rolling_spread_returns, ddof=1) if len(self.rolling_spread_returns) > 1 else 0
            
            # 평균-분산 유틸리티 적용
            reward = spread_return - self.risk_aversion * variance_spread
            
            # 포트폴리오 가치 업데이트
            self.prev_portfolio_value *= np.exp(spread_return)
        
        # 3. Sharpe 비율 기반 보상
        elif self.hard_reward == "Sharpe":
            all_returns = pd.concat([long_firm_returns, short_firm_returns])
            if all_returns.empty or len(all_returns) == 0:
                reward = 0.0
            else:
                all_returns = all_returns.fillna(0)  # NaN 제거
                avg_return = all_returns.mean()
                avg_volatility = all_returns.std()
                if pd.isna(avg_return) or pd.isna(avg_volatility) or avg_volatility < 1e-8:
                    reward = 0.0
                else:
                    sharpe = avg_return / (avg_volatility + 1e-8)
                    reward = sharpe
        elif self.hard_reward == "RollingSharpe":
            if len(self.rolling_sharpe_window) >1:
                returns_series = pd.Series(self.rolling_returns)
                mean_return = returns_series.mean()
                std_return = returns_series.std()
                if std_return > 0:
                    rolling_sharpe = mean_return / std_return
                else:
                    rolling_sharpe = 0.0
            else:
                rolling_sharpe = 0.0
            reward = rolling_sharpe
            
        # 4. Markowitz 평균-분산 최적화 보상
        elif self.hard_reward == "Markowitz":
            # UQ(r) = U(0) + U′(0)r+ 0.5U′′(0)r^2 = r - (1/2) r^2
            all_returns = pd.concat([long_firm_returns, short_firm_returns])
            if all_returns.empty:
                V = 0
            else:
                all_returns = all_returns.fillna(0)
                V = all_returns.var()
                if pd.isna(V):
                    V = 0
            alpha = 0.5
            reward = log_earning - alpha * V
        
        # 5. CVaR (Conditional Value at Risk) 기반 보상
        elif self.hard_reward == "CVaR":
            all_returns = pd.concat([long_firm_returns, short_firm_returns])
            if all_returns.empty:
                reward = log_earning
            else:
                all_returns = all_returns.fillna(0)
                q_alpha = all_returns.quantile(self.alpha)
                if pd.isna(q_alpha):
                    reward = log_earning
                else:
                    tail_losses = all_returns[all_returns <= q_alpha]
                    cvar = tail_losses.mean() if len(tail_losses) > 0 else 0.0
                    if pd.isna(cvar):
                        cvar = 0.0
                    cvar_penalty = abs(cvar)
                    reward = log_earning - self.lambda_cvar * cvar_penalty
        
        # 6. 기본 보상 함수 (안정적 운용을 위한 위험 회피)
        else:
            # 수익률 보상
            return_reward = log_earning
            
            # 변동성 페널티
            volatility = np.std(self.rolling_returns) if len(self.rolling_returns) > 1 else 0
            volatility_penalty = volatility
            
            # 최대 낙폭 페널티
            drawdown = (self.max_portfolio_value - self.current_portfolio_value) / self.max_portfolio_value
            drawdown_penalty = drawdown
            
            # 최종 보상
            reward = return_reward - volatility_penalty - drawdown_penalty
        
        # 최종 보상에 스케일링 및 다이나믹 보너스 적용
        final_reward = reward * self.reward_scale + dynamic_bonus
        
        # NaN 체크 및 방지
        if np.isnan(final_reward) or np.isinf(final_reward):
            final_reward = 0.0
            
        return float(final_reward)
    
    def step(self, action, logging=False):
        """
        Take a step in the environment.
        Args:
            action (np.array): Action from the agent.
        Returns:
            tuple: (next_state, reward, done, info)
        """
        log_earning, long_returns, short_returns = self._take_action(action)
        reward = self._calculate_reward(log_earning, long_returns, short_returns)

        self.current_step += 1
        done = self.current_step >= len(self.cluster_files) - 1 

        next_state = self._get_state() if not done else np.zeros_like(self._get_state())

        self.total_reward += reward

        # 포트폴리오 가치 업데이트
        self.prev_portfolio_value = self.current_portfolio_value
        self.current_portfolio_value += log_earning # 로그 수익률을 사용
        self.max_portfolio_value = max(self.max_portfolio_value, self.current_portfolio_value)
        
        info = {} # Add an empty info dictionary for Gym compatibility

        return next_state, reward, done, info