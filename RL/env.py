import numpy as np
import pandas as pd
import os
import sys
from glob import glob
from scipy.stats.mstats import winsorize
import logging
import copy
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.trading_logic import calculate_positions, calculate_portfolio_returns, LONG_NAN_RETURN, SHORT_NAN_RETURN


# ! NEVER USE fill_missing_with_flag
# def fill_missing_with_flag(df, fill_value=-0.5):
#     """
#     이전 시점에 데이터가 존재했으나 다음 달에 데이터가 NaN인 경우, NaN을 fill_value로 채웁니다.
    
#     Parameters:
#     - df (pd.DataFrame): 입력 데이터프레임 (행: 종목, 열: 월)
#     - fill_value (float): NaN을 채울 값
    
#     Returns:
#     - pd.DataFrame: 수정된 데이터프레임
#     """
#     # 컬럼을 연도-월 순으로 정렬 (이미 정렬되어 있다고 가정)
#     sorted_df = df.sort_index(axis=1)
    
#     # 원본 데이터를 복사하여 이전 달 데이터 시프트
#     prev_month = sorted_df.shift(axis=1)
    
#     # 마스크 생성: 이전 달에 데이터가 있고, 현재 달이 NaN인 경우
#     mask = prev_month.notna() & sorted_df.isna()
    
#     # 마스크된 위치의 NaN 값을 fill_value로 대체
#     # 여기서 sorted_df를 수정하지 않고 새로운 DataFrame을 반환
#     filled_df = sorted_df.mask(mask, fill_value)
    
#     return filled_df

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
        self.prob_files = sorted(glob(self.prob_dir + '/*'))
        self.cluster_files = sorted(glob(clusters_dir + '/*'))
        
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
        self.current_portfolio_value = 0
        self.max_portfolio_value = 0.0
        self.prev_portfolio_value = self.current_portfolio_value
        self.consecutive_non_negative_steps = 0
        return self._get_state()
    
    def _get_state(self):
        """
        STATE 정의:
        1. Number of assets (from current cluster file)
        2. S&P500 monthly return
        3. S&P500 monthly volatility
        4. Average momentum log return
        5. Top 25% quantile of momentum log returns
        6. Bottom 25% quantile of momentum log returns
        7. Momentum volatility (standard deviation)
        8. Current portfolio drawdown
        """

        # # Load current and next month's data
        # current_cluster = self.cluster_files[self.current_step] # 현재 달의 클러스터링 결과  예: "data/2001-01.csv"
        # self.current_month = current_cluster.split('/')[-1].split('.')[0]  # e.g., '2001-01'
        # current_data = pd.read_csv(current_cluster, index_col=0) # 현재 달의 클러스터링 결과 불러오기

        # self.current_data = current_data.sort_values(by='MOM1', ascending=False) # Sort by momentum
        # self.next_month = self.cluster_files[self.current_step + 1].split('/')[-1].split('.')[0] # 다음달
        
        # Load current and next month's data
        current_cluster = self.cluster_files[self.current_step]
        self.current_month = current_cluster.split('/')[-1].split('.')[0] 
        self.current_data = self.all_cluster_data[self.current_month]
        self.current_data = self.current_data.sort_values(by='MOM1', ascending=False)
        if self.current_step + 1 < len(self.cluster_files):
            self.next_month = self.cluster_files[self.current_step + 1].split('/')[-1].split('.')[0]

        #* Extract S&P500 data
        sp500 = self.index_data.loc[self.current_month]
        num_assets = sp500['Number of Assets']
        sp_return = sp500['Monthly Returns']
        sp_volatility = sp500['Monthly Volatility']
        
        #* Convert to log returns for each asset & Calculate individual asset information
        log_returns = np.log1p(self.current_data['MOM1'])  # np.log1p(x) is equivalent to log(1 + x)
        avg_asset_return = log_returns.mean()
        top_asset_return = log_returns.quantile(0.75)  # Top 25% quantile of log returns
        bottom_asset_return = log_returns.quantile(0.25)  # Bottom 25% quantile of log returns
        volatility = log_returns.std()

        #* Cluster statistics
        clusters = self.current_data['clusters']
        total_firms = len(clusters)
        cluster_counts = clusters.value_counts()
        n_clusters = len(cluster_counts)
        max_cluster_ratio = cluster_counts.max() / total_firms if total_firms > 0 else 0
        cluster_ratios = cluster_counts / total_firms if total_firms > 0 else pd.Series(dtype=float)
        entropy = - (cluster_ratios * np.log(cluster_ratios + 1e-8)).sum()

        n_clusters_norm = n_clusters / 50.0         # 예: 최대 클러스터 수를 50으로 가정
        entropy_norm = entropy / 50                # 예: 최대 엔트로피를 5 정도로 가정

        #* Portfolio State Information
        # Calculate current positions based on last action (need to store this or recalculate)
        # For simplicity, let's assume we can recalculate or store from _take_action
        # Placeholder: Need to access/recalculate long_firms and short_firms count for the *current* step
        # This might require restructuring how state is obtained relative to action execution.
        # As a temporary proxy, let's use portfolio value and drawdown.

        #* Temporal Features (Example: Rolling Portfolio Volatility)
        # self.rolling_returns 리스트 관리 (매 step 마다 log_earning 추가/오래된 값 제거)
        if len(self.rolling_returns) > self.window_size:
             self.rolling_returns.pop(0)
        if len(self.rolling_returns) > 1:
            self.rolling_volatility = np.std(self.rolling_returns)
        else:
            self.rolling_volatility = 0.0 # 초기에는 변동성 0

        # 이전 스텝의 포지션 정보 사용
        num_long_positions = getattr(self, 'num_long_positions_prev', 0)
        num_short_positions = getattr(self, 'num_short_positions_prev', 0)
        long_ratio = num_long_positions / total_firms if total_firms > 0 else 0
        short_ratio = num_short_positions / total_firms if total_firms > 0 else 0
        self.current_drawdown = self.max_portfolio_value - self.current_portfolio_value

        # Assemble the state vector
        state_vector = np.array([
            # Market Info
            num_assets,
            sp_return,
            sp_volatility,
            # Momentum Info
            avg_asset_return,
            top_asset_return,
            bottom_asset_return,
            volatility,
            # Cluster Info
            n_clusters_norm,
            max_cluster_ratio,
            entropy_norm,
            # Portfolio Info (Placeholders/Simplified)
            #long_ratio, # Requires knowing positions *before* get_state is called for next step
            #self.current_portfolio_value, # Might need normalization
            self.current_drawdown, # Might need normalization
            #self.rolling_volatility
        ], dtype=np.float32)

        # Fill NaNs just in case any calculation resulted in NaN
        state_vector = np.nan_to_num(state_vector, nan=0.0) # Replace NaN with 0

        # Debugging: Check if state_vector contains NaN values (should be handled by nan_to_num)
        # if np.any(np.isnan(state_vector)):
        #     print("Warning: state_vector contains NaN values after nan_to_num:", state_vector)

        return state_vector

    def _take_action(self, action, stoploss=0.3):
        """
        Implement Pairs Trading strategy based on continuous action.
        """
        threshold = action[0].item()
        outlier_filter = action[1].item()

        # 현재 달 데이터 및 확률 데이터 로드
        current_month = self.prob_files[self.current_step].split('/')[-1].split('.')[0]
        prob_data = self.all_prob_data[current_month]
        
        # cluster_data 변수명을 current_data로 변경하여 사용
        cluster_data = self.current_data
        cluster_data.reset_index(inplace=True)
        
        # 공통 트레이딩 로직 사용: 롱/숏 포지션 계산
        long_firms, short_firms, cluster_data_with_spread = calculate_positions(
            cluster_data, prob_data, outlier_filter, threshold
        )
        long_firms.set_index('firms', inplace=True)
        short_firms.set_index('firms', inplace=True)
        # 다음 달 수익률 데이터
        next_month_returns = self.returns_data[self.next_month].copy()
        
        # 공통 트레이딩 로직 사용: 포트폴리오 수익률 계산
        # 포트폴리오 수익률 계산
        log_earning, normal_return, long_indices, short_indices, all_indices, \
        long_firm_returns, short_firm_returns = calculate_portfolio_returns(
            long_firms=long_firms, 
            short_firms=short_firms, 
            next_month_returns=next_month_returns, 
            stoploss=stoploss
        )

        # 현재 스텝의 포지션 수 저장 (step 함수에서 다음 state 계산 시 사용)
        self.num_long_positions = len(long_firms)
        self.num_short_positions = len(short_firms)

        # 누적 포트폴리오 로그 수익률 업데이트
        self.current_portfolio_value += log_earning
        if self.current_portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.current_portfolio_value
        
        # 리워드 계산
        reward = self._calculate_reward(log_earning, long_firm_returns, short_firm_returns)
        
        return log_earning, reward
    

    def _calculate_reward(self, log_earning, long_firm_returns, short_firm_returns):
        """
        Calculate reward based on the selected reward function and incorporate
        dynamic bonus from the change in portfolio value relative to the previous step.
        
        Args:
            log_earning (float): Average log return of the portfolio
            
        Returns:
            float: Calculated reward
        """
        # Binary reward function
        if isinstance(self.hard_reward, bool) and self.hard_reward:
            base_reward = log_earning * self.reward_scale if log_earning >= 0 else self.negative_reward
        
        elif self.hard_reward == 'Extreme':
            # 손실 발생 시 고정 페널티 값 (MULTIPLIER 대신 고정값 사용)
            LOSS_PENALTY_CONSTANT = self.reward_scale # 예: -5점 (원래 log_earning * 100 보다 훨씬 안정적)
            # 이익 발생 시 보상 배율
            GAIN_SCALE_FACTOR = 1.0
            # 연속 무손실 기간에 대한 보너스 배율 (조금 더 작게 시작)
            CONSISTENCY_BONUS_SCALE = 0.02
            # 포트폴리오 롤링 변동성에 대한 페널티 배율 (더 작게 시작)
            VOLATILITY_PENALTY_FACTOR = 0.2
            # 현재 Drawdown에 대한 페널티 배율 (더 작게 시작)
            DRAWDOWN_PENALTY_FACTOR = 0.5
            # 보상 값 클리핑 범위 (선택 사항, 범위를 좁혀볼 수 있음)
            REWARD_CLIP_MIN = -10.0
            REWARD_CLIP_MAX = 5.0

            # 1. 핵심 보상: 손실 시 극단적 페널티, 이익 시 기본 보상
            if log_earning < 0:
                core_reward = -LOSS_PENALTY_CONSTANT # 큰 음수 보상
                self.consecutive_non_negative_steps = 0 # 연속 무손실 기록 리셋
            else:
                core_reward = log_earning * GAIN_SCALE_FACTOR # 양수 보상
                self.consecutive_non_negative_steps += 1 # 연속 무손실 기록 증가

            # 2. 안정성/지속성 보너스
            consistency_bonus = CONSISTENCY_BONUS_SCALE * self.consecutive_non_negative_steps

            # 3. 위험 페널티 (변동성, Drawdown)
            #    - self.rolling_volatility 와 self.current_drawdown 값이
            #      다른 곳에서 계산되어 접근 가능해야 함 (예: step 함수, _get_state)
            #    - 주의: rolling_volatility와 current_drawdown 값의 스케일을 고려하여
            #           페널티 팩터를 조절해야 함. 필요시 이 값들을 정규화 후 사용.
            volatility_penalty = VOLATILITY_PENALTY_FACTOR * getattr(self, 'rolling_volatility', 0.0)
            drawdown_penalty = DRAWDOWN_PENALTY_FACTOR * getattr(self, 'current_drawdown', 0.0)

            # 4. 최종 보상 계산
            total_reward = core_reward + consistency_bonus - volatility_penalty - drawdown_penalty

            # 5. 보상 클리핑 (선택 사항): 보상 값이 너무 극단적으로 변하는 것을 방지
            total_reward = np.clip(total_reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX)

            # 이전 값 업데이트 등
            self.prev_portfolio_value = self.current_portfolio_value

            return total_reward


        # Markowitz utility function - 단순화된 버전
        elif self.hard_reward == 'Markowitz':
            # UQ(r) = U(0) + U′(0)r+ 0.5U′′(0)r^2 = r - (1/2) r^2
            # R = E[U_Q(r)] = E(r) - αV(r)
            r = log_earning
            all_returns = pd.concat([long_firm_returns, short_firm_returns])
            V = all_returns.var() if not all_returns.empty else 0
            alpha = 0.5
            R = r - alpha * V
            base_reward = R * self.reward_scale
        
        # Conditional Value at Risk (CVaR)
        elif self.hard_reward == 'CVaR':
            all_returns = pd.concat([long_firm_returns, short_firm_returns])
            if all_returns.empty:
                base_reward = 0.0
            else:
                q_alpha = all_returns.quantile(self.alpha)
                tail_losses = all_returns[all_returns <= q_alpha]
                cvar = tail_losses.mean() if len(tail_losses) > 0 else 0.0
                cvar_penalty = abs(cvar)
                base_reward = (log_earning - self.lambda_cvar * cvar_penalty) * self.reward_scale
        
        # Sharpe ratio
        elif self.hard_reward == 'Sharpe':
            all_returns = pd.concat([long_firm_returns, short_firm_returns])
            if all_returns.empty:
                base_reward = 0.0
            else:
                avg_return = all_returns.mean()
                volatility = all_returns.std()
                sharpe = avg_return / (volatility + 1e-1)
                base_reward = sharpe * self.reward_scale
        
        elif self.hard_reward == 'Rolling_Sharpe':
            if len(self.rolling_returns) > 1:
                returns_series = pd.Series(self.rolling_returns)
                mean_return = returns_series.mean() # 연율화
                std_dev = returns_series.std() # 연율화
                rolling_sharpe = mean_return / (std_dev + 1e-8) # 분모 0 방지
                base_reward = rolling_sharpe * self.reward_scale # reward_scale 재조정 필요
            else:
                base_reward = 0.0
            
            # drawdown 패널티 추가
            if self.current_drawdown > 0:
                drawdown_penalty = self.current_drawdown * 0.1 # * self.drawdown_scale
                base_reward -= drawdown_penalty

        # Default: simple scaled log return
        else:
            base_reward = 0

        # Dynamic bonus: incorporate change in portfolio value relative to the previous step
        gamma_dynamic = self.dynamic_gamma
        delta_portfolio = self.current_portfolio_value - self.prev_portfolio_value
        dynamic_bonus = gamma_dynamic * delta_portfolio

        # Total reward combines base reward and dynamic bonus
        reward = base_reward + dynamic_bonus
        # Update previous portfolio value for next step
        self.prev_portfolio_value = self.current_portfolio_value

        return reward   

    def step(self, action, logging=False):

        log_earning, reward = self._take_action(action)
        self.total_reward += reward
        self.rolling_returns.append(log_earning)
        
        self.current_step += 1
        if logging:
            logging.info(f"Current Step: {self.current_step}, Month: {self.current_month}, Reward: {reward}, Total Reward: {self.total_reward}, Portfolio Value: {self.current_portfolio_value}")
        done = self.current_step >= len(self.cluster_files) - 1 #? End of episode

        self.done = done

        # 다음 상태 계산 전에 현재 스텝의 포지션 수를 저장
        self.num_long_positions_prev = getattr(self, 'num_long_positions', 0)
        self.num_short_positions_prev = getattr(self, 'num_short_positions', 0)

        next_state = self._get_state() if not done else np.zeros(self.num_inputs, dtype=np.float32)
        return next_state, reward, done