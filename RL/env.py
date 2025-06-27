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

class SharpeOptimReward:
    """
    Differential Sharpe Ratio + Tail-risk penalty
    -------------------------------------------------
    * decay     : EWMA 감쇠율(λ). 작을수록 긴 히스토리 반영
    * mdd_alpha : MDD 패널티 계수
    * cvar_alpha: CVaR(95%) 패널티 계수  
    * clip      : 보상 클리핑 한계 (|r| <= clip)
    """
    def __init__(
        self,
        decay: float = 0.02,           # 조금 더 빠른 적응 (기본값보다 높임)
        reward_scale: float = 1.0,
        mdd_alpha: float = 3.0,        # MDD 페널티 강화 (위험 회피)
        cvar_alpha: float = 2.0,       # CVaR 페널티 강화 (꼬리 위험 회피)
        clip: float = 3.0,             # 클리핑 범위 축소 (안정성 증대)
        **kwargs
    ):
        self.decay = decay
        self.reward_scale = reward_scale
        self.mdd_alpha = mdd_alpha
        self.cvar_alpha = cvar_alpha
        self.clip = clip

        # EWMA 통계량
        self.mean_ret = 0.0          # μ_t
        self.var_ret = 1e-8          # σ_t^2  (초기값=ε)
        # 에쿼티 커브 및 MDD 계산용
        self.equity = 1.0
        self.peak_equity = 1.0
        self.returns_history = []     # tail-risk 계산용

    def __call__(self, log_earning: float, long_firm_returns=None, short_firm_returns=None) -> float:
        """
        log_earning : 다음 스텝 로그 수익률 (ln(1+R_t))
        ------------------------------------------------------------------
        보상 = Differential Sharpe Ratio
             – MDD Penalty
             – CVaR_95 Penalty
        """
        # ---------- 1) EWMA 통계량 업데이트 ----------
        r_t = log_earning                    # 로그수익률
        lam = self.decay
        delta = r_t - self.mean_ret
        self.mean_ret += lam * delta
        self.var_ret  += lam * (delta**2 - self.var_ret)
        sigma = np.sqrt(self.var_ret) + 1e-8

        # ---------- 2) Differential Sharpe (Moody & Saffell, 2001) ----------
        # ∂Sharpe/∂r_t ≈ (r_t - μ_{t-1}) / σ_{t-1}
        dsr = delta / sigma

        # ---------- 3) 실시간 MDD 계산 ----------
        self.equity *= np.exp(r_t)
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = 1.0 - self.equity / self.peak_equity      # 비율형 DD (0~1)
        mdd_penalty = self.mdd_alpha * drawdown

        # ---------- 4) Tail-risk(CVaR_95) 페널티 ----------
        self.returns_history.append(r_t)
        # 히스토리가 충분할 때만 계산 (30 스텝 이상으로 조정)
        cvar_penalty = 0.0
        if len(self.returns_history) >= 30:
            # 최대 최근 120개월 (10년) 데이터 사용
            rets = np.array(self.returns_history[-120:])      
            var95 = np.quantile(rets, 0.05)
            cvar95 = rets[rets <= var95].mean()               # 평균 손실 (음수값)
            cvar_penalty = self.cvar_alpha * abs(cvar95)      # 절댓값 → 패널티(+)

        # ---------- 5) 변동성 페널티 (추가적인 안정성) ----------
        volatility_penalty = 0.5 * sigma  # 변동성이 클수록 페널티

        # ---------- 6) 최종 보상 ----------
        reward = (
            self.reward_scale * dsr       # Sharpe Gradient ↑
            - mdd_penalty                 # DD ↓
            - cvar_penalty                # Tail-risk ↓
            - volatility_penalty          # 변동성 ↓
        )

        # ---------- 7) 안정성 클리핑 ----------
        reward = np.clip(reward, -self.clip, self.clip)
        return float(reward)


class DirectSharpeReward:
    """
    직접적인 Sharpe Ratio 기반 보상 함수
    -------------------------------------------------
    전체 기간의 실제 Sharpe Ratio 향상에 집중
    기존 SharpeOptimReward의 복잡한 패널티 없이 순수 Sharpe 최적화
    """
    def __init__(
        self,
        reward_scale: float = 10.0,    # 더 큰 스케일로 명확한 신호
        window_size: int = 24,         # 2년 rolling window
        min_periods: int = 6,          # 최소 6개월 데이터 필요
        **kwargs
    ):
        self.reward_scale = reward_scale
        self.window_size = window_size
        self.min_periods = min_periods
        
        # 수익률 히스토리 저장
        self.returns_history = []
        self.prev_sharpe = 0.0

    def __call__(self, log_earning: float, long_firm_returns=None, short_firm_returns=None) -> float:
        """
        실제 Sharpe Ratio 개선에 기반한 보상
        """
        # 현재 수익률 추가
        self.returns_history.append(log_earning)
        
        # Window size 유지 (최근 24개월만 사용)
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)
        
        # 충분한 데이터가 있을 때만 Sharpe ratio 계산
        if len(self.returns_history) < self.min_periods:
            # 초기에는 단순 수익률 기반 보상
            return self.reward_scale * log_earning
        
        # 현재 Sharpe ratio 계산
        returns_array = np.array(self.returns_history)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1) + 1e-8  # 표본 표준편차
        current_sharpe = mean_return / std_return
        
        # Sharpe ratio 개선도를 보상으로 사용
        sharpe_improvement = current_sharpe - self.prev_sharpe
        
        # 추가 보너스: 절대적 Sharpe 수준도 고려
        absolute_sharpe_bonus = max(0, current_sharpe - 1.0) * 0.1  # Sharpe > 1.0일 때 보너스
        
        # 최종 보상
        reward = (
            self.reward_scale * sharpe_improvement +  # Sharpe 개선도 (핵심)
            absolute_sharpe_bonus +                   # 절대적 Sharpe 보너스
            0.1 * log_earning                         # 기본 수익률 (작은 가중치)
        )
        
        # 이전 Sharpe 업데이트
        self.prev_sharpe = current_sharpe
        
        return float(reward)


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
        
        # 새로운 보상 함수를 위한 변수 초기화
        self.prev_rolling_sharpe = 0.0
        self.prev_drawdown = 0.0
        
        # 'Sharpe' 보상 함수가 사용될 경우, 계산기를 리셋
        if self.hard_reward == 'Sharpe':
            self.sharpe_reward_calculator = SharpeOptimReward(
                decay=0.02,
                reward_scale=self.reward_scale,
                mdd_alpha=3.0,
                cvar_alpha=2.0,
                clip=3.0
            )
        elif self.hard_reward == 'Sharpe2':
            self.sharpe_reward_calculator = DirectSharpeReward(
                reward_scale=10.0,    # 명확한 신호를 위해 큰 스케일
                window_size=24,       # 2년 rolling window
                min_periods=6         # 최소 6개월 데이터
            )
            
        return self._get_state()
    
    def _get_current_data(self):
        """Helper to get current month's data."""
        current_month = self.cluster_files[self.current_step].split('/')[-1].split('.')[0]
        cluster_data = self.all_cluster_data[current_month]
        prob_data = self.all_prob_data.get(current_month) # Use .get for safety
        
        # Determine next month for returns data
        current_date = datetime.strptime(current_month, '%Y-%m')
        next_month_date = current_date.replace(year=current_date.year + (current_date.month == 12), 
                                               month=(current_date.month % 12) + 1)
        next_month_str = next_month_date.strftime('%Y-%m')
        
        next_month_returns = self.returns_data.get(next_month_str)

        return current_month, cluster_data, prob_data, next_month_str, next_month_returns

    def _get_state(self):
        """
        Constructs the state vector for the RL agent.

        The state includes market-wide data, cluster-specific metrics, 
        and historical performance indicators.
        """
        # --- 1. 현재 월 데이터 가져오기 ---
        self.current_month, self.current_data, self.prob_data, self.next_month, self.next_month_returns = self._get_current_data()
        
        if self.next_month_returns is None:
            # 다음 달 수익률 데이터가 없으면 종료
            return np.zeros(self.num_inputs, dtype=np.float32)

        # --- 2. 풍부한 Pairs Trading 특화 State 벡터 구성 ---
        state = []
        
        # === 2-1. 시장 환경 정보 ===
        # S&P 500 수익률 (시장 방향성)
        if self.next_month in self.index_data.index:
            sp500_return = self.index_data.loc[self.next_month, 'Monthly Returns'] 
        else:
            sp500_return = 0.0
        state.append(sp500_return)
        
        # 시장 변동성 (최근 6개월 S&P 500 변동성)
        sp500_volatility = 0.0
        if hasattr(self, 'index_data') and len(self.rolling_returns) > 1:
            # SP500 과거 데이터로 변동성 추정
            past_months = [self.cluster_files[max(0, self.current_step - i)].split('/')[-1].split('.')[0] 
                          for i in range(min(6, self.current_step + 1))]
            sp500_returns = [self.index_data.loc[month, 'Monthly Returns'] 
                           for month in past_months]
            sp500_volatility = np.std(sp500_returns) if len(sp500_returns) > 1 else 0.0
        state.append(sp500_volatility)
        
        # === 2-2. 포트폴리오 성과 지표 ===
        state.append(self.current_portfolio_value)  # 누적 로그 수익률
        state.append(self.max_portfolio_value - self.current_portfolio_value)  # Drawdown
        
        # 최근 성과 추세 (3개월, 6개월 Rolling Sharpe)
        if len(self.rolling_returns) >= 3:
            recent_3m = self.rolling_returns[-3:]
            sharpe_3m = np.mean(recent_3m) / (np.std(recent_3m) + 1e-8)
            state.append(sharpe_3m)
        else:
            state.append(0.0)
            
        if len(self.rolling_returns) >= 6:
            recent_6m = self.rolling_returns[-6:]
            sharpe_6m = np.mean(recent_6m) / (np.std(recent_6m) + 1e-8)
            state.append(sharpe_6m)
        else:
            state.append(0.0)
            
        # === 2-3. 클러스터 & Spread 품질 지표 ===
        if self.current_data is not None and len(self.current_data) > 0:
            # 현재 데이터에 spread가 없다면 계산
            current_data_with_spread = self.current_data.copy()
            if 'spread' not in current_data_with_spread.columns:
                # Spread 계산 (trading_logic와 동일한 방식)
                def compute_spread(group):
                    sorted_desc = group.sort_values(ascending=False).values
                    sorted_asc = group.sort_values(ascending=True).values
                    return sorted_desc - sorted_asc
                
                if 'clusters' in current_data_with_spread.columns and 'MOM1' in current_data_with_spread.columns:
                    current_data_with_spread['spread'] = current_data_with_spread.groupby('clusters')['MOM1'].transform(compute_spread)
            
            # 기본 spread 통계량
            spread_data = current_data_with_spread['spread'] if 'spread' in current_data_with_spread.columns else pd.Series()
            if not spread_data.empty:
                state.append(spread_data.mean())     # 평균 spread
                state.append(spread_data.std())      # spread 변동성
                state.append(spread_data.median())   # 중위값 (outlier 영향 적음)
                state.append(spread_data.quantile(0.25))  # 1사분위수
                state.append(spread_data.quantile(0.75))  # 3사분위수
                state.append((spread_data > 0).mean())    # Positive spread 비율
            else:
                state.extend([0.0] * 6)
                
            # 클러스터 다양성 지표
            clusters = current_data_with_spread['clusters'] if 'clusters' in current_data_with_spread.columns else pd.Series()
            if not clusters.empty:
                num_clusters = clusters.nunique()
                avg_cluster_size = len(clusters) / max(num_clusters, 1)
                largest_cluster_ratio = clusters.value_counts().iloc[0] / len(clusters) if len(clusters) > 0 else 0
                state.extend([num_clusters, avg_cluster_size, largest_cluster_ratio])
            else:
                state.extend([0.0, 0.0, 0.0])
                
            # Momentum 분포 정보
            mom1_data = current_data_with_spread['MOM1'] if 'MOM1' in current_data_with_spread.columns else pd.Series()
            if not mom1_data.empty:
                state.append(mom1_data.mean())       # 평균 momentum
                state.append(mom1_data.std())        # momentum 분산
                state.append(mom1_data.skew())       # 비대칭성 (왜도)
                state.append((mom1_data > 0).mean()) # Positive momentum 비율
            else:
                state.extend([0.0] * 4)
        else:
            state.extend([0.0] * 13)
            
        # === 2-4. 확률 데이터 품질 ===
        if self.prob_data is not None and len(self.prob_data) > 0:
            max_probs = self.prob_data.max(axis=1)
            state.append(max_probs.mean())           # 평균 최대 확률
            state.append(max_probs.std())            # 확률 분산
            state.append((max_probs > 0.5).mean())   # 고신뢰도 예측 비율
            
            # 클러스터 예측 분포
            prob_entropy = -np.sum(self.prob_data * np.log(self.prob_data + 1e-8), axis=1).mean()
            state.append(prob_entropy)               # 평균 엔트로피 (불확실성)
        else:
            state.extend([0.0] * 4)
            
        # === 2-5. 포지션 & 리스크 관리 ===
        # 이전 포지션 정보
        prev_long = getattr(self, 'num_long_positions_prev', 0)
        prev_short = getattr(self, 'num_short_positions_prev', 0)
        total_prev_positions = prev_long + prev_short
        
        state.append(prev_long)
        state.append(prev_short)
        state.append(total_prev_positions)
        state.append(prev_long / max(total_prev_positions, 1))  # Long ratio
        
        # 과거 승률 (최근 10스텝)
        if len(self.rolling_returns) > 0:
            recent_returns = self.rolling_returns[-10:]
            win_rate = (np.array(recent_returns) > 0).mean()
            state.append(win_rate)
            
            # 연속 승/패 횟수
            consecutive_wins = 0
            consecutive_losses = 0
            for ret in reversed(recent_returns):
                if ret > 0:
                    consecutive_wins += 1
                    break
                else:
                    consecutive_losses += 1
            state.append(consecutive_wins)
            state.append(consecutive_losses)
        else:
            state.extend([0.0, 0.0, 0.0])
            
        # === 2-6. 시간 & 계절성 정보 ===
        # 월별 인덱스 (1-12)
        current_month_num = int(self.current_month.split('-')[1])
        state.append(current_month_num / 12.0)  # 정규화
        
        # 분기 정보 (1-4)
        quarter = (current_month_num - 1) // 3 + 1
        state.append(quarter / 4.0)
        
        # 상대적 진행도 (에피소드 내 위치)
        progress = self.current_step / max(len(self.cluster_files) - 1, 1)
        state.append(progress)
        
        # State 벡터 크기 맞추기
        while len(state) < self.num_inputs:
            state.append(0.0)
        
        return np.array(state[:self.num_inputs], dtype=np.float32)


    def _take_action(self, action, stoploss=0.3):
        """
        Implement Pairs Trading strategy based on continuous action.
        """
        threshold = action[0].item()
        outlier_filter = action[1].item()

        # 현재 달 데이터 및 확률 데이터 로드
        current_month = self.prob_files[self.current_step].split('/')[-1].split('.')[0]
        prob_data = self.all_prob_data[current_month]
        
        # cluster_data 변수명을 current_data로 변경하여 사용 (copy해서 안전하게 사용)
        cluster_data = self.current_data.copy()
        
        # 안전하게 인덱스 리셋 (firms 컬럼이 없을 때만 인덱스를 컬럼으로 변환)
        if 'firms' not in cluster_data.columns:
            cluster_data = cluster_data.reset_index(drop=False)
        
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
        
        # Sharpe ratio - 기존 Differential Sharpe Ratio 사용
        elif self.hard_reward == 'Sharpe':
            # SharpeOptimReward 계산기가 초기화되어 있는지 확인
            if not hasattr(self, 'sharpe_reward_calculator'):
                self.sharpe_reward_calculator = SharpeOptimReward(
                    decay=0.02,
                    reward_scale=self.reward_scale,
                    mdd_alpha=3.0,
                    cvar_alpha=2.0,
                    clip=3.0
                )
            
            # 기존 SharpeOptimReward 클래스 사용
            base_reward = self.sharpe_reward_calculator(log_earning, long_firm_returns, short_firm_returns)
        
        # Sharpe2 ratio - 새로운 Direct Sharpe Ratio 사용
        elif self.hard_reward == 'Sharpe2':
            # DirectSharpeReward 계산기가 초기화되어 있는지 확인
            if not hasattr(self, 'sharpe_reward_calculator'):
                self.sharpe_reward_calculator = DirectSharpeReward(
                    reward_scale=10.0,    # 명확한 신호를 위해 큰 스케일
                    window_size=24,       # 2년 rolling window
                    min_periods=6         # 최소 6개월 데이터
                )
            
            # 새로운 DirectSharpeReward 클래스 사용
            base_reward = self.sharpe_reward_calculator(log_earning, long_firm_returns, short_firm_returns)
        
        elif self.hard_reward == 'ReturnVolatilityDrawdown':
            # Hyperparameters to balance return, volatility, and drawdown
            RETURN_SCALE = 1.0
            VOLATILITY_PENALTY_SCALE = 2.0
            DRAWDOWN_PENALTY_SCALE = 1.0

            # 1. Return component
            return_component = RETURN_SCALE * log_earning

            # 2. Volatility penalty component
            # A temporary list is used to include the current return for volatility calculation
            temp_rolling_returns = self.rolling_returns + [log_earning]
            if len(temp_rolling_returns) > 1:
                rolling_volatility = np.std(temp_rolling_returns)
            else:
                rolling_volatility = 0.0
            
            volatility_penalty = VOLATILITY_PENALTY_SCALE * rolling_volatility

            # 3. Drawdown penalty component
            drawdown = self.max_portfolio_value - self.current_portfolio_value
            drawdown_penalty = DRAWDOWN_PENALTY_SCALE * drawdown

            # The base reward is calculated by maximizing returns while minimizing penalties.
            base_reward = return_component - volatility_penalty - drawdown_penalty
        
        elif self.hard_reward == 'ImprovedSharpe':
            # Hyperparameters
            SHARPE_BONUS_SCALE = 0.5 
            DRAWDOWN_BONUS_SCALE = 0.5
            MIN_OBS_FOR_SHARPE = 12 # Minimum months of returns to calculate a stable Sharpe ratio

            # 1. Sharpe Improvement Bonus
            sharpe_bonus = 0.0
            # Add current return to calculate new rolling sharpe
            current_rolling_returns = self.rolling_returns + [log_earning] 
            if len(current_rolling_returns) >= MIN_OBS_FOR_SHARPE:
                returns_series = pd.Series(current_rolling_returns)
                mean_return = returns_series.mean()
                std_dev = returns_series.std()
                current_rolling_sharpe = mean_return / (std_dev + 1e-8)
                
                sharpe_bonus = SHARPE_BONUS_SCALE * (current_rolling_sharpe - getattr(self, 'prev_rolling_sharpe', 0.0))
                
                self.prev_rolling_sharpe = current_rolling_sharpe
            else:
                self.prev_rolling_sharpe = 0.0

            # 2. Drawdown Reduction Bonus
            current_drawdown = self.max_portfolio_value - self.current_portfolio_value
            drawdown_bonus = DRAWDOWN_BONUS_SCALE * (getattr(self, 'prev_drawdown', 0.0) - current_drawdown)
            self.prev_drawdown = current_drawdown

            # The base reward is the sum of the shaping bonuses. 
            # The direct log_earning reward will be added later via the dynamic_bonus.
            base_reward = sharpe_bonus + drawdown_bonus
        
        elif self.hard_reward == 'Sortino':
            # This reward function is based on the Sortino ratio concept,
            # which focuses on penalizing downside volatility rather than total volatility.
            
            # Combine returns from all positions in the current step
            all_returns = pd.concat([long_firm_returns, short_firm_returns])
            
            if all_returns.empty:
                # No positions taken, no risk, no reward.
                base_reward = 0.0
            else:
                # The portfolio's log return for the step.
                portfolio_return = log_earning
                
                # Identify returns that fall below the target (downside returns).
                # We use a target return of 0, meaning we penalize any loss.
                target_return = 0.0
                downside_returns = all_returns[all_returns < target_return]
                
                # Calculate the standard deviation of downside returns.
                # This is the measure of downside risk.
                if not downside_returns.empty:
                    downside_deviation = downside_returns.std()
                else:
                    # If there are no losses, downside risk is zero.
                    downside_deviation = 0.0
                
                # Calculate a step-based Sortino-like ratio.
                # A small epsilon is added to avoid division by zero if there's no downside risk.
                sortino_step_reward = portfolio_return / (downside_deviation + 1e-8)
                
                base_reward = sortino_step_reward * self.reward_scale

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
            current_drawdown = self.max_portfolio_value - self.current_portfolio_value
            if current_drawdown > 0:
                drawdown_penalty = current_drawdown * 0.1
                base_reward -= drawdown_penalty

        # Default: simple scaled log return
        else:
            base_reward = log_earning * self.reward_scale

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