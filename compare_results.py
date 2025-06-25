import pandas as pd
import numpy as np
import os
import sys
from scipy.stats.mstats import winsorize
import pandas as pd
import numpy as np
from glob import glob
from tqdm.auto import tqdm
import copy
import matplotlib.pyplot as plt
#from utils.inference import fill_missing_with_flag
from utils.metrics import calculate_financial_metrics_monthly
import argparse
import yaml
from utils.trading_logic import calculate_positions, calculate_portfolio_returns, calculate_turnover, calculate_positions_bm
import datetime
import warnings
import pytz
from abc import ABC, abstractmethod
from VAE.vae_trading import vae_investment_strategy
from utils.strategy_logic import calculate_monthly_portfolio_log_return

# 경로 호환성 확보
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.join(current_dir, 'RL')
sys.path.append(current_dir)
sys.path.append(rl_dir)

# RL 관련 모듈 임포트
from RL.env import TradingEnvironment
from RL.main import Trainer, set_logger

pd.set_option('display.max_columns', None)
korean_tz = pytz.timezone('Asia/Seoul')

# --------------------------------------------------------------------------------
# --- Configuration Class ---
# --------------------------------------------------------------------------------

class StrategyConfig:
    """전략 실행을 위한 모든 설정을 관리하는 클래스"""
    def __init__(self, hard_reward=None):
        self.start_month = '2006-01'
        self.commission_rate = 0.00
        self.stoploss = -0.3
        self.out_filter = 0.5
        self.threshold = 1
        
        self.cc_params = {
            'batch': 1024, 'hidden': 128, 'bins': 64,
            'std': 0.1, 'mask': 0.1, 'cluster_tau': 1.0
        }
        self.vae_params = {'n_clusters': 30}
        self.bm_list = ['kmeans_20', 'agglo_0.5', 'dbscan_0.1']
        self.cc_n_clusters = [10, 20, 30, 40, 50]
        
        with open('./RL/SAC_config.yaml', 'r') as f:
            self.rl_config = yaml.load(f, Loader=yaml.FullLoader)
            
        # hard_reward 값이 제공된 경우 rl_config 업데이트
        if hard_reward is not None:
            self.rl_config['env_args']['hard_reward'] = hard_reward
            
        self.current_time = datetime.datetime.now(korean_tz).strftime('%Y%m%d_%H%M%S')
        self.output_dir = './res'
        self.cache_dir = './res/cached_results'
        self.save_output = True

# --------------------------------------------------------------------------------
# --- Core Strategy Logic (기존 함수 유지) ---
# --------------------------------------------------------------------------------

def investment_strategy(
    return_data: pd.DataFrame,
    clusters_dir: str,
    prob_dir: str,
    start_month: str = None,
    threshold: float = 0.5,
    actions: list = None,
    out_filter: float = 0.3,
    stoploss: float = -0.3,
    save_output: bool = False,
    commission_rate: float = 0.0025,
    strategy_name: str = None,
    is_bm: bool = False,  # 벤치마크 모델 여부
    positions_output_dir: str = None  # 포지션 저장 디렉토리
):
    """
    일반화된 페어 트레이딩 전략에 대한 수익률 계산
    
    Args:
        return_data: 수익률 데이터
        clusters_dir: 클러스터 디렉토리 경로
        prob_dir: 확률 디렉토리 경로
        start_month: 시작 월 (YYYY-MM 형식)
        threshold: 스프레드 임계값
        actions: 액션 리스트 (RL 모델용)
        out_filter: 아웃라이어 필터링 임계값
        stoploss: 손절 임계값
        save_output: 결과 저장 여부
        commission_rate: 수수료율
        strategy_name: 전략 이름
        is_bm: 벤치마크 모델 여부
        positions_output_dir: 포지션 저장 디렉토리 (None인 경우 './res/positions/{strategy_name}' 사용)
    
    Returns:
        pd.DataFrame: 전략의 월별 성과
    """
    # 정렬 및 인덱스 문자열로 변환
    return_data.index = return_data.index.astype(int)
    sorted_months = sorted(return_data.columns)
    sorted_months = [str(month) for month in sorted_months]
    
    # RL 사용 여부 확인
    use_rl = actions is not None
    if use_rl:
        print(f"강화학습 액션을 사용합니다. 액션 수: {len(actions)}")
    
    # 결과 저장 데이터프레임 초기화 - 추가 컬럼 포함
    strategy_returns = pd.DataFrame(index=sorted_months, columns=[
        'Earning', 'Turnover', 'Cumulative_Log_Return', 'Cumulative_Return', 
        'Normal_Return', 'Commission', 'Positions', 'Long_Positions', 'Short_Positions',
        'Threshold', 'Outlier_Filter'  # 추가 컬럼
    ])
    cumulative_return = 0
    
    # 이전 달 포지션 초기화
    prev_long_positions = {}
    prev_short_positions = {}
    
    # 포지션 저장 관련 설정
    if positions_output_dir is None and strategy_name is not None:
        positions_output_dir = f'./res/positions/{strategy_name}'
    
    if save_output and positions_output_dir:
        os.makedirs(positions_output_dir, exist_ok=True)
        
        # 전체 포지션 정보를 저장할 데이터프레임 초기화
        all_positions = []
    
    for i, current_month in tqdm(enumerate(sorted_months), desc=f"Processing {strategy_name}"):
        next_month = sorted_months[i+1] if i < len(sorted_months) - 1 else None
        
        if next_month is None:
            break

        # For RL models, stop if there are no more actions for the current month
        is_dynamic_action = actions is not None
        if is_dynamic_action and i >= len(actions):
            print(f"[Info] Ran out of actions for RL model at month {current_month}. Stopping this strategy.")
            break
            
        # 클러스터 파일 로드 및 표준화
        cluster_file = os.path.join(clusters_dir, f'{current_month}.csv')
        if not os.path.exists(cluster_file):
            print(f"[Warning] Cluster file not found: {cluster_file}")
            strategy_returns.loc[next_month, 'Earning'] = 0
            strategy_returns.loc[next_month, 'Turnover'] = 0
            strategy_returns.loc[next_month, 'Cumulative_Log_Return'] = cumulative_return
            strategy_returns.loc[next_month, 'Cumulative_Return'] = np.expm1(cumulative_return)
            strategy_returns.loc[next_month, 'Normal_Return'] = 0
            strategy_returns.loc[next_month, 'Commission'] = 0
            strategy_returns.loc[next_month, 'Positions'] = 0
            strategy_returns.loc[next_month, 'Long_Positions'] = 0
            strategy_returns.loc[next_month, 'Short_Positions'] = 0
            strategy_returns.loc[next_month, 'Threshold'] = 0
            strategy_returns.loc[next_month, 'Outlier_Filter'] = 0
            continue
                
        cluster_data = pd.read_csv(cluster_file)
        # 데이터 표준화: 'firms' 컬럼을 인덱스로 설정
        if 'Unnamed: 0' in cluster_data.columns:
            cluster_data.rename(columns={'Unnamed: 0': 'firms'}, inplace=True)
        if 'firms' in cluster_data.columns:
            cluster_data.set_index('firms', inplace=True)

        # RL/동적 액션 처리
        current_threshold = threshold
        current_out_filter = out_filter
        
        if is_dynamic_action:
            action = actions[i]
            current_threshold = float(action[0])
            current_out_filter = float(action[1])
            cluster_data['action_threshold'] = current_threshold
            cluster_data['action_outlier_filter'] = current_out_filter
            
        # 확률 데이터 로드
        prob_data = None
        if not is_bm and prob_dir:
            prob_file = os.path.join(prob_dir, f'{current_month}.csv')
            if os.path.exists(prob_file):
                prob_data = pd.read_csv(prob_file, index_col='firms')

        # 다음 달 수익률
        next_month_returns = return_data[next_month]
        
        # --- 통합된 백테스팅 함수 호출 ---
        log_earning, long_firms, short_firms, _ = calculate_monthly_portfolio_log_return(
            current_cluster_data=cluster_data,
            next_month_returns_series=next_month_returns,
            threshold=current_threshold,
            stoploss=stoploss,
            current_prob_data=prob_data,
            outlier_filter=current_out_filter,
        )
        
        long_indices = long_firms.index
        short_indices = short_firms.index
        all_indices = long_indices.union(short_indices)

        # 포트폴리오 수익률 계산 (중복 제거)
        # log_earning, normal_return, long_indices, short_indices, all_indices, \
        #     long_firm_returns, short_firm_returns = calculate_portfolio_returns(
        #     long_firms=long_firms,
        #     short_firms=short_firms,
        #     next_month_returns=next_month_returns,
        #     stoploss=stoploss
        # )
        
        # 현재 포지션 비중 계산 (균등 비중 가정)
        current_long_positions = {}
        current_short_positions = {}
        
        # 롱 포지션 비중 계산
        num_long = len(long_indices)
        if num_long > 0:
            long_weight = 1.0 / num_long
            for idx in long_indices:
                current_long_positions[idx] = long_weight
        
        # 숏 포지션 비중 계산
        num_short = len(short_indices)
        if num_short > 0:
            short_weight = 1.0 / num_short
            for idx in short_indices:
                current_short_positions[idx] = short_weight
        
        # 턴오버율 계산
        long_turnover, short_turnover, total_turnover = calculate_turnover(
            current_long_positions=current_long_positions,
            current_short_positions=current_short_positions,
            prev_long_positions=prev_long_positions,
            prev_short_positions=prev_short_positions
        )
        
        # 수수료 계산 및 수익률 조정
        commission = total_turnover * commission_rate
        adjusted_log_earning = log_earning - commission
        adjusted_normal_return = np.expm1(adjusted_log_earning)
        
        # 누적 수익률 계산
        cumulative_return += adjusted_log_earning
        
        # 결과 저장 - 추가 정보 포함
        strategy_returns.loc[next_month, 'Earning'] = adjusted_log_earning
        strategy_returns.loc[next_month, 'Turnover'] = total_turnover
        strategy_returns.loc[next_month, 'Cumulative_Log_Return'] = cumulative_return
        strategy_returns.loc[next_month, 'Cumulative_Return'] = np.expm1(cumulative_return)
        strategy_returns.loc[next_month, 'Normal_Return'] = adjusted_normal_return
        strategy_returns.loc[next_month, 'Commission'] = commission
        strategy_returns.loc[next_month, 'Positions'] = len(all_indices)
        strategy_returns.loc[next_month, 'Long_Positions'] = len(long_indices)
        strategy_returns.loc[next_month, 'Short_Positions'] = len(short_indices)
        
        # Correctly log dynamic or static thresholds
        strategy_returns.loc[next_month, 'Threshold'] = current_threshold if current_threshold is not None else 0
        strategy_returns.loc[next_month, 'Outlier_Filter'] = current_out_filter if current_out_filter is not None else 0
        
        print(f"Month: {next_month}, Positions: {len(all_indices)}, "
              f"Return: {adjusted_normal_return:.4f}, "
              f"Turnover: {total_turnover:.4f}, "
              f"Commission: {commission:.4f}, "
              f"Cumulative Return: {np.expm1(cumulative_return):.4f}")
              
        # 포지션 정보 저장
        if save_output and positions_output_dir:
            # 롱 포지션 정보
            long_positions_data = []
            if len(long_indices) > 0:
                # long_firms 데이터프레임에서 필요한 정보 추출
                for firm_id in long_indices:
                    if firm_id in long_firms.index:
                        firm_data = long_firms.loc[firm_id]
                        
                        # 다음 달 수익률 값 가져오기 (없으면 NaN)
                        return_value = next_month_returns.get(firm_id, np.nan)
                        
                        # 포지션 정보 저장
                        long_pos = {
                            'month': current_month,
                            'next_month': next_month,
                            'firm_id': firm_id,
                            'position': 'LONG',
                            'threshold': current_threshold if current_threshold is not None else 0,
                            'next_month_return': return_value,
                            'weight': current_long_positions.get(firm_id, 0)
                        }
                        
                        # MOM1, cluster, spread 등 가능한 필드 추가
                        for col in ['MOM1', 'clusters', 'spread']:
                            if hasattr(firm_data, col) or (isinstance(firm_data, pd.Series) and col in firm_data.index):
                                long_pos[col] = firm_data[col] if isinstance(firm_data, pd.Series) else getattr(firm_data, col)
                            
                        long_positions_data.append(long_pos)
            
            # 숏 포지션 정보
            short_positions_data = []
            if len(short_indices) > 0:
                # short_firms 데이터프레임에서 필요한 정보 추출
                for firm_id in short_indices:
                    if firm_id in short_firms.index:
                        firm_data = short_firms.loc[firm_id]
                        
                        # 다음 달 수익률 값 가져오기 (없으면 NaN)
                        return_value = next_month_returns.get(firm_id, np.nan)
                        
                        # 포지션 정보 저장
                        short_pos = {
                            'month': current_month,
                            'next_month': next_month,
                            'firm_id': firm_id,
                            'position': 'SHORT',
                            'threshold': current_threshold if current_threshold is not None else 0,
                            'next_month_return': return_value,
                            'weight': current_short_positions.get(firm_id, 0)
                        }
                        
                        # MOM1, cluster, spread 등 가능한 필드 추가
                        for col in ['MOM1', 'clusters', 'spread']:
                            if hasattr(firm_data, col) or (isinstance(firm_data, pd.Series) and col in firm_data.index):
                                short_pos[col] = firm_data[col] if isinstance(firm_data, pd.Series) else getattr(firm_data, col)
                            
                        short_positions_data.append(short_pos)
            
            # 월별 포지션 정보 저장
            if long_positions_data or short_positions_data:
                # 롱 및 숏 포지션 데이터 합치기
                month_positions = pd.DataFrame(long_positions_data + short_positions_data)
                
                if not month_positions.empty:
                    # 전체 저장 리스트에 추가
                    all_positions.append(month_positions)
                    
                    # 월별 파일로 저장
                    month_file = os.path.join(positions_output_dir, f'positions_{current_month}.csv')
                    month_positions.to_csv(month_file, index=False)
        
        # 이전 달 포지션 업데이트
        prev_long_positions = current_long_positions
        prev_short_positions = current_short_positions
    
    # 모든 월의 포지션 정보를 하나의 파일로 저장
    if save_output and positions_output_dir and all_positions:
        combined_positions = pd.concat(all_positions, ignore_index=True)
        all_positions_file = os.path.join(positions_output_dir, 'all_positions.csv')
        combined_positions.to_csv(all_positions_file, index=False)
        print(f"모든 포지션 정보가 저장되었습니다: {all_positions_file}")
    
    # 결과 저장
    if save_output:
        current_time = datetime.datetime.now(korean_tz).strftime('%Y%m%d_%H%M%S')
        output_dir = './output/strategies/'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{strategy_name}_{current_time}.csv')
        strategy_returns.to_csv(output_file)
        print(f"Results saved to {output_file}")
    
    return strategy_returns

# 기존 함수를 새 함수를 사용하도록 수정
def optimized_investment_strategy(return_data: pd.DataFrame, number_of_cluster: int, threshold: float,
                                  batch: int = 1024, bins: int = 64, hidden: int = 128,
                                  std: float = 0.05, mask: float = 0.05, bm: str = False,
                                  out_filter: float = 0.5, stoploss: float = -0.3,
                                  save_output: bool = False, start_month: str = '1995-01',
                                  commission_rate: float = 0.0025, cluster_tau: float = 1.0):
    """
    페어트레이딩을 계산하는 함수입니다.
    
    Returns:
        pd.DataFrame: 전략의 월별 성과 (Earning, Turnover 컬럼 포함)
    """
    # 클러스터 디렉토리 경로 설정
    if bm:
        # ML 벤치마크 모델 사용 (kmeans, agglo, dbscan 등)
        clusters_dir = f'./res/clusters/{bm}/'
        prob_dir = None  # 벤치마크 모델은 확률 디렉토리가 없을 수 있음
        strategy_name = f"{bm}_t{threshold}"
        print(f"벤치마크 모델 사용 중: {bm}, 클러스터 디렉토리: {clusters_dir}")
    else:
        # Contrastive Clustering 모델 사용
        dir_name = f'batch_{batch}_n_bins_{bins}_hidden_{hidden}_std_{std}_mask_{mask}_ctau_{cluster_tau}'
        clusters_dir = f'./res/{dir_name}/predictions/{number_of_cluster}'
        prob_dir = clusters_dir.replace('predictions', 'prob')
        strategy_name = f"Optimized_c{number_of_cluster}_t{threshold}"
        print(f"Contrastive Clustering 모델 사용 중, 클러스터 디렉토리: {clusters_dir}")
    
    # 포지션 저장 디렉토리 설정
    positions_output_dir = f'./res/positions/{strategy_name}' if save_output else None
    
    # 새로운 통합 함수 호출
    strategy_returns = investment_strategy(
        return_data=return_data,
        clusters_dir=clusters_dir,
        prob_dir=prob_dir,
        start_month=start_month,
        threshold=threshold,
        actions=None,
        out_filter=out_filter,
        stoploss=stoploss,
        save_output=save_output,
        commission_rate=commission_rate,
        strategy_name=strategy_name,
        is_bm=bm,
        positions_output_dir=positions_output_dir
    )
    
    return strategy_returns


def rl_investment_strategy(return_data: pd.DataFrame, actions: list, start_month: str,
                           clusters_dir: str, prob_dir: str = None, bm: str = False,
                           save_output: bool = True, stoploss: float = -0.3,
                           commission_rate: float = 0.001):
    """
    강화학습 기반 페어트레이딩을 계산하는 함수입니다.
    
    Returns:
        pd.DataFrame: 전략의 월별 성과 (Earning, Turnover 컬럼 포함)
    """
    # 클러스터 디렉토리 경로 설정
    if bm:
        # ML 벤치마크 모델 사용 (kmeans, agglo, dbscan 등)
        actual_clusters_dir = f'./res/clusters/{bm}/'
        actual_prob_dir = None  # 벤치마크 모델은 확률 디렉토리가 없을 수 있음
    else:
        # 제공된 clusters_dir 사용
        actual_clusters_dir = clusters_dir
        actual_prob_dir = prob_dir
    
    # 확인 메시지 출력
    print(f"RL 클러스터 디렉토리: {actual_clusters_dir}")
    
    # 포지션 저장 디렉토리 설정
    strategy_name = "RL"
    positions_output_dir = f'./res/positions/{strategy_name}' if save_output else None
    
    # 새로운 통합 함수 호출
    strategy_returns = investment_strategy(
        return_data=return_data,
        clusters_dir=actual_clusters_dir,
        prob_dir=actual_prob_dir,
        start_month=start_month,
        threshold=None,
        actions=actions,
        out_filter=None,  # actions에서 결정되므로 사용하지 않음
        stoploss=stoploss,
        save_output=save_output,
        commission_rate=commission_rate,
        strategy_name=strategy_name,
        is_bm=bm,
        positions_output_dir=positions_output_dir
    )
    
    return strategy_returns


def bm_investment_strategy(
    return_data: pd.DataFrame,
    bm: str,
    threshold: float = 1,
    out_filter: float = 0, # 벤치마크 모델에서는 아웃라이어 필터링 사용 X
    stoploss: float = -0.3,
    save_output: bool = False,
    start_month: str = None,
    commission_rate: float = 0.0025,
    strategy_name: str = 'bm'
):
    """
    벤치마크 모델에 대한 투자 전략 계산 함수
    
    Args:
        return_data: 수익률 데이터
        bm: 벤치마크 모델 이름
        threshold: 스프레드 임계값
        out_filter: 아웃라이어 필터링 임계값
        stoploss: 손절 임계값
        save_output: 결과 저장 여부
        start_month: 시작 월 (YYYY-MM 형식)
        commission_rate: 수수료율
        strategy_name: 전략 이름
    
    Returns:
        pd.DataFrame: 전략의 월별 성과
    """
    # 클러스터 디렉토리 설정
    clusters_dir = f'./res/clusters/{bm}/'
    print(f"Loading clusters from: {clusters_dir}")
    
    # 포지션 저장 디렉토리 설정
    positions_output_dir = f'./res/positions/{strategy_name}' if save_output else None
    
    # 투자 전략 실행
    strategy_returns = investment_strategy(
        return_data=return_data,
        clusters_dir=clusters_dir,
        prob_dir=None,  # BM 모델은 확률 데이터 없음
        start_month=start_month,
        threshold=threshold,
        out_filter=out_filter,
        stoploss=stoploss,
        save_output=save_output,
        commission_rate=commission_rate,
        strategy_name=strategy_name,
        is_bm=True,  # 벤치마크 모델임을 표시
        positions_output_dir=positions_output_dir
    )
    
    return strategy_returns

# --------------------------------------------------------------------------------
# --- Helper Functions ---
# --------------------------------------------------------------------------------

def get_rl_model_name(rl_config):
    """RL 모델 저장 이름 생성"""
    env_args = rl_config['env_args']
    agent_args = rl_config['agent_args']
    num_cluster = rl_config['num_cluster']
    save_name_config = rl_config['save_name']
    
    base_name = (
        f"{agent_args['agent_type']}{num_cluster}_"
        f"{env_args['hard_reward']}_scale{env_args['reward_scale']}_"
        f"dim{agent_args['hidden_size'][0]}_g{env_args['dynamic_gamma']}_"
        f"layers{agent_args.get('layers', 3)}_"
        f"gsize{agent_args.get('group_size', 8)}_lr{agent_args['lr']}"
    )
    
    if save_name_config == 'automatic':
        return base_name
    return f"{save_name_config}_{base_name}"

def initialize_rl_trainer(rl_config, save_name):
    """RL Trainer 객체 초기화"""
    logger = set_logger(seed=rl_config['seed'], run_name=save_name, save_dir='./res/RL')
    return Trainer(
        batch_size=rl_config['batch_size'],
        start_steps=rl_config['start_steps'],
        n_steps=rl_config['n_steps'],
        save_name=save_name,
        device='cuda',
        seed=rl_config['seed'],
        env_args=rl_config['env_args'],
        agent_args=rl_config['agent_args'],
        replay_size=rl_config['replay_size'],
        update_per_step=rl_config['update_per_step'],
        logger=logger
    )

def extract_results(strategy_df: pd.DataFrame, model_name: str):
    """전략 실행 결과에서 주요 지표 추출"""
    print(f"Extracting results for {model_name}...")
    
    # VAE 모델의 경우 컬럼 이름이 다를 수 있음
    if 'Earning' not in strategy_df.columns and 'Log_Return' in strategy_df.columns:
        strategy_df = strategy_df.rename(columns={'Log_Return': 'Earning'})

    # 필수 컬럼이 없는 경우 대비
    required_cols = ['Earning', 'Turnover', 'Cumulative_Return', 'Positions', 'Commission']
    for col in required_cols:
        if col not in strategy_df.columns:
            strategy_df[col] = np.nan

    results = {
        'log_returns': strategy_df['Earning'],
        'turnover': strategy_df['Turnover'],
        'cumulative_returns': strategy_df['Cumulative_Return'],
        'positions': strategy_df['Positions'],
        'commission': strategy_df['Commission']
    }
    results['log_returns_after_commission'] = results['log_returns']
    results['cumulative_returns_after_commission'] = results['cumulative_returns']
    return results

# --------------------------------------------------------------------------------
# --- Strategy Runner Functions (Wrappers) ---
# --------------------------------------------------------------------------------

def run_bm_strategy(config: StrategyConfig, return_data: pd.DataFrame, bm_name: str):
    """벤치마크 모델 전략 실행"""
    print(f"\n===== 벤치마크 모델 분석: {bm_name} =====")
    
    # 캐시 파일 경로 설정 및 확인
    os.makedirs(config.cache_dir, exist_ok=True)
    cache_file = os.path.join(config.cache_dir, f"bm_{bm_name}.pkl")
    
    if os.path.exists(cache_file):
        print(f"'{cache_file}'에서 캐시된 결과를 로드합니다.")
        bm_returns = pd.read_pickle(cache_file)
    else:
        print("캐시된 결과가 없어 새로 실행합니다...")
        strategy_name = f"{bm_name}_{config.current_time}"
        bm_returns = bm_investment_strategy(
            return_data=return_data,
            bm=bm_name,
            threshold=config.threshold,
            stoploss=config.stoploss,
            start_month=config.start_month,
            commission_rate=config.commission_rate,
            strategy_name=strategy_name,
            save_output=config.save_output
        )
        # 실행 후 결과를 캐시 파일로 저장
        bm_returns.to_pickle(cache_file)
        print(f"결과를 '{cache_file}'에 캐시로 저장했습니다.")
        
    return extract_results(bm_returns, bm_name)

def run_cc_strategy(config: StrategyConfig, return_data: pd.DataFrame, n_cluster: int):
    """Contrastive Clustering 모델 전략 실행"""
    model_name = f'cc_{n_cluster}'
    print(f"\n===== Contrastive Clustering 모델 분석 (클러스터 수: {n_cluster}) =====")

    # 캐시 파일 경로 설정 및 확인
    os.makedirs(config.cache_dir, exist_ok=True)
    cache_file = os.path.join(config.cache_dir, f"cc_{n_cluster}.pkl")

    if os.path.exists(cache_file):
        print(f"'{cache_file}'에서 캐시된 결과를 로드합니다.")
        strategy_returns = pd.read_pickle(cache_file)
    else:
        print("캐시된 결과가 없어 새로 실행합니다...")
        strategy_returns = optimized_investment_strategy(
            return_data,
            number_of_cluster=n_cluster,
            threshold=config.threshold,
            out_filter=config.out_filter,
            start_month=config.start_month,
            commission_rate=config.commission_rate,
            save_output=config.save_output,
            stoploss=config.stoploss,
            **config.cc_params
        )
        # 실행 후 결과를 캐시 파일로 저장
        strategy_returns.to_pickle(cache_file)
        print(f"결과를 '{cache_file}'에 캐시로 저장했습니다.")

    return extract_results(strategy_returns, model_name)

def run_vae_strategy(config: StrategyConfig, return_data: pd.DataFrame):
    """VAE 모델 전략 실행"""
    model_name = 'vae'
    print(f"\n===== VAE 모델 분석 =====")
    
    # 캐시 파일 경로 설정 및 확인
    os.makedirs(config.cache_dir, exist_ok=True)
    cache_file = os.path.join(config.cache_dir, "vae.pkl")

    if os.path.exists(cache_file):
        print(f"'{cache_file}'에서 캐시된 결과를 로드합니다.")
        vae_results_df = pd.read_pickle(cache_file)
    else:
        print("캐시된 결과가 없어 새로 실행합니다...")
        vae_results_df = vae_investment_strategy(
            return_data=return_data,
            start_month=config.start_month,
            commission_rate=config.commission_rate,
            save_output=config.save_output,
            n_clusters=config.vae_params['n_clusters'],
            threshold=config.threshold,
            out_filter=config.out_filter,
            stoploss=config.stoploss,
            save_dir=config.output_dir
        )
        # 실행 후 결과를 캐시 파일로 저장
        vae_results_df.to_pickle(cache_file)
        print(f"결과를 '{cache_file}'에 캐시로 저장했습니다.")
        
    return extract_results(vae_results_df, model_name)

def run_rl_strategy(config: StrategyConfig, return_data: pd.DataFrame):
    """RL 모델 전략 실행"""
    model_name = 'rl_model'
    print(f"\n===== {model_name} 분석 =====")
    
    rl_config = config.rl_config
    env_args = rl_config['env_args']
    num_cluster = rl_config['num_cluster']
    env_args['clusters_dir'] = f"res/batch_{config.cc_params['batch']}_n_bins_{config.cc_params['bins']}_hidden_{config.cc_params['hidden']}_std_{config.cc_params['std']}_mask_{config.cc_params['mask']}_ctau_{config.cc_params['cluster_tau']}/predictions/{num_cluster}"
    
    if 'start_month' not in env_args or not env_args['start_month']:
        env_args['start_month'] = config.start_month

    save_name = get_rl_model_name(rl_config)
    trainer = initialize_rl_trainer(rl_config, save_name)
    _, _, actions, _, _ = trainer.inference('./res/RL/models', save_name)
    
    print(f"\nRL 모델 액션 요약 (총 {len(actions)}개):")
    if actions:
        avg_threshold = np.mean([a[0] for a in actions])
        avg_outlier = np.mean([a[1] for a in actions])
        print(f"- 평균 임계값: {avg_threshold:.4f}, 평균 아웃라이어 필터: {avg_outlier:.4f}")

    rl_strategy_returns = rl_investment_strategy(
        return_data=return_data,
        actions=actions,
        start_month=config.start_month,
        clusters_dir=env_args['clusters_dir'],
        prob_dir=env_args['clusters_dir'].replace('predictions', 'prob'),
        save_output=config.save_output,
        commission_rate=config.commission_rate,
        stoploss=config.stoploss
    )
    return extract_results(rl_strategy_returns, model_name)

# --------------------------------------------------------------------------------
# --- Result Analysis and Saving ---
# --------------------------------------------------------------------------------

def analyze_and_save_results(all_results: dict, config: StrategyConfig):
    """모든 전략 결과를 분석하고 저장"""
    log_returns_df = pd.DataFrame({name: res['log_returns'] for name, res in all_results.items()}).fillna(0)
    log_returns_after_commission_df = pd.DataFrame({name: res['log_returns_after_commission'] for name, res in all_results.items()}).fillna(0)

    print("\n\n" + "="*20 + " 최종 결과 " + "="*20)
    
    # 수수료 제외 성능
    print("\n=== 모든 모델 성능 지표 비교 (수수료 제외) ===")
    metrics = calculate_financial_metrics_monthly(log_returns_df, verbose=True)
    print(metrics.to_string(index=True))

    # 수수료 포함 성능
    print(f"\n=== 모든 모델 성능 지표 비교 (수수료 {config.commission_rate*100:.2f}% 포함) ===")
    metrics_after_commission = calculate_financial_metrics_monthly(log_returns_after_commission_df, verbose=True)
    print(metrics_after_commission.to_string(index=True))

    # 최고 성능 모델 요약
    print("\n--- 최고 성능 모델 요약 ---")
    if not metrics.empty:
        best_model = metrics['Sharpe Ratio'].idxmax()
        print(f"\n최고 성능 모델 (Sharpe Ratio 기준, 수수료 제외): {best_model}")
        print(metrics.loc[best_model])

    if not metrics_after_commission.empty:
        best_model_with_commission = metrics_after_commission['Sharpe Ratio'].idxmax()
        print(f"\n최고 성능 모델 (Sharpe Ratio 기준, 수수료 포함): {best_model_with_commission}")
        print(metrics_after_commission.loc[best_model_with_commission])
    
    # 결과 파일 저장
    os.makedirs(config.output_dir, exist_ok=True)
    log_returns_df.to_csv(f'{config.output_dir}/all_log_returns_{config.current_time}.csv')
    log_returns_after_commission_df.to_csv(f'{config.output_dir}/all_log_returns_with_commission_{config.current_time}.csv')
    metrics.to_csv(f'{config.output_dir}/all_metrics_{config.current_time}.csv')
    metrics_after_commission.to_csv(f'{config.output_dir}/all_metrics_with_commission_{config.current_time}.csv')
    
    print(f"\n모든 결과가 저장되었습니다: '{config.output_dir}/' 디렉토리")

# --------------------------------------------------------------------------------
# --- Main Execution ---
# --------------------------------------------------------------------------------

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="모델 비교 및 결과 저장")
    parser.add_argument('--hard_reward', type=str, default='Sharpe', 
                        help="RL 모델의 hard_reward 방식 지정 (예: Sharpe, Markowitz, CVaR, ForPairsTrading 등)")
    args = parser.parse_args()
    
    # hard_reward 인자를 StrategyConfig에 전달
    config = StrategyConfig(hard_reward=args.hard_reward)
    
    # 데이터 로드 및 필터링
    frame = pd.read_pickle('data/log_returns_by_month.pkl')
    filtered_frame = frame[frame.columns[frame.columns >= config.start_month]]

    all_results = {}

    ## 1. 벤치마크 모델 실행
    for bm_name in config.bm_list:
        all_results[bm_name] = run_bm_strategy(config, filtered_frame, bm_name)

    ## 2. VAE 모델 실행
    all_results['vae'] = run_vae_strategy(config, filtered_frame)

    ## 3. Contrastive Clustering 모델 실행
    for n_cluster in config.cc_n_clusters:
        model_name = f'cc_{n_cluster}'
        all_results[model_name] = run_cc_strategy(config, filtered_frame, n_cluster)

    ## 4. RL 모델 실행
    all_results['rl_model'] = run_rl_strategy(config, filtered_frame)

    ## 최종 결과 분석 및 저장
    analyze_and_save_results(all_results, config)
    
    print(f"\n실행 완료 시간: {datetime.datetime.now(korean_tz).strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()