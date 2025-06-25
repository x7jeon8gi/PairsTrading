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
import pickle

# 경로 호환성 확보
current_dir = os.path.dirname(os.path.abspath(__file__))
rl_dir = os.path.join(current_dir, 'RL')
sys.path.append(current_dir)
sys.path.append(rl_dir)

CACHE_DIR = './res/cache'
os.makedirs(CACHE_DIR, exist_ok=True)

long_nan_return = -0.25
short_nan_return = 0.00
korean_tz = pytz.timezone('Asia/Seoul')

# RL 관련 모듈 임포트
from RL.env import TradingEnvironment
from RL.main import Trainer, set_logger

pd.set_option('display.max_columns', None)

# start_month 변수를 제거하고 함수들의 매개변수로 사용
frame = pd.read_pickle('data/log_returns_by_month.pkl')

# 새로운 통합 함수 추가
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
    
    for i, current_month in tqdm(enumerate(sorted_months), desc="Processing months"):
        next_month = sorted_months[i+1] if i < len(sorted_months) - 1 else None
        
        if next_month is None:
            break
            
        # 클러스터 파일 로드
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
        
        # RL 모델인 경우, 해당 timestep의 action에서 threshold와 outlier_filter 설정
        current_threshold = threshold
        current_out_filter = out_filter
        
        if use_rl and i < len(actions):
            action = actions[i]
            # RL 환경의 액션에서 값 추출
            current_threshold = float(action[0])
            current_out_filter = float(action[1])
            print(f"Month: {current_month}, RL Action - Threshold: {current_threshold:.3f}, Outlier Filter: {current_out_filter:.3f}")
        
        # 확률 데이터 로드 (BM 모델이 아닌 경우만)
        prob_data = None
        if not is_bm and prob_dir is not None:
            prob_file = os.path.join(prob_dir, f'{current_month}.csv')
            if os.path.exists(prob_file):
                prob_data = pd.read_csv(prob_file, index_col=0)
            else:
                print(f"[Warning] Probability file not found: {prob_file}")
        
        # 다음 달 수익률 데이터 준비
        next_month_returns = return_data[next_month]
        
        # 포지션 계산
        if is_bm:
            # 벤치마크 모델은 다른 포지션 계산 함수 사용
            long_firms, short_firms, filtered_data = calculate_positions_bm(
                cluster_data=cluster_data,
                threshold=current_threshold
            )
        else:
            # 일반 모델은 기존 포지션 계산 함수 사용
            # cluster_data에 reset_index 적용 (firms 컬럼을 인덱스에서 일반 컬럼으로)
            if 'firms' in cluster_data.index.names:
                cluster_data.reset_index(inplace=True)
                
            long_firms, short_firms, filtered_data = calculate_positions(
                cluster_data=cluster_data,
                prob_data=prob_data,
                outlier_filter=current_out_filter,
                threshold=current_threshold
            )
        
        # firms를 인덱스로 설정
        long_firms.set_index('firms', inplace=True)
        short_firms.set_index('firms', inplace=True)
        filtered_data.set_index('firms', inplace=True)
        
        # 포트폴리오 수익률 계산
        log_earning, normal_return, long_indices, short_indices, all_indices, \
            long_firm_returns, short_firm_returns = calculate_portfolio_returns(
            long_firms=long_firms,
            short_firms=short_firms,
            next_month_returns=next_month_returns,
            stoploss=stoploss
        )
        
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
        strategy_returns.loc[next_month, 'Threshold'] = current_threshold
        strategy_returns.loc[next_month, 'Outlier_Filter'] = current_out_filter
        
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
                            'threshold': current_threshold,
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
                            'threshold': current_threshold,
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

# --- Helper Functions ---
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


def extract_results_from_dataframe(returns_df: pd.DataFrame) -> dict:
    """
    전략 수익률 데이터프레임에서 다양한 지표를 추출합니다.
    """
    results_dict = {
        'log_returns': returns_df['Earning'],
        'turnover': returns_df['Turnover'],
        'cumulative_returns': returns_df['Cumulative_Return'],
        'cumulative_log_returns': returns_df['Cumulative_Log_Return'],
        'normal_returns': returns_df['Normal_Return'],
        'positions': returns_df['Positions'],
        'long_positions': returns_df['Long_Positions'],
        'short_positions': returns_df['Short_Positions'],
        'commission': returns_df['Commission']
    }
    # 수수료가 이미 Earning에 포함되어 있으므로 동일한 값 사용
    results_dict['log_returns_after_commission'] = results_dict['log_returns']
    results_dict['cumulative_returns_after_commission'] = results_dict['cumulative_returns']
    return results_dict


def save_to_cache(cache_path, data):
    """데이터를 캐시 파일에 저장합니다."""
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"결과를 캐시에 저장했습니다: {cache_path}")

def load_from_cache(cache_path):
    """캐시 파일에서 데이터를 불러옵니다."""
    if os.path.exists(cache_path):
        print(f"캐시에서 결과를 불러옵니다: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

def run_bm_models(return_data, threshold, out_filter, stoploss, start_month, commission_rate, current_time, save_output=True):
    """벤치마크 모델들을 실행하고 결과를 반환합니다."""
    print("\n===== 벤치마크 모델 분석 시작 =====")
    bm_list = ['kmeans_20', 'agglo_0.5', 'dbscan_0.1']
    bm_results = {}

    for bm in bm_list:
        # 캐시 키 생성
        cache_key = f"bm_{bm}_t{threshold}_of{out_filter}_sl{stoploss}_s{start_month}_cr{commission_rate}.pkl"
        cache_path = os.path.join(CACHE_DIR, cache_key)
        
        # 캐시 확인
        cached_result = load_from_cache(cache_path)
        if cached_result:
            bm_results[bm] = cached_result
            continue
            
        print(f"\n--- 벤치마크 모델 실행 (캐시 없음): {bm} ---")
        bm_returns = bm_investment_strategy(
            return_data=return_data,
            bm=bm,
            threshold=threshold,
            out_filter=out_filter,
            stoploss=stoploss,
            start_month=start_month,
            commission_rate=commission_rate,
            strategy_name=f"{bm}_{current_time}",
            save_output=save_output
        )
        result = extract_results_from_dataframe(bm_returns)
        bm_results[bm] = result
        save_to_cache(cache_path, result)
    
    # ML 모델 성능 지표 계산
    ml_log_returns = pd.concat([bm_results[bm]['log_returns'] for bm in bm_list], axis=1)
    ml_log_returns.columns = bm_list
    ml_log_returns.fillna(0, inplace=True)
    print("\n=== ML 모델 성능 지표 ===")
    calculate_financial_metrics_monthly(ml_log_returns, verbose=True)
    
    return bm_results

def run_cc_models(return_data, cluster_numbers, threshold, batch, bins, hidden, mask, std, out_filter, start_month, commission_rate, cluster_tau, save_output=True):
    """Contrastive Clustering 모델들을 실행하고 결과를 반환합니다."""
    print("\n===== Contrastive Clustering 모델 분석 시작 =====")
    cc_results = {}
    
    for n_cluster in cluster_numbers:
        # 캐시 키 생성
        cache_key = (f"cc_{n_cluster}_t{threshold}_b{batch}_bins{bins}_h{hidden}_m{mask}_"
                     f"std{std}_of{out_filter}_s{start_month}_cr{commission_rate}_ctau{cluster_tau}.pkl")
        cache_path = os.path.join(CACHE_DIR, cache_key)

        # 캐시 확인
        cached_result = load_from_cache(cache_path)
        if cached_result:
            cc_results[f'cc_{n_cluster}'] = cached_result
            continue

        print(f"\n--- 클러스터 수 (캐시 없음): {n_cluster} ---")
        strategy_returns = optimized_investment_strategy(
            return_data, number_of_cluster=n_cluster, threshold=threshold, batch=batch, bins=bins, hidden=hidden,
            mask=mask, std=std, out_filter=out_filter, start_month=start_month, commission_rate=commission_rate,
            cluster_tau=cluster_tau, save_output=save_output
        )
        result = extract_results_from_dataframe(strategy_returns)
        cc_results[f'cc_{n_cluster}'] = result
        save_to_cache(cache_path, result)
        
    # CC 모델 성능 지표 계산
    cc_log_returns = pd.concat([cc_results[f'cc_{n_cluster}']['log_returns'] for n_cluster in cluster_numbers], axis=1)
    cc_log_returns.columns = [f'cc_{n_cluster}' for n_cluster in cluster_numbers]
    cc_log_returns.fillna(0, inplace=True)
    print("\n=== CC 모델 성능 지표 ===")
    calculate_financial_metrics_monthly(cc_log_returns, verbose=True)
    
    return cc_results


def run_vae_model(return_data, start_month, commission_rate, save_output, n_clusters, threshold, out_filter, stoploss):
    """VAE 모델을 실행하고 결과를 반환합니다."""
    print("\n===== VAE 모델 분석 시작 =====")

    # 캐시 키 생성
    cache_key = f"vae_{n_clusters}_t{threshold}_of{out_filter}_sl{stoploss}_s{start_month}_cr{commission_rate}.pkl"
    cache_path = os.path.join(CACHE_DIR, cache_key)

    # 캐시 확인
    cached_result = load_from_cache(cache_path)
    if cached_result:
        return cached_result

    print("\n--- VAE 모델 실행 (캐시 없음) ---")
    vae_returns = vae_investment_strategy(
        return_data=return_data,
        start_month=start_month,
        commission_rate=commission_rate,
        save_output=save_output,
        n_clusters=n_clusters,
        threshold=threshold,
        out_filter=out_filter,
        stoploss=stoploss,
        save_dir='./res'
    )

    # vae metric
    vae_log_returns = pd.concat([vae_returns['Earning']], axis=1)
    vae_log_returns.columns = ['vae']
    vae_log_returns = vae_log_returns.fillna(0)
    vae_log_returns = np.log1p(vae_log_returns)
    print("\n=== VAE 모델 성능 지표 ===")
    vae_metrics = calculate_financial_metrics_monthly(vae_log_returns, verbose=True)
    print(vae_metrics)
    
    result = extract_results_from_dataframe(vae_returns)
    save_to_cache(cache_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE, RL, 및 벤치마크 모델 비교")
    parser.add_argument('--start_month', type=str, default='2006-01', help='분석 시작 월 (YYYY-MM)')
    parser.add_argument('--commission_rate', type=float, default=0.00, help='거래 수수료율')
    parser.add_argument('--stoploss', type=float, default=-0.3, help='손절매 비율')
    parser.add_argument('--out_filter', type=float, default=0.5, help='이상치 필터링 임계값')
    parser.add_argument('--threshold', type=float, default=1, help='진입 임계값')
    parser.add_argument('--save_output', action='store_true', help='결과 및 포지션 저장 여부')
    
    current_time = datetime.datetime.now(korean_tz).strftime('%Y%m%d_%H%M%S')
    # 기본 시작 월 설정
    default_start_month = '2006-01'

    out_filter = 0.5 # ! Outlier filter
    thres = 1
    batch = 1024
    hidden = 128
    bins = 64
    std = 0.1   
    mask = 0.1
    cluster_tau = 1.0

    # 거래 수수료율 설정 (0.25% = 0.0025)
    commission_rate = 0.00
    stoploss = -0.3

    # 데이터 필터링
    filtered_frame = frame[frame.columns[frame.columns >= default_start_month]]

    # 벤치마크 모델 실행
    results = run_bm_models(
        return_data=filtered_frame,
        threshold=thres,
        out_filter=out_filter,
        stoploss=stoploss,
        start_month=default_start_month,
        commission_rate=commission_rate,
        current_time=current_time,
        save_output=True
    )

    # VAE 모델 실행
    vae_results = run_vae_model(
        return_data=filtered_frame,
        start_month=default_start_month,
        commission_rate=commission_rate,
        save_output=True,
        n_clusters=30,
        threshold=thres,
        out_filter=out_filter,
        stoploss=stoploss,
    )

    # Contrastive Clustering 모델 실행
    cc_results = run_cc_models(
        return_data=filtered_frame,
        cluster_numbers=[10, 20, 30, 40, 50],
        threshold=thres,
        batch=batch,
        bins=bins,
        hidden=hidden,
        mask=mask,
        std=std,
        out_filter=out_filter,
        start_month=default_start_month,
        commission_rate=commission_rate,
        cluster_tau=cluster_tau,
        save_output=True
    )

    # RL Model
    print("\n===== 강화학습 모델 분석 =====")
    # RL 설정 로드
    with open('./RL/SAC_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 필요한 매개변수 설정
    seed = config['seed']
    batch_size = config['batch_size']
    start_steps = config['start_steps']
    n_steps = config['n_steps']
    env_args = config['env_args']
    agent_args = config['agent_args']
    replay_size = config['replay_size']
    update_per_step = config['update_per_step']
    
    # 클러스터 경로 설정 (config에 맞게 조정)
    num_cluster = config['num_cluster']
    env_args['clusters_dir'] = env_args['clusters_dir'] + f'{num_cluster}'
    
    # env_args의 시작 월과 종료 월 설정 (기본값 없으면 설정)
    if 'start_month' not in env_args or not env_args['start_month']:
        env_args['start_month'] = default_start_month
    
    # 모델 이름 구성
    reward_method = env_args['hard_reward']
    reward_scale = env_args['reward_scale']
    dynamic_gamma = env_args['dynamic_gamma']
    hidden = agent_args['hidden_size'][0]
    agent_type = agent_args['agent_type']
    kl_weight = agent_args.get('kl_weight', 0.0)
    group_size = agent_args.get('group_size', 5)
    layers = agent_args.get('layers', 3)
    lr = agent_args['lr']
    if config['save_name'] == 'automatic':
        save_name =  f'{agent_type}{num_cluster}_{reward_method}_scale{reward_scale}_dim{hidden}_g{dynamic_gamma}_layers{layers}_gsize{group_size}_lr{lr}'
    else:
        prefix = config['save_name']
        save_name = f'{prefix}_{agent_type}{num_cluster}_{reward_method}_scale{reward_scale}_dim{hidden}_g{dynamic_gamma}_layers{layers}_gsize{group_size}_lr{lr}'
    
    # 로거 설정
    logger = set_logger(seed=seed, run_name=save_name, save_dir='./res/RL')
    
    # Trainer 객체 올바르게 초기화
    trainer = Trainer(
        batch_size=batch_size,
        start_steps=start_steps,
        n_steps=n_steps,
        save_name=save_name,
        device='cuda',
        seed=seed,
        env_args=env_args,
        agent_args=agent_args,
        replay_size=replay_size,
        update_per_step=update_per_step,
        logger=logger
    )
    
    # 저장된 모델 불러오기
    total_reward, rewards, actions, portfolio_values, states = trainer.inference('./res/RL/models', save_name)
    
    # RL 액션 상세 정보 출력
    print(f"\n강화학습 모델 액션 요약:")
    print(f"- 총 액션 수: {len(actions)}")
    if len(actions) > 0:
        avg_threshold = np.mean([action[0] for action in actions])
        avg_outlier = np.mean([action[1] for action in actions])
        print(f"- 평균 임계값(threshold): {avg_threshold:.4f}")
        print(f"- 평균 아웃라이어 필터: {avg_outlier:.4f}")
    
    # RL 모델의 액션을 활용한 투자 전략 실행
    print("\nRL 모델 액션을 사용한 투자 전략 실행 중...")
    rl_strategy_returns = rl_investment_strategy(
        return_data=filtered_frame,
        actions=actions,
        start_month=default_start_month,
        clusters_dir=env_args['clusters_dir'],
        prob_dir=env_args['clusters_dir'].replace('predictions', 'prob'),
        save_output=True,
        commission_rate=commission_rate
    )
    
    # 결과 추출 및 가공
    rl_results = extract_results_from_dataframe(rl_strategy_returns)
    
    # 다른 모델들의 결과와 함께 성능 비교 (수수료 제외)
    all_log_returns = pd.DataFrame({
        'kmeans_20': results['kmeans_20']['log_returns'],
        'agglo_0.5': results['agglo_0.5']['log_returns'],
        'dbscan_0.1': results['dbscan_0.1']['log_returns'],
        'cc_10': cc_results['cc_10']['log_returns'],
        'cc_20': cc_results['cc_20']['log_returns'],
        'cc_30': cc_results['cc_30']['log_returns'],
        'cc_40': cc_results['cc_40']['log_returns'],
        'cc_50': cc_results['cc_50']['log_returns'],
        'vae': vae_results['log_returns'],
        'rl_model': rl_results['log_returns']
    })
    
    # 수수료 포함 결과 비교
    all_log_returns_after_commission = pd.DataFrame({
        'kmeans_20': results['kmeans_20']['log_returns_after_commission'],
        'agglo_0.5': results['agglo_0.5']['log_returns_after_commission'],
        'dbscan_0.1': results['dbscan_0.1']['log_returns_after_commission'],
        'cc_10': cc_results['cc_10']['log_returns_after_commission'],
        'cc_20': cc_results['cc_20']['log_returns_after_commission'],
        'cc_30': cc_results['cc_30']['log_returns_after_commission'],
        'cc_40': cc_results['cc_40']['log_returns_after_commission'],
        'cc_50': cc_results['cc_50']['log_returns_after_commission'],
        'vae': vae_results['log_returns_after_commission'],
        'rl_model': rl_results['log_returns_after_commission']
    })
    
    # 모든 모델의 성능 지표 계산 (수수료 제외)
    print("\n=== 모든 모델 성능 지표 비교 (수수료 제외) ===")
    all_metrics = calculate_financial_metrics_monthly(all_log_returns, verbose=True)
    print(all_metrics.to_string(index=True))
    
    # 모든 모델의 성능 지표 계산 (수수료 포함)
    print(f"\n=== 모든 모델 성능 지표 비교 (수수료 포함) ===")
    all_metrics_after_commission = calculate_financial_metrics_monthly(all_log_returns_after_commission, verbose=True)
    print(all_metrics_after_commission.to_string(index=True))
    
    # 최종 결과 요약 출력 (수수료 제외)
    best_model = all_metrics['Sharpe Ratio'].idxmax()
    print(f"\n최고 성능 모델 (Sharpe Ratio 기준, 수수료 제외): {best_model}")
    print(f"Sharpe Ratio: {all_metrics.loc[best_model, 'Sharpe Ratio']:.4f}")
    print(f"연간화 수익률: {all_metrics.loc[best_model, 'Annualized Return']*100:.2f}%")
    print(f"최대 낙폭(MDD): {all_metrics.loc[best_model, 'MDD']*100:.2f}%")
    
    # 최종 결과 요약 출력 (수수료 포함)
    best_model_with_commission = all_metrics_after_commission['Sharpe Ratio'].idxmax()
    print(f"\n최고 성능 모델 (Sharpe Ratio 기준, 수수료 포함): {best_model_with_commission}")
    print(f"Sharpe Ratio: {all_metrics_after_commission.loc[best_model_with_commission, 'Sharpe Ratio']:.4f}")
    print(f"연간화 수익률: {all_metrics_after_commission.loc[best_model_with_commission, 'Annualized Return']*100:.2f}%")
    print(f"최대 낙폭(MDD): {all_metrics_after_commission.loc[best_model_with_commission, 'MDD']*100:.2f}%")
    
    # 결과 저장
    output_dir = './res'
    os.makedirs(output_dir, exist_ok=True)
    
    all_log_returns.to_csv(f'{output_dir}/all_log_returns_{current_time}.csv')
    all_log_returns_after_commission.to_csv(f'{output_dir}/all_log_returns_with_commission_{current_time}.csv')
    all_metrics.to_csv(f'{output_dir}/all_metrics_{current_time}.csv')
    all_metrics_after_commission.to_csv(f'{output_dir}/all_metrics_with_commission_{current_time}.csv')
    
    print(f"\n모든 결과가 저장되었습니다: '{output_dir}/' 디렉토리")
    print(f"실행 완료 시간: {datetime.datetime.now(korean_tz).strftime('%Y-%m-%d %H:%M:%S')}")
    