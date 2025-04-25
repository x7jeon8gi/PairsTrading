import numpy as np
import pandas as pd

# 전역 변수로 NaN 처리 값 설정
LONG_NAN_RETURN = -0.25
SHORT_NAN_RETURN = 0.00

def calculate_positions(
    cluster_data: pd.DataFrame,
    prob_data: pd.DataFrame,
    outlier_filter: float,
    threshold: float
):
    """
    클러스터 데이터와 확률 데이터를 사용하여 롱/숏 포지션을 결정합니다.
    
    Args:
        cluster_data: 클러스터링 결과 데이터프레임
        prob_data: 확률 데이터프레임
        outlier_filter: 아웃라이어 필터 임계값 (0~1)
        threshold: 스프레드 임계값 계수
        
    Returns:
        tuple: (long_firms, short_firms, cluster_data_with_spread)
    """
    # 확률 필터링 적용
    if 'firms' in prob_data.columns:
        prob_data.set_index('firms', inplace=True)
    prob_data['max_prob'] = prob_data.max(axis=1)
    median_max_prob = prob_data['max_prob'].quantile(outlier_filter)
    
    # 클러스터와 확률 데이터 병합
    combined_data = cluster_data.merge(prob_data, left_on='firms', right_index=True)
    combined_data['clusters'] = np.where(combined_data['max_prob'] > median_max_prob,
                                    combined_data['clusters'], 0)
    
    # 클러스터 0 제외하고 필터링
    filtered_data = combined_data[['clusters', 'MOM1', 'firms']]
    filtered_data = filtered_data[filtered_data['clusters'] != 0]
    
    # 모멘텀과 클러스터 기준으로 정렬
    filtered_data.sort_values(by=['MOM1', 'firms'], ascending=[False, True], inplace=True)
    
    # 스프레드 계산 함수
    def compute_spread(group):
        sorted_desc = group.sort_values(ascending=False).values
        sorted_asc = group.sort_values(ascending=True).values
        return sorted_desc - sorted_asc
    
    # 각 클러스터 내 스프레드 계산
    filtered_data['spread'] = filtered_data.groupby('clusters')['MOM1'].transform(compute_spread)
    spread_std = filtered_data['spread'].std()
    
    # 포지션 결정
    filtered_data['Long_or_Short'] = (-filtered_data['spread'] / filtered_data['spread'].abs())
    filtered_data['inPortfolio'] = filtered_data['spread'].abs() > spread_std * threshold
    
    # 롱 포지션과 숏 포지션 구분
    long_firms = filtered_data[(filtered_data['Long_or_Short'] == 1) & filtered_data['inPortfolio']]
    short_firms = filtered_data[(filtered_data['Long_or_Short'] == -1) & filtered_data['inPortfolio']]
    
    return long_firms, short_firms, filtered_data

def calculate_positions_bm(
    cluster_data: pd.DataFrame,
    threshold: float
):
    """
    벤치마크 모델을 위한 포지션 계산 함수입니다.
    확률 데이터를 사용하지 않고, 클러스터 -1을 제외합니다.
    
    Args:
        cluster_data: 클러스터링 결과 데이터프레임
        threshold: 스프레드 임계값 계수
        
    Returns:
        tuple: (long_firms, short_firms, filtered_data)
    """
    # 클러스터 -1 제외하고 필터링
    filtered_data = cluster_data[['clusters', 'MOM1', 'firms']]
    filtered_data = filtered_data[filtered_data['clusters'] != -1]
    
    # 모멘텀과 클러스터 기준으로 정렬
    filtered_data.sort_values(by=['MOM1', 'firms'], ascending=[False, True], inplace=True)
    
    # 스프레드 계산 함수
    def compute_spread(group):
        sorted_desc = group.sort_values(ascending=False).values
        sorted_asc = group.sort_values(ascending=True).values
        return sorted_desc - sorted_asc
    
    # 각 클러스터 내 스프레드 계산
    filtered_data['spread'] = filtered_data.groupby('clusters')['MOM1'].transform(compute_spread)
    spread_std = filtered_data['spread'].std()
    
    # 포지션 결정
    filtered_data['Long_or_Short'] = (-filtered_data['spread'] / filtered_data['spread'].abs())
    filtered_data['inPortfolio'] = filtered_data['spread'].abs() > spread_std * threshold
    
    # 롱 포지션과 숏 포지션 구분
    long_firms = filtered_data[(filtered_data['Long_or_Short'] == 1) & filtered_data['inPortfolio']]
    short_firms = filtered_data[(filtered_data['Long_or_Short'] == -1) & filtered_data['inPortfolio']]
    

    return long_firms, short_firms, filtered_data

def calculate_portfolio_returns(
    long_firms: pd.DataFrame,
    short_firms: pd.DataFrame,
    next_month_returns: pd.Series,
    stoploss: float = -0.3
):
    """
    롱/숏 포지션에 기반하여 포트폴리오 수익률을 계산합니다.
    
    Args:
        long_firms: 롱 포지션 데이터프레임
        short_firms: 숏 포지션 데이터프레임
        next_month_returns: 다음 월의 수익률 시리즈
        stoploss: 손절 하한값
        
    Returns:
        tuple: (log_earning, normal_return, long_indices, short_indices, all_indices)
    """

    #!CAUTION 롱/숏 포지션과 return_data 인덱스 일치 여부 확인하면 안됩니다. (미래참조 문제 있음)
    #long_indices = [idx for idx in long_firms.index if idx in next_month_returns.index]
    #short_indices = [idx for idx in short_firms.index if idx in next_month_returns.index]
    long_indices = long_firms.index
    short_indices = short_firms.index

    # 숏 포지션 수익률 반전
    next_month_returns = next_month_returns.copy()
    next_month_returns.loc[short_indices] *= -1
    
    # 손절 적용
    stoploss_value = -np.abs(stoploss)
    condition_apply_stoploss = (next_month_returns.notna()) & (next_month_returns < stoploss_value)
    next_month_returns = next_month_returns.where(~condition_apply_stoploss, stoploss_value)
    
    # NaN 처리
    next_month_returns.loc[long_indices] = next_month_returns.loc[long_indices].fillna(LONG_NAN_RETURN)
    next_month_returns.loc[short_indices] = next_month_returns.loc[short_indices].fillna(SHORT_NAN_RETURN)
    #next_month_returns.fillna(0, inplace=True) 딱히 필요 없음
    
    # 포지션별 평균 로그 수익률 계산
    all_indices = np.concatenate([long_indices, short_indices]) # concatenate 사용
    num_positions = len(all_indices)
    
    if num_positions == 0:
        log_earning = 0.0
        normal_return = 0.0
    else:
        # 포지션의 총 수익률 합산
        positions_sum = next_month_returns.loc[all_indices].sum()

        # 로그 수익률 계산 (균등 비중 가정)
        log_earning = positions_sum / num_positions
        
        # 로그 수익률을 일반 수익률로 변환
        normal_return = np.expm1(log_earning)
    
    long_firm_returns = next_month_returns.loc[long_indices].fillna(LONG_NAN_RETURN)
    short_firm_returns = next_month_returns.loc[short_indices].fillna(SHORT_NAN_RETURN)

    return log_earning, normal_return, long_indices, short_indices, all_indices, long_firm_returns, short_firm_returns

def calculate_turnover(
    current_long_positions: dict,
    current_short_positions: dict,
    prev_long_positions: dict,
    prev_short_positions: dict
):
    """
    현재와 이전 포지션을 기반으로 턴오버율을 계산합니다.
    
    Args:
        current_long_positions: 현재 롱 포지션 {자산ID: 비중}
        current_short_positions: 현재 숏 포지션 {자산ID: 비중}
        prev_long_positions: 이전 롱 포지션 {자산ID: 비중}
        prev_short_positions: 이전 숏 포지션 {자산ID: 비중}
        
    Returns:
        tuple: (long_turnover, short_turnover, total_turnover)
    """
    # 롱 포지션 턴오버 계산
    long_turnover = 0.0
    if prev_long_positions or current_long_positions:
        # 1. 이전 롱 포지션에서 제거된 자산들
        for idx in prev_long_positions:
            if idx not in current_long_positions:
                weight_diff = abs(prev_long_positions[idx])
                long_turnover += weight_diff
        
        # 2. 새로 추가된 롱 포지션 자산들
        for idx in current_long_positions:
            if idx not in prev_long_positions:
                weight_diff = abs(current_long_positions[idx])
                long_turnover += weight_diff
        
        # 3. 비중이 변경된 롱 포지션 자산들
        for idx in set(prev_long_positions.keys()) & set(current_long_positions.keys()):
            weight_diff = abs(current_long_positions[idx] - prev_long_positions[idx])
            long_turnover += weight_diff
    
    # 숏 포지션 턴오버 계산
    short_turnover = 0.0
    if prev_short_positions or current_short_positions:
        # 1. 이전 숏 포지션에서 제거된 자산들
        for idx in prev_short_positions:
            if idx not in current_short_positions:
                weight_diff = abs(prev_short_positions[idx])
                short_turnover += weight_diff
        
        # 2. 새로 추가된 숏 포지션 자산들
        for idx in current_short_positions:
            if idx not in prev_short_positions:
                weight_diff = abs(current_short_positions[idx])
                short_turnover += weight_diff
        
        # 3. 비중이 변경된 숏 포지션 자산들
        for idx in set(prev_short_positions.keys()) & set(current_short_positions.keys()):
            weight_diff = abs(current_short_positions[idx] - prev_short_positions[idx])
            short_turnover += weight_diff
    
    # 전체 턴오버율 계산 (롱 + 숏) / 2
    total_turnover = (long_turnover + short_turnover) / 2
    
    return long_turnover, short_turnover, total_turnover 