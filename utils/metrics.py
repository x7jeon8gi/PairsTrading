from typing import Union, Tuple, Optional, List
import numpy as np
import torch
from sklearn import metrics
import pandas as pd 

def cal_mse(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    ) -> Union[float, torch.Tensor]:
    
    
    assert type(predictions) == type(targets), (
        f"types of inputs and target must match, but got"
        f"type(inputs)={type(predictions)}, type(target)={type(targets)}"
    )
    
    lib = np if isinstance(predictions, np.ndarray) else torch
    
    return lib.mean(lib.square(predictions - targets))

def calculate_financial_metrics_monthly(monthly_log_returns_df, risk_free_rate=0.0, verbose=False):
    """
    월별 log return 데이터로부터 누적 log return을 계산하고,
    이를 기반으로 연간화된 수익률, 변동성, Sharpe, Sortino, MDD, Calmar Ratio 등을 산출합니다.
    
    Parameters:
    - monthly_log_returns_df: 월별 log return 데이터가 저장된 DataFrame. (컬럼은 모델/전략별)
    - risk_free_rate: 무위험 이자율 (연간 기준), 기본값 0.0
    - verbose: 중간 계산 결과를 출력할지 여부, 기본값 False
    
    Returns:
    - metrics_df: 각 모델별로 계산된 재무 지표들을 담은 DataFrame.
    """
    # 입력 데이터가 DataFrame인지 확인
    if not isinstance(monthly_log_returns_df, pd.DataFrame):
        if isinstance(monthly_log_returns_df, pd.Series):
            monthly_log_returns_df = pd.DataFrame(monthly_log_returns_df)
        else:
            raise TypeError("Input must be a pandas DataFrame or Series")
    
    # NaN 값 확인 및 경고
    if monthly_log_returns_df.isna().any().any():
        print("경고: 입력 데이터에 NaN 값이 포함되어 있습니다. 계산 결과가 정확하지 않을 수 있습니다.")
        # NaN은 0으로 대체 (다른 방법도 가능)
        monthly_log_returns_df = monthly_log_returns_df.fillna(0)
    
    metrics = {}
    # 이미 월별 log return이 주어졌으므로 누적 log return 계산 (시작점 0)
    cum_log_returns_df = monthly_log_returns_df.cumsum()
    
    # 월별 데이터 → 연 12회
    periods_per_year = 12
    
    for model in monthly_log_returns_df.columns:
        cum_log_returns = cum_log_returns_df[model]
        period_returns = monthly_log_returns_df[model]
        
        # 데이터 길이 확인
        if len(period_returns) < 2:
            print(f"경고: 모델 {model}의 데이터 길이({len(period_returns)})가 너무 짧습니다. 최소 2개월 이상의 데이터가 필요합니다.")
            metrics[model] = {
                "Annualized Return": np.nan,
                "Annual Std": np.nan,
                "MDD": np.nan,
                "Sharpe Ratio": np.nan,
                "Sortino Ratio": np.nan,
                "Calmar Ratio": np.nan,
            }
            continue
        
        # 총 기간을 연 단위로 변환
        total_years = len(cum_log_returns) / periods_per_year
        
        # Annualized Return: 누적 log return의 연평균 효과를 간단 수익률로 변환
        # 계산 로직 확인: 누적 로그 수익률을 연간화한 후 일반 수익률로 변환
        final_log_return = cum_log_returns.iloc[-1]
        # np.expm1(x) is more accurate than np.exp(x) - 1 for small values of x
        ar = np.expm1(final_log_return / total_years)
        
        # Annualized Standard Deviation
        std = period_returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe Ratio
        sharpe_ratio = (ar - risk_free_rate) / std if std != 0 else np.nan
        
        # Sortino Ratio: 음의 return에 대한 리스크만 고려
        downside = period_returns[period_returns < 0]
        if not downside.empty:
            # 다운사이드 위험 계산 개선: 제곱평균제곱근(RMS) 사용
            downside_risk = np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year)
            sortino_ratio = (ar - risk_free_rate) / downside_risk if downside_risk != 0 else np.nan
        else:
            # 음의 수익률이 없는 경우
            downside_risk = 0
            sortino_ratio = np.inf  # 완벽한 경우 무한대로 설정
        
        # Maximum Drawdown (MDD)
        # 누적 로그 수익률을 단순 수익률로 전환 (exp(log return) = simple return + 1)
        # np.expm1(x) is more accurate than np.exp(x) - 1 for small values of x
        cumulative_returns = np.expm1(cum_log_returns)
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / (peak + 1)  # 수정: 분모에 1 추가해서 정확한 drawdown 계산
        mdd = drawdown.min()
        
        # Calmar Ratio: 연간 수익률 대비 최대 낙폭 비율
        calmar_ratio = ar / abs(mdd) if mdd != 0 else np.inf
        
        # 계산 결과 출력 (디버깅용)
        if verbose:
            print(f"\n=== {model} 모델 계산 결과 ===")
            print(f"데이터 길이: {len(period_returns)}개월 ({total_years:.2f}년)")
            print(f"누적 로그 수익률: {final_log_return:.4f}")
            print(f"연간화 수익률: {ar:.4f} ({ar*100:.2f}%)")
            print(f"연간화 표준편차: {std:.4f}")
            print(f"다운사이드 위험: {downside_risk if 'downside_risk' in locals() else 'N/A'}")
            print(f"최대 낙폭(MDD): {mdd:.4f} ({mdd*100:.2f}%)")
            print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            print(f"Sortino Ratio: {sortino_ratio:.4f}")
            print(f"Calmar Ratio: {calmar_ratio:.4f}")
        
        # 결과 저장
        metrics[model] = {
            "Annualized Return": ar,
            "Annual Std": std,
            "MDD": mdd,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Calmar Ratio": calmar_ratio,
        }
    
    # 결과 DataFrame 생성 및 반환
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=[
        "Annualized Return", "Annual Std", "MDD", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"
    ]).round(4)
    
    return metrics_df

def calculate_metrics(log_returns_list, target_return=0.0):
    # Convert log returns to simple returns (using expm1 for better numerical stability)
    simple_returns = np.expm1(log_returns_list)
    target_return_simple = np.expm1(target_return)

    # Max Drawdown Percent
    def calculate_mdd(log_returns_list):
        cum_log_return = np.cumsum(log_returns_list)
        cumulative_returns = np.expm1(cum_log_return)
        max_cumulative_returns = pd.Series(cumulative_returns).cummax()
        drawdown = (cumulative_returns - max_cumulative_returns) / max_cumulative_returns
        mdd = drawdown.min()
        return mdd

    def calculate_sortino_ratio(returns, target_return=0):
        expected_return = np.mean(returns)
        downside_returns = returns[returns < target_return] - target_return
        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        sortino_ratio = (expected_return - target_return) / downside_deviation if downside_deviation != 0 else np.nan
        return sortino_ratio, downside_deviation

    max_drawdown = calculate_mdd(log_returns_list)
    
    # Annual Return (using geometric mean)
    average_log_return = np.mean(log_returns_list)
    annual_return = np.expm1(average_log_return * 12)

    # Standard deviation (convert log returns to simple returns)
    monthly_std_simple = np.std(simple_returns, ddof=1)
    standard_deviation = monthly_std_simple * np.sqrt(12)
    
    # Sharpe Ratio
    sharpe_ratio = annual_return / standard_deviation if standard_deviation != 0 else np.nan

    # Sortino Ratio
    sortino_ratio, downside_deviation = calculate_sortino_ratio(simple_returns, target_return_simple)

    return sharpe_ratio, annual_return, standard_deviation, max_drawdown, sortino_ratio, downside_deviation