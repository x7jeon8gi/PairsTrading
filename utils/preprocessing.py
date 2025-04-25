import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from sklearn.impute import KNNImputer
import logging
from datetime import datetime
import re
from utils.logger import get_logger

# 기본 로거 설정 (main3.py에서 주입되지 않을 경우 사용)
logger = logging.getLogger(__name__)

# 선택할 컬럼 정의
SELECTED_COLS = [
    # PERMNO 표기
    'PERMNO',

    # 모멘텀 (필수, 주요 피처)
    'MOM1', 'MOM2', 'MOM3', 'MOM4', 'MOM5', 'MOM6',
    'MOM7', 'MOM8', 'MOM9', 'MOM10', 'MOM11', 'MOM12',
    'MOM13', 'MOM14', 'MOM15', 'MOM16', 'MOM17', 'MOM18',
    'MOM19', 'MOM20', 'MOM21', 'MOM22', 'MOM23', 'MOM24',

    # 재무정보 (기업 펀더멘탈, 추가 정보성 피처)
    'atq',      # 총자산
    'ltq',      # 총부채
    'dlcq',     # 단기부채
    'dlttq',    # 장기부채
    'seqq',     # 주주자본 (Equity)
    'cheq',     # 현금성 자산
    'saleq',    # 매출액
    'niq',      # 순이익
    'oiadpq',   # 영업이익
    'piq',      # 세전이익
    'dpq',      # 감가상각비
    'epspxq',   # 희석 EPS
]

def preprocess_data(df, file_name=None, logger=None):
    """
    데이터프레임을 전처리하는 함수
    
    Args:
        df (pd.DataFrame): 전처리할 데이터프레임
        file_name (str, optional): 처리 중인 파일 이름
        logger (Logger, optional): 로깅에 사용할 logger
    
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    # 로거 설정 - main3.py에서 주입되지 않을 경우 기본 로거 사용
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 파일명에서 년월 추출
    year_month = None
    if file_name:
        # 파일명에서 날짜 추출 (예: 20010131.csv -> 200101)
        match = re.search(r'(\d{6})\d{2}', os.path.basename(file_name))
        if match:
            year_month = match.group(1)
        else:
            # 다른 형식 시도 (예: data_2001_01.csv -> 200101)
            match = re.search(r'_(\d{4})_(\d{2})', os.path.basename(file_name))
            if match:
                year_month = f"{match.group(1)}{match.group(2)}"
    
    logger.info(f"전처리 시작: {file_name if file_name else '데이터프레임'}")
    logger.info(f"원본 데이터 크기: {df.shape}")
    
    # 선택한 컬럼만 추출
    try:
        df_selected = df[SELECTED_COLS].copy()
        logger.info(f"선택된 컬럼으로 데이터 추출 완료: {df_selected.shape}")
    except KeyError as e:
        logger.error(f"컬럼 선택 중 오류 발생: {e}")
        # 사용 가능한 컬럼만 선택
        available_cols = [col for col in SELECTED_COLS if col in df.columns]
        logger.info(f"사용 가능한 컬럼으로 대체: {available_cols}")
        df_selected = df[available_cols].copy()
    
    # PERMNO 컬럼 -> index
    if 'PERMNO' in df_selected.columns:
        df_selected.set_index('PERMNO', inplace=True)
        logger.info("PERMNO를 인덱스로 설정")
    
    # 결측치 처리
    df_selected = handle_missing_values(df_selected, logger, year_month)
    
    logger.info(f"전처리 완료: {df_selected.shape}")
    return df_selected

def handle_missing_values(df, logger=None, year_month=None):
    """
    결측치 처리 함수 (KNNImputer 사용)
    
    Args:
        df (pd.DataFrame): 결측치를 처리할 데이터프레임
        logger (Logger, optional): 로깅에 사용할 logger
        year_month (str, optional): 데이터의 년월 정보
    
    Returns:
        pd.DataFrame: 결측치가 처리된 데이터프레임
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 결측치 비율 확인
    missing_ratio = df.isnull().sum() / len(df)
    
    logger.info("결측치 비율:")
    for column, ratio in missing_ratio[missing_ratio > 0].items():
        logger.info(f"{column}: {ratio:.4f} ({int(ratio * len(df))} / {len(df)})")
    
    # 월별 결측치 비율 요약 로깅
    if year_month:
        summary_msg = f"[{year_month}] 결측치 요약: "
        summary_msg += ", ".join([f"{col}: {ratio:.2f}" for col, ratio in missing_ratio[missing_ratio > 0].items()])
        logger.info(summary_msg)
    
    # 결측치가 50% 이상인 컬럼 제거
    columns_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
    if columns_to_drop:
        logger.warning(f"다음 컬럼들이 결측치가 50% 이상이어서 제거됩니다: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    
    # KNNImputer를 사용하여 결측치 채우기
    if df.isnull().sum().sum() > 0:
        try:
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            df_imputed = pd.DataFrame(
                imputer.fit_transform(df),
                columns=df.columns,
                index=df.index
            )
            logger.info("KNNImputer를 사용하여 결측치를 채웠습니다.")
        except Exception as e:
            logger.error(f"KNNImputer 적용 중 오류 발생: {e}")
            logger.info("중앙값으로 결측치를 대체합니다.")
            df_imputed = df.fillna(df.median())
    else:
        logger.info("결측치가 없습니다.")
        df_imputed = df
    
    return df_imputed

def handle_outliers(df, logger=None):
    """
    이상치 처리 함수 (Modified Z-score 방식)
    
    Args:
        df (pd.DataFrame): 이상치를 처리할 데이터프레임
        logger (Logger, optional): 로깅에 사용할 logger
    
    Returns:
        pd.DataFrame: 이상치가 처리된 데이터프레임
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    df_clean = df.copy()
    total_outliers = 0
    
    for column in df.columns:
        # NaN 값을 제외하고 중앙값 계산
        median = df[column].dropna().median()
        
        # MAD(Median Absolute Deviation) 계산 - NaN 값 제외
        mad = np.nanmedian(np.abs(df[column].dropna() - median))

        # mad가 0이면 해당 컬럼은 스킵
        if mad == 0 or np.isnan(mad):
            logger.warning(f"{column}: MAD가 0이거나 NaN이라 이상치 탐지를 건너뜁니다.")
            continue
        
        # 수정된 Z-score 계산 (0.6745는 정규 분포에서 MAD를 표준편차로 변환하는 상수)
        modified_zscore = 0.6745 * (df[column] - median) / mad
        
        # 임계값 설정 (일반적으로 3.5 또는 3.0 사용)
        threshold = 3.5
        
        # 이상치 마스킹 (이상치를 NaN으로 설정)
        mask = abs(modified_zscore) > threshold
        df_clean.loc[mask, column] = np.nan
        
        # 마스킹된 데이터 수 출력
        n_outliers = np.sum(mask)
        total_outliers += n_outliers
        if n_outliers > 0:
            logger.info(f"{column}: {n_outliers}개의 이상치가 마스킹되었습니다.")
    
    logger.info(f"총 {total_outliers}개의 이상치가 마스킹되었습니다.")
    
    return df_clean

def normalize_data(df, logger=None):
    """
    데이터 정규화 함수 (StandardScaler 사용)
    
    Args:
        df (pd.DataFrame): 정규화할 데이터프레임
        logger (Logger, optional): 로깅에 사용할 logger
    
    Returns:
        pd.DataFrame: 정규화된 데이터프레임
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    try:
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(df)
        logger.info("데이터 정규화 완료")
        return pd.DataFrame(normalized_data, columns=df.columns, index=df.index)
    except Exception as e:
        logger.error(f"정규화 중 오류 발생: {e}")
        return df

def process_directory(input_dir, output_dir, logger=None):
    """
    디렉토리 내의 모든 CSV 파일을 처리하는 함수
    
    Args:
        input_dir (str): 입력 디렉토리 경로
        output_dir (str): 출력 디렉토리 경로
        logger (Logger, optional): 로깅에 사용할 logger
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('log', exist_ok=True)
    
    # 전체 파일 목록
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    logger.info(f"총 {len(file_list)}개의 파일을 처리합니다.")
    
    for idx, filename in enumerate(file_list):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        
        logger.info(f"[{idx+1}/{len(file_list)}] 처리 중: {filename}")
        
        try:
            df = pd.read_csv(input_path)
            df_processed = preprocess_data(df, input_path, logger)
            df_processed.to_csv(output_path, index=True)
            logger.info(f"전처리된 데이터가 {output_path}에 저장되었습니다.")
        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생: {e}")
    
    logger.info("모든 파일 처리 완료")

if __name__ == "__main__":
    # 예시 사용법
    # 단일 파일 처리
    # df = preprocess_data(pd.read_csv("path/to/your/data.csv"), "path/to/your/data.csv")
    
    # 디렉토리 내 모든 파일 처리
    # process_directory("path/to/input/directory", "path/to/output/directory")
    pass
