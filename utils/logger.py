import logging
import os
from datetime import datetime

def get_logger(name, log_dir='log'):
    """
    로거를 설정하고 반환하는 함수
    
    Args:
        name (str): 로거 이름
        log_dir (str): 로그 파일을 저장할 디렉토리
    
    Returns:
        logging.Logger: 설정된 로거
    """
    # 로그 디렉토리 생성
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일명 설정 (날짜 + 모듈명)
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f"{date_str}_{name.replace('.', '_')}.log")
    
    # 로거 생성 및 레벨 설정
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 설정되어 있다면 추가하지 않음
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger