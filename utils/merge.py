import h5py
import os
import pathlib
from pathlib import Path
from glob import glob
import numpy as np


def merge_h5py_files(file_list):
    # 파일들을 h5py.File(...)을 사용해 불러옵니다.
    h5_files = [h5py.File(f, 'r') for f in file_list]

    # 각 파일의 키를 사용하여 데이터를 불러와서 병합합니다.
    merged_data = {}
    for key in h5_files[0].keys():
        # 병합할 데이터를 저장하는 빈 리스트 생성
        combined_data = []
        
        for h5_file in h5_files:
            combined_data.append(h5_file[key][:])

        # NumPy array 형식으로 데이터 배열을 하나로 합칩니다.
        merged_data[key] = np.concatenate(combined_data, axis=0)

    # 열린 파일들을 닫습니다.
    for h5_file in h5_files:
        h5_file.close()

    return merged_data


# if __name__ =="__main__":
    
#     dir = pathlib.Path.cwd()
#     data_dir = Path(dir, 'FnDataLab/data/')
#     data_dir = os.path.join(data_dir, 'dset*.h5')
#     ls = glob(data_dir)

#     ls12 = ls[:2]
    
#     dataset = merge_h5py_files(ls12)
#     print(dataset)