import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler

class Embedding_dataset(Dataset):
    def __init__(self, config, data=None, is_train=True):
        super().__init__()
        self.config = config
        self.is_train = is_train
        
        if isinstance(data, str):
            if data == "Train":
                self.data = pd.read_pickle(config['data']['train']).dropna()
            elif data == 'Test':
                self.data = pd.read_pickle(config['data']['test']).dropna()
        elif isinstance(data, Path):
            self.data = pd.read_pickle(str(data)).dropna()  # Convert Path to str if needed
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError('No such data')
        
        # 모든 값이 동일한 열은 제거
        self.data = self.data.loc[:, self.data.nunique() != 1]

        #! firm 열이 있으면 제거
        if 'firm' in self.data.columns:
            self.X = self.data.drop(columns=['firm'])
        else:
            self.X = self.data

        #! 'mom1' 열이 있으면 제거
        if 'mom1' in self.data.columns:
            # self.Y = self.data['mom1'].values
            self.X = self.X.drop(columns=['mom1'])
        else:
            pass

        self.num_features = self.X.shape[1]
        
        self.X = self._normalize(self.X)
        self.feature_n_bins = [config['model']['n_bins']] * self.num_features # todo: bin config
        self._prepare_embedding(self.X)
        self.augment_std = config['model']['augment_std']
        
        self.preprocessed_data = self._prepare_data(self.X)

    def __len__(self):
        return len(self.data)
    
    def _normalize(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def _prepare_embedding(self, X):
        # self.feature_n_bins = [64] * self.num_features # todo: bin config
        self.bin_edges = []
        for feature_idx in range(self.X.shape[1]):  
            quantiles = np.linspace(0.0, 1.0, self.feature_n_bins[feature_idx] + 1)
            self.bin_edges.append(np.unique(np.quantile(self.X[:, feature_idx], quantiles))) 

    def _apply_ple(self, X, add_cls=True):
        """
        On Embeddings for Numerical Features in Tabular Deep Learning(https://arxiv.org/pdf/2203.05556.pdf)
        위 논문의 컨셉을 유지하되, 완벽하게 동일한 구현은 아니며 많은 부분 생략되어 있음
        그러나 Piecewise linear encoding을 통한 embedding의 개념 자체는 동일
        Unsupervised 특성 상 Decision Tree를 통한 binning을 사용하지 않고, quantile을 통한 binning을 사용
        """
        N, M = X.shape
        cls_dim = 1 if add_cls else 0
        embedded = np.zeros((N, M + cls_dim, max(self.feature_n_bins)))
        
        if add_cls:
            # Initialize CLS token with random values or ones.
            # embedded[:, 0, :] = np.random.normal(0, 1, (N, max(self.feature_n_bins)))  # Random initialization
            embedded[:, 0, :] = np.ones((N, max(self.feature_n_bins)))  # Zero initialization
        
        for feature_idx in range(M):
            bins = np.digitize(X[:, feature_idx], np.r_[-np.inf, self.bin_edges[feature_idx][1:-1], np.inf]) - 1
            if bins.shape ==(1,):
                pass
            bin_mask = np.eye(self.feature_n_bins[feature_idx])[bins]
            x = bin_mask * self.bin_edges[feature_idx][1:][bins][:, None]
            previous_bins_mask = np.arange(self.feature_n_bins[feature_idx]) < bins[:, None]
            x[previous_bins_mask] = 1.0
            embedded[:, feature_idx + cls_dim, :self.feature_n_bins[feature_idx]] = x
                
        if N == 1:
            embedded = embedded.squeeze(0)
            
        return embedded
    
    def _augment_with_gaussian_noise(self, X):
        """
        매우 Simple한 augmentation
        TODO: 다른 augmentation 방법들을 고려해볼 것
        """
        if self.augment_std is None:
            std = 0.05
        else:
            std = self.augment_std
        return X + np.random.normal(0, std, X.shape)
    
    def _augment_with_random_masking(self, X, mask=0.1):
        """
        Random masking augmentation
        일부 Columm을 랜덤하게 0으로 만들어줌 
        """
        N, M = X.shape
        mask = np.random.binomial(1, 1- mask, (N, M))
        X = X * mask
        return X
    
    def _prepare_data(self, X):
        preprocessed_data = []

        for i in range(X.shape[0]):
            x = X[i].reshape(-1, self.num_features)

            if self.is_train:
                x_aug1 = self._augment_with_gaussian_noise(x)
                x_aug2 = self._augment_with_random_masking(x)
                x_1 = self._apply_ple(x_aug1, add_cls=True)
                x_2 = self._apply_ple(x_aug2, add_cls=True)
            else:
                x_1 = self._apply_ple(x, add_cls=True)
                x_2 = self._apply_ple(x, add_cls=True)

            preprocessed_data.append((x_1, x_2))
            
        return preprocessed_data
    
    def __getitem__(self, idx):
        # 사전 처리된 데이터 반환
        return [torch.FloatTensor(data) for data in self.preprocessed_data[idx]]

    # def __getitem__(self, idx):
    #     X = self.X[idx].reshape(-1, self.num_features)

    #     if self.is_train:
    #         # 2 different Augmentations
    #         X_aug1 = self._augment_with_gaussian_noise(X)
    #         X_aug2 = self._augment_with_random_masking(X)

    #         # Apply PLE
    #         X_1 = self._apply_ple(X_aug1, add_cls=True)
    #         X_2 = self._apply_ple(X_aug2, add_cls=True)
    #         # y= self.Y[idx]

    #     else:
    #         # No Augmentations during inference
    #         X_1 = self._apply_ple(X, add_cls=True)
    #         X_2 = self._apply_ple(X, add_cls=True)
    #         # y= self.Y[idx]

    #     return torch.FloatTensor(X_1), torch.FloatTensor(X_2), # torch.FloatTensor(y)

    # def __getitem__(self, idx):
    #     # Retrieve pre-processed data
    #     X = self.X[idx]
    #     X_aug = self.X_aug[idx]
    #     # y = self.Y[idx]  # Uncomment if you have labels
    #     return torch.FloatTensor(X), torch.FloatTensor(X_aug)  #, y