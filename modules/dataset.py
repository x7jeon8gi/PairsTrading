import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy

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

        self.num_features = self.X.shape[1]
        
        self.X = self._normalize(self.X)
        self.feature_n_bins = [config['model']['n_bins']] * self.num_features # todo: bin config
        self._prepare_embedding(self.X)
        self.augment_std = config['model']['augment_std']
        self.masking_ratio = config['model']['masking_ratio']

        # * (*, num_features) -> (*, num_features, n_bins)
        self.preprocessed_data = self._prepare_data(self.X, cls_init=config['model']['cls_init'])

    def __len__(self):
        return len(self.data)
    
    def _normalize(self, X, type='standard'):
        if type == 'standard':
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        elif type == 'minmax':
            scaler = MinMaxScaler()
            return scaler.fit_transform(X)
        else:
            raise ValueError('Invalid normalization type')

    def _prepare_embedding(self, X):
        # self.feature_n_bins = [64] * self.num_features # todo: bin config
        self.bin_edges = []
        for feature_idx in range(self.X.shape[1]):  
            quantiles = np.linspace(0.0, 1.0, self.feature_n_bins[feature_idx] + 1)
            self.bin_edges.append(np.unique(np.quantile(self.X[:, feature_idx], quantiles))) 

    def _apply_ple(self, X, add_cls=True, cls_init='random'):
        """
        On Embeddings for Numerical Features in Tabular Deep Learning(https://arxiv.org/pdf/2203.05556.pdf)
        위 논문의 컨셉을 유지하되, 완벽하게 동일한 구현은 아니며 많은 부분 생략되어 있음
        그러나 Piecewise linear encoding을 통한 embedding의 개념 자체는 동일
        Unsupervised 특성 상 Decision Tree를 통한 binning을 사용하지 않고, quantile을 통한 binning을 사용
        """

        # # Pytorch로 구현
        # N, M = X.shape
        # cls_dim = 1 if add_cls else 0
        # # PyTorch 텐서로 `embedded` 초기화
        # embedded = torch.zeros(N, M + cls_dim, max(self.feature_n_bins))
        
        # if add_cls:
        #     # Initialize CLS token with random values, ones, or as a trainable parameter
        #     if cls_init == 'random':
        #         cls_values = torch.randn(N, 1, max(self.feature_n_bins))
        #     elif cls_init == 'ones':
        #         cls_values = torch.ones(N, 1, max(self.feature_n_bins))
        #     elif cls_init == 'learnable':
        #         # 학습 가능한 CLS 토큰 생성
        #         self.cls_token = nn.Parameter(torch.randn(1, max(self.feature_n_bins)))
        #         cls_values = self.cls_token.expand(N, -1).unsqueeze(1)
        #     else:
        #         raise ValueError('Invalid cls_init')

        #     embedded[:, 0, :] = cls_values.squeeze()

        # for feature_idx in range(1,M):
        #     # Binning using PyTorch operations, assuming `self.bin_edges` is a PyTorch tensor
        #     bin_edges_tensor = torch.cat([
        #         torch.tensor([-np.inf], dtype=torch.float32),
        #         torch.tensor(self.bin_edges[feature_idx], dtype=torch.float32)[1:-1],
        #         torch.tensor([np.inf], dtype=torch.float32)
        #     ])

        #     bins = torch.bucketize(X[:, feature_idx], bin_edges_tensor, right=True) - 1
        #     bin_mask = torch.nn.functional.one_hot(bins, num_classes=self.feature_n_bins[feature_idx]).float()
        #     x = bin_mask * bin_edges_tensor[1:][bins].unsqueeze(1)

        #     previous_bins_mask = torch.arange(self.feature_n_bins[feature_idx]).unsqueeze(0) < bins.unsqueeze(1)
        #     x[previous_bins_mask] = 1.0
        #     embedded[:, feature_idx + cls_dim, :self.feature_n_bins[feature_idx]] = x
                    
        # if N == 1:
        #     embedded = embedded.squeeze(0)
            
        # return embedded

        N, M = X.shape

        # cls_dim = 1 if add_cls else 0
        # embedded = np.zeros((N, M + cls_dim, max(self.feature_n_bins)))
    
        # if add_cls:
        #     # Initialize CLS token with random values or ones.
        #     if cls_init == 'random':
        #         embedded[:, 0, :] = np.random.normal(0, 1, (N, max(self.feature_n_bins)))
        #     elif cls_init == 'ones':
        #         embedded[:, 0, :] = np.ones((N, max(self.feature_n_bins)))  # One initialization
        #     else:
        #         raise ValueError('Invalid cls_init')
        
        cls_dim = 0
        embedded = np.zeros((N, M +cls_dim, max(self.feature_n_bins))) # ! ?

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
    
    def _augment_with_random_masking(self, X):
        """
        Random masking augmentation
        일부 Columm을 랜덤하게 0으로 만들어줌 
        """
        if self.masking_ratio is None:
            masking_ratio = 0.01
        else:
            masking_ratio = self.masking_ratio
        N, M = X.shape
        mask = np.random.binomial(1, 1- masking_ratio, (N, M))
        X = X * mask
        return X
    
    def _prepare_data(self, X, Y=None, cls_init='random'):
        preprocessed_data = []

        for i in range(X.shape[0]):
            x = X[i].reshape(-1, self.num_features)

            if self.is_train:
                x_aug1 = self._augment_with_gaussian_noise(x)
                x_aug2 = self._augment_with_random_masking(x) #: random masking은 성능을 degrading 시킬 수 있음.
                # x_aug2 = copy.deepcopy(x)
                x_1 = self._apply_ple(x_aug1, add_cls=True, cls_init=cls_init)
                x_2 = self._apply_ple(x_aug2, add_cls=True, cls_init=cls_init)
            else:
                x_1 = self._apply_ple(x, add_cls=True, cls_init=cls_init)
                x_2 = self._apply_ple(x, add_cls=True, cls_init=cls_init)

            preprocessed_data.append((x_1, x_2))
            
        return preprocessed_data
    
    def __getitem__(self, idx):
        # 사전 처리된 데이터 반환
        data = self.preprocessed_data[idx]
        x_w = torch.FloatTensor(data[0])  
        x_s = torch.FloatTensor(data[1])
        return x_w, x_s