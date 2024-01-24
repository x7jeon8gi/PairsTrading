import os
import torch
import numpy as np
from tqdm.auto import tqdm

def inference(loader, model, device, numpy_array=True):
    """
    Input
    ----------
    loader: DataLoader
    model: nn.Module
    device: torch.device("cuda" or "cpu")
    
    Output
    ----------
    feature_vectors: (N, D) 개별 인스턴스들의 임베딩
    label_vectors: (N, C) 개별 인스턴스들의 임베딩의 라벨 확률 분포
    cluster_vectors: (N, ) 개별 인스턴스들의 임베딩의 라벨
    """
    model.eval()
    
    feature_vectors = []
    label_vectors = []
    cluster_vectors = []
    
    for x_w, x_s in tqdm(loader, position=1):
        x_w = x_w.to(device)
        
        with torch.no_grad():
            feature_vector, label_vector = model.forward_zc(x_w)
            feature_vectors.extend(feature_vector.detach().cpu().numpy())
            
            cluster_vector = torch.argmax(label_vector, dim=1).detach().cpu().numpy()
            cluster_vectors.extend(cluster_vector)
            
            label_vector = label_vector.detach().cpu().numpy()
            label_vectors.extend(label_vector)
            
    if numpy_array:
        feature_vectors = np.array(feature_vectors)
        label_vectors = np.array(label_vectors)
        cluster_vectors = np.array(cluster_vectors)
    
    return feature_vectors, label_vectors, cluster_vectors
            