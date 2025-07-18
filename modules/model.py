import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import ContinuousTransformerWrapper, Encoder

class Network(nn.Module):
    """
    Backbone Network
    x_transformers 라이브러리를 사용하여 구현 (!pip install x-transformers)
    projectors는 각각 instance projector와 cluster projector로 구성
    projector 구성에 따라 모델의 성능이 달라질 수 있음
    """
    def __init__(self, 
                 cluster_num, 
                 dim_in_out,
                 num_features, 
                 hidden_dim, 
                 depth, heads, 
                 pre_norm, 
                 use_simple_rmsnorm,
                 cls_init='ones',
                 dropout_mask=0.1):
        
        super(Network, self).__init__()

        self.hidden_dim = hidden_dim
        self.dim = dim_in_out
        self.cluster_num = cluster_num
        self.dropout_mask = dropout_mask

        if cls_init == 'ones':
            self.cls_token = nn.Parameter(torch.ones(1, self.dim))
        elif cls_init == 'random':
            self.cls_token = nn.Parameter(torch.randn(1, self.dim))
        else:
            raise ValueError('Invalid cls_init')
        self.emb_linear = nn.Linear(self.dim, self.hidden_dim)
        self.emb_activation = nn.GELU()

        self.transformer = ContinuousTransformerWrapper(
            dim_in = hidden_dim,
            dim_out = hidden_dim,
            max_seq_len = num_features + 1,
            emb_dropout = self.dropout_mask,
            
            attn_layers = Encoder(
                dim= hidden_dim,
                depth = depth,
                heads = heads,
                attn_dropout = self.dropout_mask,
                ff_dropout = self.dropout_mask,
                pre_norm = pre_norm,
                use_simple_rmsnorm = use_simple_rmsnorm
            )
        )

        self.instance_projector = nn.Sequential(
            nn.Linear(self.hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_mask),
            nn.Linear(hidden_dim, self.dim),
        )
        self.cluster_projector = nn.Sequential(
            # nn.BatchNorm1d(hidden_dim), # If use small batch size, It could be worse
            nn.Linear(self.hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_mask),
            nn.Linear(hidden_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x_i, x_j, return_ci=True):
        
        cls_tokens = self.cls_token.expand(x_i.size(0), -1).unsqueeze(1)
        x_i = torch.cat((cls_tokens, x_i), dim=1)
        x_j = torch.cat((cls_tokens, x_j), dim=1)

        x_i = self.emb_linear(x_i)
        x_j = self.emb_linear(x_j)
        
        x_i = self.emb_activation(x_i)
        x_j = self.emb_activation(x_j)

        h_i = self.transformer(x_i) # Batch, seq_len, dim
        h_j = self.transformer(x_j)
        
        # Get CLS Representation
        h_i_cls = h_i[:,0,:] # (Batch, Dim)
        h_j_cls = h_j[:,0,:]
        
        z_i = F.normalize(self.instance_projector(h_i_cls), dim=1)
        z_j = F.normalize(self.instance_projector(h_j_cls), dim=1)
        
        c_j = self.cluster_projector(h_j_cls)

        if return_ci:
            c_i = self.cluster_projector(h_i_cls)
            return z_i, z_j, c_i, c_j
        else:
            return z_i, z_j, c_j
        
    def forward_c(self, x):
        cls_tokens = self.cls_token.expand(x.size(0), -1).unsqueeze(1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.emb_linear(x)
        x = self.emb_activation(x)

        h = self.transformer(x)
        h = h[:,0,:]
        c = self.cluster_projector(h)
        c = F.softmax(c, dim=1)
        return c

    def forward_zc(self, x):
        cls_tokens = self.cls_token.expand(x.size(0), -1).unsqueeze(1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.emb_linear(x)
        x = self.emb_activation(x)

        h = self.transformer(x)
        h = h[:,0,:]
        
        z = F.normalize(self.instance_projector(h), dim=1)
        c = self.cluster_projector(h)
        c = F.softmax(c, dim=1)
        return z, c
