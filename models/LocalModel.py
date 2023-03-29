import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LocalModel(nn.Module):
    def __init__(self, nums_word, word_dim, max_desc_len, max_neigh_len):
        super(LocalModel, self).__init__()
        self.attention = nn.Sequential(
                nn.Linear(word_dim*2, word_dim, bias=False),
                nn.ReLU(),
                nn.Linear(word_dim, word_dim, bias=False),
                nn.ReLU(),
                nn.Linear(word_dim, 1, bias=False),
                nn.Sigmoid(),
                )
        self.col_norm = nn.Softmax(dim=2)
        self.col_pool = nn.MaxPool2d((1, max_neigh_len))
        self.row_norm = nn.Softmax(dim=1)
        self.row_pool = nn.MaxPool2d((max_neigh_len, 1))
        self.word_embddings = nn.Embedding(nums_word+1, word_dim)
        self._init_weight()
        
    def _init_weight(self):
        # init embedding
        nn.init.xavier_uniform_(self.word_embddings.weight, gain=nn.init.calculate_gain('relu'))
        # init attention
        for layer in self.attention:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        
    
    def forward(self, params, u_matrix, i_matrix, w_emb):
        """
        

        Args:
            u_matrix ([type]): [batch, max_neigh_len, max_desc_len]
            i_matrix ([type]): [batch, max_neigh_len, max_desc_len]

        Returns:
            [type]: [description]
        """
        # 学习每个item的description
        shape = u_matrix.shape
        # # shape: [batch*max_neigh_len, max_desc_len, word_dim]
        u_matrix = w_emb(u_matrix.view(shape[0]*shape[1], -1)).view(shape[0], shape[1], -1)
        # # shape: [batch*max_neigh_len, max_desc_len, word_dim]
        i_matrix = w_emb(i_matrix.view(shape[0]*shape[1], -1)).view(shape[0], shape[1], -1)
        
        # co-attention
        u_matrix_repeat = torch.repeat_interleave(u_matrix, i_matrix.shape[1], axis=1)
        i_matrix_repeat = i_matrix.repeat(1, u_matrix.shape[1], 1)
        # shape: [batch, max_neigh_len*max_neigh_len, filter_num*2]
        h = torch.cat([u_matrix_repeat, i_matrix_repeat], -1)
        weights = self.attention(h).view(u_matrix.shape[0], u_matrix.shape[1], -1)
        u_weight = self.col_pool(weights).view(u_matrix.shape[0], -1, 1)
        i_weight = self.row_pool(weights).view(i_matrix.shape[0], -1, 1)
        u_features = torch.mean(u_matrix*u_weight, axis=1).view(u_matrix.shape[0], -1)
        v_features = torch.mean(i_weight*i_matrix, axis=1).view(i_matrix.shape[0], -1)
        return u_features, v_features
        


