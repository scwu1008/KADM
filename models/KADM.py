import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .GlobalModel import GlobalModel
from .LocalModel import LocalModel

class KADM(nn.Module):
    def __init__(self, params):
        self.params = params
        super(KADM, self).__init__()
        self.local_model = LocalModel( 
            params.num_words, 
            params.word_dim, 
            params.MAX_DESC_LEN,
            params.MAX_NEIGH_LEN
            )
        self.global_model = GlobalModel(params)
        self.attention = nn.Linear(params.emb_dim*2, 2, bias=False)
        self.fc_layer = nn.Linear(params.emb_dim*2, 1, bias=False)
    
    def forward(self, u_matrix, i_matrix, graph, _ids, w_emb):
        u_local, i_local = self.local_model(self.params, u_matrix, i_matrix, w_emb)
        u_global, i_global = self.global_model(graph, _ids)
        u_weights = self.attention(torch.cat([u_local, u_global], -1))
        i_weights = self.attention(torch.cat([i_local, i_global], -1))
        u_final = u_local * u_weights[:,0] + u_global * u_weights[:,1]
        i_final =  i_local * i_weights[:,0] + i_global * i_weights[:,1]
        return self.fc_layer(torch.cat([u_final, i_final], -1))
    
    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(torch.load(path))
    
    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name
        