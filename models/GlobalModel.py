from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch
import os
import numpy as np


class GlobalModel(nn.Module):
    def __init__(self, params):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        # self.relation2id = relation2id

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)
        

        self._init_weight(params)
    def _init_weight(self, params):
        # init embedding
        if params.use_kge_embeddings:
            path = '{}/{}_{}'.format(params.kge_dir, params.kge_model, params.dataset)
            node_features = np.load(os.path.join(path, 'relation_embeddings_{}_{}.npy'.format(params.kge_model, params.emb_dim)))
            self.rel_emb.weight.data.copy_(torch.from_numpy(node_features[:self.params.num_rels]))
            if params.finetune_ebd:
                self.rel_emb.weight.requires_grad = True
            else:
                self.rel_emb.weight.requires_grad = False
        else:
            nn.init.xavier_uniform_(self.rel_emb.weight, gain=nn.init.calculate_gain('relu'))
        # nn.init.xavier_uniform_(self.relation_emb.weight)
        # init attention
        for layer in self.fc_layer:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                
    def forward(self, data, _ids):
        g = data
        g.ndata['h'] = self.gnn(g)

        g_out = mean_nodes(g, 'repr')
        shape = g_out.shape
        for i, _id in enumerate(_ids):
            head_ids = (g.ndata['id'] == _id+1).nonzero().squeeze(1)
            head_embs = g.ndata['repr'][head_ids].view(-1, self.params.num_gcn_layers, self.params.emb_dim)
            if i == 0:
                u_embs = torch.mean(head_embs, axis=1)
                u_embs = torch.mean(head_embs, axis=0).view(1, -1)
            else:
                temp = torch.mean(head_embs, axis=1)
                temp = torch.mean(temp, axis=0).view(1, -1)
                u_embs = torch.cat((u_embs, temp), axis=0)

        tail_ids = (g.ndata['id'] == -1).nonzero().squeeze(1)
        tail_embs = torch.mean(g.ndata['repr'][tail_ids], axis=1)
        
        return u_embs, tail_embs