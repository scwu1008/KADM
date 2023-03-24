import os
import lmdb
import dgl
import json
import torch
import logging
import numpy as np
import random
import collections
from re import sub
from typing import ItemsView
from torch.utils import data
from torchtext.vocab import Vocab, Vectors
from utils.data_utils import read_KG, read_data, read_description, read_neighbor, tprint
from subgraph_extraction.graph_sampler import extract_save_subgraph, links2subgraphs
from utils.graph_utils import ssp_multigraph_to_dgl, deserialize
from dgl.data.utils import save_graphs, load_graphs

# random.seed(2021)


def load_word_embeddings(params):
    def _read_words(path):
        '''
            Count the occurrences of all words
            @param path: item2desc dict file
            @return words: list of words (with duplicates)
        '''
        item2desc = json.load(open(path))
        words = []
        for example in item2desc:
            words += item2desc[example].split()
        return words
    tprint('Loading word vectors')
    path = os.path.join(params.wv_path, params.word_vector)
    if not os.path.exists(path):
        # Download the word vector and save it locally:
        tprint('Downloading word vectors')
        import urllib.request
        urllib.request.urlretrieve(
            'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec',
            path)

    vectors = Vectors(params.word_vector, cache=params.wv_path)
    vocab = Vocab(collections.Counter(_read_words(os.path.join(params.root, 'item2desc.json'))), vectors=vectors,
                  specials=['<pad>', '<unk>'], min_freq=5)

    # print word embedding statistics
    wv_size = vocab.vectors.size()
    tprint('Total num. of words: {}, word vector dimension: {}'.format(
        wv_size[0],
        wv_size[1]))

    num_oov = wv_size[0] - torch.nonzero(
            torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
    tprint(('Num. of out-of-vocabulary words'
           '(they are initialized to zeros): {}').format( num_oov))
    return vocab

def generate_subgraph_datasets(params, splits):
    root = params.root
    MAX_NEIGH_LEN = params.MAX_NEIGH_LEN
    graphs = {}
    for split in splits:
        if split == 'test':
            inter_data = read_data(os.path.join(root, "test_all.csv"))
                # self.user2candidates = json.load(open(os.path.join(root, 'user2can.json')))
        elif split == 'train':
            inter_data = read_data(os.path.join(root, "train_data_neg.csv"))
        else:
            inter_data = read_data(os.path.join(root, "valid_all.csv"))
                # self.user2candidates = json.load(open(os.path.join(root, 'user2can.json')))
        graphs[split] = {"triplets": inter_data}
        
    # TODO 这里注意 zero-padding，即item和entity索引应该从 1 开始，0用于代表padding值
    i2e = json.load(open(os.path.join(root, 'item2ent.json')))
    logging.info("finish reading item2ent mapping data...")
    user2history = read_neighbor(os.path.join(root, 'user2his.json'), MAX_NEIGH_LEN)
    logging.info("finish reading user2his mapping data...")
    adj_list, _, _, _, _, _ = read_KG(root)
    e2id = json.load(open(os.path.join(root, "entity2id.json")))
    relation2id = json.load(open(os.path.join(root, "relation2id.json")))
    params.e2id = e2id
    params.i2e = i2e
    params.u2his = user2history
    logging.info("finish reading source data...")

    links2subgraphs(adj_list, graphs, params, splits[0])

def get_kge_embeddings(params):

    path = '{}/{}_{}'.format(params.kge_dir, params.kge_model, params.dataset)
    node_features = np.load(os.path.join(path, 'entity_embeddings_{}_{}.npy'.format(params.kge_model, params.emb_dim)))
    with open(os.path.join(params.root, 'entity2id.json')) as json_file:
        kge_entity2id = json.load(json_file)
        # kge_id2entity = {v: int(k) for k, v in kge_entity2id.items()}

    return node_features, kge_entity2id

class RecData(data.Dataset):
    """
    数据集加载与预处理

    """

    def __init__(self, 
                 root, 
                 params,
                 db_path,
                 train=True, 
                 test=False, 
                 dgl_graph=None,
                 MAX_DESC_LEN=20, 
                 MAX_NEIGH_LEN=20, 
                 vocab=None,
                 logging=None) -> None:
        """
        初始化数据集信息，主要目标获取所有数据，并根据训练、验证、测试划分数据

        Args:
            root (string): 数据集所在目录路径信息
            train (bool, optional): 标明是否为训练阶段. Defaults to True.
            test (bool, optional): 标明是否为测试阶段. Defaults to False.
            logging (object): 日志操作
        """
        super().__init__()
        self.params = params
        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        
        self.test = test
        self.train = train
        self.G = dgl_graph
        self.max_desc_len = MAX_DESC_LEN
        self.max_neigh_len = MAX_NEIGH_LEN
        self.node_features, self.kge_entity2id = get_kge_embeddings(params) if params.use_kge_embeddings else (None, None)
        
        if self.test:
            self.inter_data = read_data(os.path.join(root, "test_all.csv"))
            self.db = self.main_env.open_db("test".encode())
            # self.user2candidates = json.load(open(os.path.join(root, 'user2can.json')))
        elif self.train:
            self.inter_data = read_data(os.path.join(root, "train_data_neg.csv"))
            self.db = self.main_env.open_db("train".encode())
            self.all_items = set(self.inter_data[:, 1])
        else:
            self.inter_data = read_data(os.path.join(root, "valid_all.csv"))
            self.db = self.main_env.open_db("valid".encode())
            # self.user2candidates = json.load(open(os.path.join(root, 'user2can.json')))
        
        # TODO 这里注意 zero-padding，即item和entity索引应该从 1 开始，0用于代表padding值
        self.i2e = json.load(open(os.path.join(root, 'item2ent.json')))
        logging.info("finish reading item2ent mapping data...")
        # self.word2id = json.load(open(os.path.join(root, 'word2id.json')))
        # logging.info("finish reading word2id mapping data...")
        # self.nums_word = len(self.word2id)
        self.user2history = read_neighbor(os.path.join(root, 'user2his.json'), MAX_NEIGH_LEN)
        self.all_user_his = json.load(open(os.path.join(root, 'user2his.json'), "r"))
        logging.info("finish reading user2his mapping data...")
        # 读取item的 Collaborative neighbors
        self.item2neigh = read_neighbor(os.path.join(root, 'item2neigh.json'), MAX_NEIGH_LEN)
        logging.info("finish reading item2neigh mapping data...")
        # self.item2desc = read_description(os.path.join(root, 'item2desc.json'), MAX_DESC_LEN, self.word2id)
        self.item2desc = read_description(os.path.join(root, 'item2desc.json'), MAX_DESC_LEN, vocab)
        logging.info("finish reading item2desc mapping data...")
        
        if os.path.exists(os.path.join(root, 'graph.bin')):
            self.adj_list, _, _, _, _, _ = read_KG(root)
            graph, _ = load_graphs(os.path.join(root, 'graph.bin'), [0])
            self.graph = graph[0]
            self.e2id = json.load(open(os.path.join(root, "entity2id.json")))
            self.relation2id = json.load(open(os.path.join(root, "relation2id.json")))
            self.num_rels = len(self.relation2id)
            self.nums_ent = len(self.e2id)
            self.aug_num_rels = len(self.relation2id)
        else:
            ssp_graph, triplets, entity2id, elation2id, id2entity, id2relation = read_KG(root)
            self.e2id = entity2id
            self.adj_list = ssp_graph
            self.num_rels = len(ssp_graph)
            self.nums_ent = len(entity2id)
            # Add transpose matrices to handle both directions of relations.
            if self.params.add_traspose_rels:
                ssp_graph_t = [adj.T for adj in ssp_graph]
                ssp_graph += ssp_graph_t

            # the effective number of relations after adding symmetric adjacency matrices and/or self connections
            self.aug_num_rels = len(ssp_graph)
            self.graph = ssp_multigraph_to_dgl(ssp_graph)
            save_graphs(os.path.join(root, 'graph.bin'), [self.graph])
        logging.info("finish construct graph data...")
        logging.info("finish reading source data...")
        
    
    def __getitem__(self, index):
        _id = index
        if self.train:
            uid, iid, label, _, neg_iid = self.inter_data[index]
            all_his_of_u = set(self.all_user_his[str(uid)])
            # neg_iid = random.sample(list(self.all_items-all_his_of_u), 1)[0]
            neighbors_of_neg_item = self.item2neigh[neg_iid]
            # description_of_item = self.item2desc[iid]
            desc_of_neg_item_neighbor = np.array(list(map(lambda x: self.item2desc[x] if x!=0 else [0 for _ in range(self.max_desc_len)], neighbors_of_neg_item)))
            if len(desc_of_neg_item_neighbor) < self.max_neigh_len:
                desc_of_neg_item_neighbor = np.pad(desc_of_neg_item_neighbor, ((0, self.max_neigh_len-len(desc_of_neg_item_neighbor)), (0,0)), 'constant', constant_values = (0,0))
        else:
            uid, iid, label = self.inter_data[index]
        history_iids_of_user = self.user2history[uid]
        history_eids_of_user = list(map(lambda iid: int(self.e2id[self.i2e[str(iid)]]), history_iids_of_user))
        desc_of_item_hist = np.array(list(map(lambda x: self.item2desc[x], history_iids_of_user)))
        if len(desc_of_item_hist) < self.max_neigh_len:
            desc_of_item_hist = np.pad(desc_of_item_hist, ((0, self.max_neigh_len-len(desc_of_item_hist)), (0,0)), 'constant',constant_values = (0,0))
        
        neighbors_of_item = self.item2neigh[iid]
        # description_of_item = self.item2desc[iid]
        desc_of_item_neighbor = np.array(list(map(lambda x: self.item2desc[x] if x!=0 else [0 for _ in range(self.max_desc_len)], neighbors_of_item)))
        if len(desc_of_item_neighbor) < self.max_neigh_len:
            desc_of_item_neighbor = np.pad(desc_of_item_neighbor, ((0, self.max_neigh_len-len(desc_of_item_neighbor)), (0,0)), 'constant',constant_values = (0,0))
        
        if not self.train:
            with self.main_env.begin(db=self.db) as txn:
                str_id = '{:08}'.format(index).encode('ascii')
                enclosing_subgraph_nodes, pos_n_labels = deserialize(txn.get(str_id), False).values()
                # nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id), True).values()
                subgraph_pos = self._prepare_subgraphs(enclosing_subgraph_nodes, label, pos_n_labels, _id)
                # subgraph_neg = self._prepare_subgraphs(neg_enclosing_subgraph_nodes, label, neg_n_labels)
            # target_eid = int(self.e2id[self.i2e[str(iid)]])
            # args_ = [history_eids_of_user, target_eid, label]
            # enclosing_subgraph_nodes, r_label, n_labels = extract_save_subgraph(args_, self.adj_list, self.params)
            # subgraph = self._prepare_subgraphs(enclosing_subgraph_nodes, r_label, n_labels, _id)
            return uid, iid, label, \
                desc_of_item_hist.tolist(), desc_of_item_neighbor.tolist(), \
                    subgraph_pos, _id
        else:
            with self.main_env.begin(db=self.db) as txn:
                str_id = '{:08}'.format(index).encode('ascii')
                enclosing_subgraph_nodes, neg_enclosing_subgraph_nodes, pos_n_labels, neg_n_labels = deserialize(txn.get(str_id), True).values()
                # nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id), True).values()
                subgraph_pos = self._prepare_subgraphs(enclosing_subgraph_nodes, label, pos_n_labels, _id)
                subgraph_neg = self._prepare_subgraphs(neg_enclosing_subgraph_nodes, label, neg_n_labels, _id)
            
            
            # target_eid = int(self.e2id[self.i2e[str(iid)]])
            # args_ = [history_eids_of_user, target_eid, label]
            # enclosing_subgraph_nodes, r_label, n_labels = extract_save_subgraph(args_, self.adj_list, self.params)
            # subgraph = self._prepare_subgraphs(enclosing_subgraph_nodes, r_label, n_labels, _id)
            
            # neg_target_eid = int(self.e2id[self.i2e[str(neg_iid)]])
            # neg_args_ = [history_eids_of_user, neg_target_eid, label]
            # neg_enclosing_subgraph_nodes, r_label, n_labels = extract_save_subgraph(neg_args_, self.adj_list, self.params)
            # neg_subgraph = self._prepare_subgraphs(neg_enclosing_subgraph_nodes, r_label, n_labels, _id)
            
            return uid, iid, label, \
                desc_of_item_hist.tolist(), desc_of_item_neighbor.tolist(), desc_of_neg_item_neighbor.tolist(), \
                    subgraph_pos, subgraph_neg, _id
        
        # TODO 处理subgraph的构造和features初始化的问题
    def __len__(self):
        return len(self.inter_data)
    
    def _prepare_subgraphs(self, nodes, r_label, n_labels, _id):
        # n_labels 论文中的 节点的 position encoding 数据
        subgraph = self.graph.subgraph(nodes)
        # type 表示关系类型
        subgraph.edata['type'] = self.graph.edata['type'][subgraph.edata[dgl.EID]]
        # r_label 表示 rel_id
        # subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)
        
        # edges_btw_roots = subgraph.edge_ids(0, 1)
        # rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        # # torch.Tensor.nelement() 统计张量中元素的数量
        # if rel_link.squeeze().nelement() == 0:
        #     subgraph.add_edge(0, 1)
        #     subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
        #     subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)

        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        # kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        kge_nodes = nodes
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats, _id)

        return subgraph

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        # n_nodes = subgraph.number_of_nodes()
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        # label_feats[np.arange(n_nodes), n_labels] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None, _id=0):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        # label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        # label_feats[np.arange(n_nodes), 0] = 1
        # label_feats[np.arange(n_nodes), self.max_n_label[0] + 1] = 1
        # n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        if n_feats is not None:
            subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
            # if self.params.finetune_ebd:
            #     subgraph.ndata['feat'].requires_grad = True
            # else:
            #     subgraph.ndata['feat'].requires_grad = False
        else:
            subgraph.ndata['feat'] = torch.rand(n_nodes, self.params.inp_dim)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = _id+1  # head
        n_ids[tail_id] = -1  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        # self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph
    
if __name__ == "__main__":
    pass
    # root = 
    # data = RecData()