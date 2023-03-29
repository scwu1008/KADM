import os
import pdb
import datetime
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csc_matrix, data
import matplotlib.pyplot as plt
random.seed(2021)

def tprint(s):
    '''
        print datetime and s
        @params:
            s (str): the string to be printed
    '''
    print('{}: {}'.format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), s),
          flush=True)

def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


def read_KG(data_dir):
    """
    读取 KG data、entity/relaion mapping information以及每个关系下的邻接矩阵信息

    Args:
        data_dir (string): KG data文件所在目录

    Returns:
        adj_list (list<csc_matrix>)): [adj_of_relation_1, adj_of_relation_2, ...] 存储所有关系下的邻接矩阵信息
        triplets (ndarray): array([[head, tail, relation], ...]) 存储KG中三元组信息
        entity2id (dict): {entId: mapping_id, ...} entity2id mapping信息
        relation2id (dict): {relId: mapping_id, ...} relation2id mapping信息
        id2entity (dict): {mapping_id: entId, ...} id2entity mapping信息
        id2relation (dict): {mapping_id: relId, ...} id2relation mapping信息
    """
    
    entity2id = json.load(open(os.path.join(data_dir, "entity2id.json")))
    relation2id = json.load(open(os.path.join(data_dir, "relation2id.json")))

    # triplets = []
    triplets = pd.read_csv(os.path.join(data_dir, "kg.csv")).to_numpy()
    # for line in tqdm(lines):
    #     triplet = line.split()
    #     triplets.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])
    # triplets = np.array(triplets)
            
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets[:, 1] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), 
                                    (triplets[:, 0][idx].squeeze(1), 
                                     triplets[:, 2][idx].squeeze(1)
                                     )
                                    ), 
                                   shape=(len(entity2id), len(entity2id))
                                   )
                        )

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')
            
def read_data(path):
    """
    读取训练/验证/测试数据

    Args:
        path (string): 数据所在路径

    Returns:
        data (list): inter_data, 三元组列表
    """
    # data = list([])
    # with open(path, "r") as f:
    #     lines = f.readlines()
    #     for line in tqdm(lines, desc="read data"):
    #         data.append(list(map(lambda x: int(x), line.split())))
    return pd.read_csv(path).to_numpy().astype(np.int64)


def read_neighbor(path, max_neigh_length):
    """
    读取user的历史交互item set/item的Collaborative neighbors set，对数量小于 max_neigh_length的做 zero-padding

    Args:
        path (string): history file path
        max_neigh_length (int): history set的最大长度

    Returns:
        user2history (dict): {uid/iid: [iid_1, iid_2, ...]} 用户历史交互item set字典
    """
    _2neighbors = json.load(open(path))
    neighrbors = {}
    for _id, iids in _2neighbors.items():
        if len(iids) >= max_neigh_length:
            new_iids = random.sample(iids, max_neigh_length)
        else:
            # 对于长度不够的用0做padding
            new_iids = iids
        neighrbors[int(_id)] = list(map(lambda x: int(x), new_iids))
    return neighrbors

def read_description(path, max_desc_length, vocab):
    """
    读取item的description信息，对description长度小于 max_desc_length zero-padding

    Args:
        path (string): textual description file path
        max_desc_length (int): textual description的最大长度

    Returns:
        item2desc (dict): {iid: [word_1, word_2, ...]} item的描述信息字典
    """
    item2desc = json.load(open(path))
    new_item2desc = {}
    for iid, desc in item2desc.items():
        desc = desc.split()
        if len(desc) >= max_desc_length:
            new_desc = list(map(lambda x: int(vocab.stoi[x]) if x in vocab.stoi else int(vocab.stoi['<unk>']), desc[:max_desc_length]))
        else:
            # 对于长度不够的用0做padding
            new_desc = list(map(lambda x: int(vocab.stoi[x]) if x in vocab.stoi else int(vocab.stoi['<unk>']), desc)) + \
                [int(vocab.stoi['<unk>']) for _ in range(max_desc_length-len(desc))]
        new_item2desc[int(iid)] = new_desc
    return new_item2desc