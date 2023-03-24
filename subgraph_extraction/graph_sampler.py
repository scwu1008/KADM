import os
import math
import struct
import logging
import random
import pickle as pkl
import pdb
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import sys
import torch
from scipy.special import softmax
from utils.dgl_utils import _bfs_relational
from utils.graph_utils import incidence_matrix, remove_nodes, ssp_to_torch, serialize, deserialize, get_edge_count, diameter, radius
import networkx as nx


# def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
#     pos_edges = edges
#     neg_edges = []

#     # if max_size is set, randomly sample train links
#     if max_size < len(pos_edges):
#         perm = np.random.permutation(len(pos_edges))[:max_size]
#         pos_edges = pos_edges[perm]

#     # sample negative links for train/test
#     n, r = adj_list[0].shape[0], len(adj_list)

#     # distribution of edges across reelations
#     theta = 0.001
#     edge_count = get_edge_count(adj_list)
#     rel_dist = np.zeros(edge_count.shape)
#     idx = np.nonzero(edge_count)
#     rel_dist[idx] = softmax(theta * edge_count[idx])

#     # possible head and tails for each relation
#     valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
#     valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

#     pbar = tqdm(total=len(pos_edges))
#     while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
#         neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
#         if np.random.uniform() < constrained_neg_prob:
#             if np.random.uniform() < 0.5:
#                 neg_head = np.random.choice(valid_heads[rel])
#             else:
#                 neg_tail = np.random.choice(valid_tails[rel])
#         else:
#             if np.random.uniform() < 0.5:
#                 neg_head = np.random.choice(n)
#             else:
#                 neg_tail = np.random.choice(n)

#         if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
#             neg_edges.append([neg_head, neg_tail, rel])
#             pbar.update(1)

#     pbar.close()

#     neg_edges = np.array(neg_edges)
#     return pos_edges, neg_edges


def links2subgraphs(A, graphs, params, split_name):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    # max_n_label = {'value': np.array([0, 0])}
    # subgraph_sizes = []
    # enc_ratios = []
    num_pruned_nodes = []

    BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['triplets'], A, params, split_name) * 1.5
    links_length = 0
    for split_name, split in graphs.items():
        links_length += len(split['triplets']) * 2
    map_size = links_length * BYTES_PER_DATUM
    
    if not os.path.exists(params.db_path):
        os.makedirs(params.db_path)
    env = lmdb.open(params.db_path, map_size=map_size, max_dbs=2)

    def extraction_helper(A, links, split_name, split_env):

        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, params, split_name)) as p:
            args_ = zip(range(len(links)), links)
            for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
                # num_pruned_nodes.append(datum['num_pruned_nodes'])

                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for links in {split_name} set")
        # labels = np.ones(len(split['pos']))
        db_name_pos = split_name
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split['triplets'], split_name, split_env)
        

def get_average_subgraph_size(sample_size, links, A, params, split_name):
    total_size = 0
    temp = links[np.random.choice(len(links), sample_size)]
    if split_name == "train":
        for (u1, n2, r_label, _, _) in tqdm(temp, desc="Get average subgraph size"):
            n2 = int(params.e2id[params.i2e[str(n2)]])
            history_iids_of_user = params.u2his[u1]
            n1_set = list(map(lambda iid: int(params.e2id[params.i2e[str(iid)]]), history_iids_of_user))
            nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1_set, n2), A, params.hop, params.max_nodes_per_hop)
            datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
            total_size += len(serialize(datum))
    else:
        for (u1, n2, r_label) in tqdm(temp, desc="Get average subgraph size"):
            n2 = int(params.e2id[params.i2e[str(n2)]])
            history_iids_of_user = params.u2his[u1]
            n1_set = list(map(lambda iid: int(params.e2id[params.i2e[str(iid)]]), history_iids_of_user))
            nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1_set, n2), A, params.hop, params.max_nodes_per_hop)
            datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
            total_size += len(serialize(datum))
    return total_size / sample_size


def intialize_worker(A, params, split_name):
    global A_, params_, split_name_
    A_, params_, split_name_ = A, params, split_name


def extract_save_subgraph(args_):
    idx, link = args_
    if split_name_ == "train":
        (u1, n2, r_label, _, n3) = link
        n2 = int(params_.e2id[params_.i2e[str(n2)]])
        n3 = int(params_.e2id[params_.i2e[str(n3)]])
        history_iids_of_user = params_.u2his[u1]
        n1_set = list(map(lambda iid: int(params_.e2id[params_.i2e[str(iid)]]), history_iids_of_user))
        pos_nodes, pos_n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1_set, n2), A_, params_.hop, params_.max_nodes_per_hop)
        neg_nodes, neg_n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1_set, n3), A_, params_.hop, params_.max_nodes_per_hop)
        datum = {'pos_nodes': pos_nodes, 'neg_nodes': neg_nodes, 'pos_n_labels': pos_n_labels, 'neg_n_labels': neg_n_labels}
    else:
        (u1, n2, r_label) = link
        n2 = int(params_.e2id[params_.i2e[str(n2)]])
        history_iids_of_user = params_.u2his[u1]
        n1_set = list(map(lambda iid: int(params_.e2id[params_.i2e[str(iid)]]), history_iids_of_user))
        pos_nodes, pos_n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1_set, n2), A_, params_.hop, params_.max_nodes_per_hop)
        datum = {'pos_nodes': pos_nodes, 'pos_n_labels': pos_n_labels}
    str_id = '{:08}'.format(idx).encode('ascii')
    return (str_id, datum)

# 修改前
# def extract_save_subgraph(args_, A, params):
#     (n1_set, n2, r_label) = args_
#     nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1_set, n2), A, params.hop, True, params.max_nodes_per_hop)
#     return nodes, r_label, n_labels
    # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
    # if max_label_value_ is not None:
    #     n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

    # datum = {'nodes': nodes, 'g_label': g_label, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
    # str_id = '{:08}'.format(idx).encode('ascii')

    # return (str_id, datum)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, A_list, h=1, max_nodes_per_hop=None, ):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T
    # 读取 history set 相关的所有 neighbor 信息 
    root1_nei = get_neighbor_nodes(set(ind[0]), A_incidence, h, max_nodes_per_hop)
    # 读取 target entity 的 neighbor 信息
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    # if enclosing_sub_graph:
    # 取交集
    subgraph_nodes = ind[0] + list([ind[1]]) + list(subgraph_nei_nodes_int)
    # else:
        # subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]
    # incidence_matrix() 是合并多个 relation adj
    labels, enclosing_subgraph_nodes = node_label(incidence_matrix(subgraph), max_distance=h, length=len(ind[0]))

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]
    # pruned_subgraph_nodes = subgraph_nodes
    # pruned_labels = labels

    # if max_node_label_value is not None:
    #     pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
    num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)

    return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes


def node_label(subgraph, max_distance=1, length=1):
    # implementation of the node labeling scheme described in the paper
    roots = [list(range(length)), [length]]
    sgs_single_root = [remove_nodes(subgraph, root) for root in roots]
    # numpy.clip(a, a_min, a_max, out=None)[source]
    # 将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
    target_dists = np.clip(ssp.csgraph.dijkstra(sgs_single_root[0], indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7)
    history_dists = np.clip(ssp.csgraph.dijkstra(sgs_single_root[1], indices=roots[0], directed=False, unweighted=True, limit=1e6)[:, len(roots[0]):], 0, 1e7)
    history_dists = np.min(history_dists, axis=0)
    dist_to_roots = [target_dists[0], history_dists]
    # dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0], dist_to_roots[1])), dtype=int)

    target_node_labels = np.array([[0, 1] for _ in range(len(roots[0]))]+[[1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes