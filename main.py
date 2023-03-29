import os
import sys
import time
import copy
import itertools
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import embedding.factory as ebd

from tqdm import tqdm
from scipy.sparse import SparseEfficiencyWarning
from torch.utils.data import DataLoader, dataset

from subgraph_extraction.dataset import RecData, generate_subgraph_datasets, load_word_embeddings
# from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl, collate_dgl_train
from utils.metrics import ndcg_at_k, recall_at_k
from models.KADM import KADM

# from managers.evaluator import Evaluator
# from managers.trainer import Trainer

from warnings import simplefilter

BASE_DIR = os.path.dirname( os.path.abspath(__file__) )

sys.path.append(BASE_DIR)

logging.getLogger("Logger")

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s %(name)s %(levelname)s %(message)s',
                   datefmt='%a, %d %b %Y %H:%M:%S')

def grad_param(model, keys):
    '''
        Return a generator that generates learnable parameters in
        model[key] for key in keys.
    '''
    if len(keys) == 1:
        return filter(lambda p: p.requires_grad,
                model[keys[0]].parameters())
    else:
        return filter(lambda p: p.requires_grad,
                itertools.chain.from_iterable(
                    model[key].parameters() for key in keys))


def train_epoch(model, optimizer, params, epoch, train_loader, test_data, max_recall, best_model):
    loss, count = 0, 0
    train = True
    model['ebd'].train()
    model['clf'].train()
    for i, data in enumerate(tqdm(train_loader, desc="train_epoch")):
        uid, iid, label, \
        desc_of_item_hist, desc_of_item_neighbor, desc_of_neg_item_neighbor, \
        batched_graph_pos, batched_graph_neg, _ids = data
        
        pos_scores = model['clf'](desc_of_item_hist.to(params.device), desc_of_item_neighbor.to(params.device), batched_graph_pos.to(params.device), _ids.to(params.device), model['ebd'])
        neg_scores = model['clf'](desc_of_item_hist.to(params.device), desc_of_neg_item_neighbor.to(params.device), batched_graph_neg.to(params.device), _ids.to(params.device), model['ebd'])
        
        optimizer.zero_grad()
        bprloss = -torch.mean(torch.log(torch.sigmoid(pos_scores-neg_scores)))
        loss += bprloss
        count += 1
        bprloss.backward()
        optimizer.step()
        
        # if (count) % params.eval_every_iter == 0:
    mean_ndcg_at_k, mean_recall_at_k = evaluator(model, test_data, params)
    logging.info(f'Epoch {epoch} Step {i+1} train loss={loss}\
        valid result: NDCG@20: {mean_ndcg_at_k} Recall@20: {mean_recall_at_k}\
        max_recall@20: {max_recall}')
    if mean_recall_at_k > max_recall:
        max_recall = mean_recall_at_k
        best_model = copy.deepcopy(model)
                
                
        
    return loss, best_model, max_recall
        
def evaluator(model, data_loader, params, test=False):
    model['ebd'].eval()
    model['clf'].eval()
    all_scores = np.array([])
    uids = []
    iids = []
    labels = []
    for i, data in enumerate(tqdm(data_loader, desc="evaluator")):
        # if i == 10:
        #     break
        uid, iid, label, \
        desc_of_item_hist, desc_of_item_neighbor, \
        batched_graph_pos, _ids = data
        
        scores = model['clf'](
            desc_of_item_hist.to(params.device), 
            desc_of_item_neighbor.to(params.device), 
            batched_graph_pos.to(params.device),
            _ids.to(params.device),
            model['ebd']
            )
        all_scores = np.append(all_scores, scores.detach().cpu().numpy())
        uids += uid
        iids += iid
        labels += label
    # if not test:
    #     model.train()
    result = pd.DataFrame(
        {
            "uid": uids,
            "iid": iids,
            "label": labels,
            "score": all_scores.tolist()
        }
    )
    mean_ndcg_at_k, mean_recall_at_k = eval(result, 20)
    
    return mean_ndcg_at_k, mean_recall_at_k

def eval(df, k):
    uids = list(set(df['uid']))
    all_ndcg_at_k, all_recall_at_k = [], []
    for uid in uids:
        u_df = df[df['uid']==uid]
        u_scores = u_df['score'].to_list()
        u_labels = u_df['label'].to_list()
        sorted_id = sorted(range(len(u_scores)), key=lambda k: u_scores[k], reverse=True)
        sorted_labels = u_df.iloc[sorted_id, 2]
        u_ndcg_at_k, u_recall_at_k = ndcg_at_k(sorted_labels, k), recall_at_k(sorted_labels, k, sum(u_labels))
        all_ndcg_at_k.append(u_ndcg_at_k)
        all_recall_at_k.append(u_recall_at_k)
    return sum(all_ndcg_at_k) / len(uids), sum(all_recall_at_k) / len(uids)
    

def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    splits = ['train', 'test', 'valid']
    i = 0
    params.db_path = os.path.join(params.root, f'data/{splits[i]}/subgraphs_en_{params.enclosing_sub_graph}_hop_{params.hop}_max_nodes_per_hop_{params.max_nodes_per_hop}_max_neighbor_{params.MAX_NEIGH_LEN}')
    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params, [splits[i]])
        return 
        
    vocab = load_word_embeddings(params)
    
    train = RecData(
                 root=params.root, 
                 params=params,
                 db_path=os.path.join(params.root, f'data/{splits[0]}/subgraphs_en_{params.enclosing_sub_graph}_hop_{params.hop}_max_nodes_per_hop_{params.max_nodes_per_hop}_max_neighbor_{params.MAX_NEIGH_LEN}'),
                 train=True, 
                 test=False, 
                 dgl_graph=None,
                 MAX_DESC_LEN=params.MAX_DESC_LEN, 
                 MAX_NEIGH_LEN=params.MAX_NEIGH_LEN, 
                 vocab=vocab,
                 logging=logging)
    train_loader = DataLoader(
            train,
            batch_size=params.batch_size, 
            shuffle=True, 
            num_workers=params.num_workers, 
            collate_fn=params.collate_fn_train
            )

    
    test = RecData(
                 root=params.root, 
                 params=params,
                 db_path=os.path.join(params.root, f'data/{splits[1]}/subgraphs_en_{params.enclosing_sub_graph}_hop_{params.hop}_max_nodes_per_hop_{params.max_nodes_per_hop}_max_neighbor_{params.MAX_NEIGH_LEN}'),
                 train=False, 
                 test=True, 
                 dgl_graph=None,
                 MAX_DESC_LEN=params.MAX_DESC_LEN, 
                 MAX_NEIGH_LEN=params.MAX_NEIGH_LEN, 
                 vocab=vocab,
                 logging=logging)
    test_loader = DataLoader(
            test,
            batch_size=params.batch_size, 
            shuffle=False, 
            num_workers=params.num_workers, 
            collate_fn=params.collate_fn
            )

    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels
    # params.inp_dim = train.n_feat_dim
    params.num_words = vocab.vectors.size()[0]

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    # params.max_label_value = train.max_n_label
    # initialize model
    model = {}
    model["ebd"] = ebd.get_embedding(vocab, params)
    params.word_dim = model["ebd"].ebd_dim
    model["clf"] = KADM(params)
    
    model['ebd'].to(params.device)
    model['clf'].to(params.device)
    optimizer = torch.optim.Adam(grad_param(model, ['ebd', 'clf']), lr=params.lr)
    # print(optimizer.param_groups[0]['lr'])
    max_recall = float("-inf")
    best_model = None
    # graph_classifier = initialize_model(params, dgl_model, params.load_model)
    for epoch in range(1, params.num_epochs + 1):
        if epoch > 1:
            optimizer.param_groups[0]['lr'] = params.lr * params.learning_rate_decay_factor
        time_start = time.time()
        loss, best_model, max_recall = train_epoch(
            model, 
            optimizer,
            params,
            epoch,
            train_loader,  
            test_loader, 
            max_recall, 
            best_model
            )
        time_elapsed = time.time() - time_start
        logging.info(f'Epoch {epoch} with loss: {loss} in {time_elapsed}')
    path = os.path.join(f'checkpoints/KADM_dataset_{params.dataset}_lr_{params.lr}_epochs_{params.num_epochs}_bt_{params.batch_size}_word_dim_{params.word_dim}_ent_dim_{params.inp_dim}_hop_{params.hop}_max_nodes_per_hop_{params.max_nodes_per_hop}_max_neighbor_{params.MAX_NEIGH_LEN}')
    best_model['clf'].save(path)
    
    test_ndcg_at_k, test_recall_at_k = evaluator(best_model, test_loader, params, test=True)
    logging.info(f'Evaluation result of best model on valid data: \
        NDCG@20: {test_ndcg_at_k} Recall@20: {test_recall_at_k}')
    
    logging.info(f"Device: {params.device}")
    logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    
def set_random_seed(np_seed, torch_seed):
    np.random.seed(np_seed)                  
    torch.manual_seed(torch_seed)       
    torch.cuda.manual_seed(torch_seed)      
    torch.cuda.manual_seed_all(torch_seed)  

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str, default="ml-1m-new",
                        help="Dataset string, {ml-1m, lastFM}")
    parser.add_argument("--gpu", type=int, default=1,
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true',
                        help='Load existing model?')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")

    # Training regime params
    parser.add_argument("--num_epochs", "-ne", type=int, default=50,
                        help="Learning rate of the optimizer")
    parser.add_argument("--eval_every", type=int, default=3,
                        help="Interval of epochs to evaluate the model?")
    parser.add_argument("--eval_every_iter", type=int, default=100,
                        help="Interval of iterations to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=100,
                        help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use?")
    parser.add_argument("--momentum", type=float, default=0.5,
                        help="param for SGD")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate of the optimizer")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=1,
							help="Learning rate decays by this much.")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default=5e-4,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")
    
    

    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2,
                        help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=100,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=True,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument("--kge_dir", type=str, default="/data/wsc-data/NEW/KADM-NEW/subgraph_extraction/kge_embedding",
                        help="the dir to save the kge embeddings")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument("--MAX_DESC_LEN", type=int, default=100,
                        help="The maximum length of item description")
    parser.add_argument("--MAX_NEIGH_LEN", type=int, default=30,
                        help="The maximum size of user history set or item neighbor set")

    # Model params
    parser.add_argument("--beta", "-bt", type=float, default=0.7,
                        help="The weight of local model in KADM")
    parser.add_argument("--inp_dim", "-id", type=int, default=50,
                        help="entity embedding size")
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=50,
                        help="Relation embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=50,
                        help="Relation embedding size for attention")
    parser.add_argument("--emb_dim", "-dim", type=int, default=50,
                        help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=2,
                        help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate in GNN layers")
    parser.add_argument("--edge_dropout", type=float, default=0.5,
                        help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='mlp',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True,
                        help='whether to have attn in model or not')
    parser.add_argument("--word_dim", "-wd", type=int, default=300,
                        help="The size of word embedding for local model")
    parser.add_argument("--embedding", type=str, default="cnn",
                        help=("document embedding method. Options: "
                              "[avg, tfidf, meta, oracle, cnn]"))
    
    # base word embedding
    parser.add_argument("--wv_path", type=str,
                        default="/data/wsc-data/NEW/RecSysDatasets/word2vector/",
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default="wiki.en.vec",
                        help=("Name of pretrained word embeddings."))
    parser.add_argument("--finetune_ebd", action="store_true", default=True,
                        help=("Finetune embedding during meta-training"))
    
    parser.add_argument("--bert", default=False, action="store_true",
                        help=("set true if use bert embeddings "
                              "(only available for sent-level datasets: "
                              "huffpost, fewrel"))
    # cnn configuration
    parser.add_argument("--cnn_num_filters", type=int, default=50,
                        help="Num of filters per filter size [default: 50]")
    parser.add_argument("--cnn_filter_sizes", type=int, nargs="+",
                        default=[3],
                        help="Filter sizes [default: 3]")

    params = parser.parse_args()
    # initialize_experiment(params, __file__)
    params.main_dir = "/data/wsc-data/NEW/RecSysDatasets/"
    params.root = os.path.join(params.main_dir, params.dataset)

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.collate_fn_train = collate_dgl_train
    params.move_batch_to_device = move_batch_to_device_dgl
    set_random_seed(304, 2021)
    main(params)
