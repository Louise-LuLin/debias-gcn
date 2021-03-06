"""
Util functions
"""
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import time
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import argparse
import tqdm

from sklearn.metrics import roc_auc_score, ndcg_score

import dgl.data
from create_dataset import MyDataset

def compute_entropy_loss(pos_score, neg_score, index_dict, device='cpu', byNode=False):
    """Compute cross entropy loss for link prediction
    """

    scores = torch.cat([pos_score, neg_score]).to(device)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_metric(pos_score, neg_score, index_dict, device='cpu', byNode=False):
    """Compute AUC, NDCG metric for link prediction
    """
            
    if byNode == False:
        y_pred = torch.sigmoid(torch.cat([pos_score, neg_score])) # the probability of positive label
        
        y_true = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        
        auc = roc_auc_score(y_true.cpu(), y_pred.cpu())
        ndcg = ndcg_score( np.expand_dims(y_true.cpu().numpy(), axis=0), np.expand_dims(y_pred.cpu().numpy(), axis=0)) # super slow! you can comment it 
    else:
        auc = []
        ndcg = []
        for src_n, idxs in tqdm.tqdm(index_dict.items()):
            if len(idxs) == 0: # this may happen when not sample by node
                continue
            
            y_pred = torch.sigmoid(torch.cat([pos_score[idxs], neg_score[idxs]]))

            y_true = torch.cat([torch.ones(pos_score[idxs].shape[0]), torch.zeros(neg_score[idxs].shape[0])]) 
            
            auc.append(roc_auc_score(y_true.cpu(), y_pred.cpu()))
            ndcg.append(ndcg_score( np.expand_dims(y_true.cpu().numpy(), axis=0), np.expand_dims(y_pred.cpu().numpy(), axis=0)))

        auc = np.mean(np.array(auc))
        ndcg = np.mean(np.array(ndcg))

    return auc, ndcg

def evaluate_acc(logits, labels, mask):
    """Compute Accuracy for node classification
    """

    logits = logits[mask]
    labels = labels[mask]
    _, indices = torch.max(logits, dim=1)
    correct =torch.sum(indices==labels)
    return correct.item() * 1.0 / len(labels) # Accuracy

def load_node_classification_data(data_type='cora', device='cpu'):
    """Construct dataset for node classification
    
    Parameters
    ----------
    data_type :
        name of dataset

    Returns
    -------
    graph : 
        graph data; dgl.graph
    features :
        node feature; torch tensor
    labels :
        node label; torch tensor
    train/valid/test_mask :
        node mask for different set; torch tensor

    """

    if data_type == 'cora':
        dataset = dgl.data.CoraGraphDataset()
    elif data_type == 'citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
    else:
        dataset = MyDataset(data_name=data_type)
    
    graph = dataset[0].to(devide)
    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device)
    train_mask = graph.ndata['train_mask'].to(device)
    valid_mask = graph.ndata['val_mask'].to(device)
    test_mask  = graph.ndata['test_mask'].to(device)
    return graph, features, labels, train_mask, valid_mask, test_mask

def construct_link_prediction_data(data_type='cora', device='cpu'):
    """Construct dataset for link prediction
    
    Parameters
    ----------
    data_type :
        name of dataset

    Returns
    -------
    train_graph :
        graph reconstructed by all training nodes; dgl.graph
    features :
        node feature; torch tensor
    train_pos_g :
        graph reconstructed by positive training edges; dgl.graph
    train_neg_g :
        graph reconstructed by negative training edges; dgl.graph
    test_pos_g :
        graph reconstructed by positive testing edges; dgl.graph
    test_neg_g :
        graph reconstructed by negative testing edges; dgl.graph
    """

    if data_type == 'cora':
        dataset = dgl.data.CoraGraphDataset()
    elif data_type == 'citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
    else:
        dataset = MyDataset(data_name=data_type)
    
    graph = dataset[0]
    features = graph.ndata['feat']

    # Split the edge set for training and testing sets:
    # -  Randomly picks 10% of the edges in test set as positive examples
    # -  Leave the rest for the training set
    # -  Sample the same number of edges for negative examples in both sets
    u, v = graph.edges()

    eids = np.arange(graph.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = graph.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    if data_type != 'movielens':
        mask = np.eye(adj.shape[0])
    else: # for bipartite graph, mask user-user pairs
        mask = np.zeros_like(adj.todense())
        mask[:adj.shape[0], :adj.shape[0]] = 1 # requirement: src node starts at 0
    adj_neg = 1 - adj.todense() - mask
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), graph.number_of_edges() // 2)
    # neg_eids = np.random.permutation(np.arange(len(neg_u))) # np.random.choice(len(neg_u), graph.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    # tranform to tensor
    # test_pos_u, test_pos_v = torch.tensor(test_pos_u), torch.tensor(test_pos_v)
    test_neg_u, test_neg_v = torch.tensor(test_neg_u), torch.tensor(test_neg_v)
    # train_pos_u, train_pos_v = torch.tensor(train_pos_u), torch.tensor(train_pos_v)
    train_neg_u, train_neg_v = torch.tensor(train_neg_u), torch.tensor(train_neg_v)

    print ('==== Link Prediction Data ====')
    print ('  TrainPosEdges: ', len(train_pos_u))
    print ('  TrainNegEdges: ', len(train_neg_u))
    print ('  TestPosEdges: ', len(test_pos_u))
    print ('  TestNegEdges: ', len(test_neg_u))

    # Remove the edges in the test set from the original graph:
    # -  A subgraph will be created from the original graph by ``dgl.remove_edges``
    train_graph = dgl.remove_edges(graph, eids[:test_size])

    # Construct positive graph and negative graph
    # -  Positive graph consists of all the positive examples as edges
    # -  Negative graph consists of all the negative examples
    # -  Both contain the same set of nodes as the original graph
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=graph.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=graph.number_of_nodes())

    return train_graph.to(device), features.to(device), \
           train_pos_g.to(device), train_neg_g.to(device), \
           test_pos_g.to(device), test_neg_g.to(device)

def construct_link_prediction_data_nodewise(data_type='cora', device='cpu'):
    """Construct dataset for link prediction
    
    Parameters
    ----------
    data_type :
        name of dataset

    Returns
    -------
    train_graph :
        graph reconstructed by all training nodes; dgl.graph
    features :
        node feature; torch tensor
    train_pos_g :
        graph reconstructed by positive training edges; dgl.graph
    train_neg_g :
        graph reconstructed by negative training edges; dgl.graph
    test_pos_g :
        graph reconstructed by positive testing edges; dgl.graph
    test_neg_g :
        graph reconstructed by negative testing edges; dgl.graph
    """

    if data_type == 'cora':
        dataset = dgl.data.CoraGraphDataset()
    elif data_type == 'citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
    else:
        dataset = MyDataset(data_name=data_type)
    
    graph = dataset[0]
    features = graph.ndata['feat']

    # Split the edge set for training and testing sets:
    # -  Randomly picks 10% of the edges in test set as positive examples
    # -  Leave the rest for the training set
    # -  Sample the same number of edges for negative examples in both sets
    u, v, eids = graph.edges(form='all')

    # edges grouped by node
    src_nodes = set(u.numpy().tolist()) # all source node idx
    des_nodes = set(v.numpy().tolist()) # all destination node idx
    edge_dict = {}
    eid_dict = {}
    for i in range(len(u.numpy().tolist())):
        if u.numpy()[i] not in edge_dict:
            edge_dict[u.numpy()[i]] = []
        edge_dict[u.numpy()[i]].append(v.numpy()[i])
        eid_dict[(u.numpy()[i], v.numpy()[i])] = eids.numpy()[i]
    
    # sample edges by node
    neg_rate = 30
    test_rate = 0.1
    test_pos_u, test_pos_v = [], []
    test_neg_u, test_neg_v = [], []
    train_pos_u, train_pos_v = [], []
    train_neg_u, train_neg_v = [], []
    test_eids = []
    train_index_dict = {}
    test_index_dict = {}
    for src_n, des_ns in tqdm.tqdm(edge_dict.items()):
        
        pos_des_ns = np.random.permutation(des_ns)
        candidate_negs = list(des_nodes - set(pos_des_ns))

        # split test/train while sampling neg
        test_pos_size = int(len(pos_des_ns) * test_rate)
        for n in range(len(pos_des_ns)):
            # for each pos edge, sample neg_rate neg edges
            neg_des_ns = np.random.choice(candidate_negs, neg_rate)

            if n < test_pos_size: # testing set
                test_neg_v += list(neg_des_ns)
                test_neg_u += [src_n for i in range(len(neg_des_ns))]
                test_pos_v += [pos_des_ns[n] for i in range(len(neg_des_ns))]
                test_pos_u += [src_n for i in range(len(neg_des_ns))]
                test_eids.append(eid_dict[(src_n, pos_des_ns[n])])
                # store index grouped by node
                test_index_dict[src_n] = [len(test_neg_v)-1-i for i in range(len(neg_des_ns))]
            else: # training set
                train_neg_v += list(neg_des_ns)
                train_neg_u += [src_n for i in range(len(neg_des_ns))]
                train_pos_v += [pos_des_ns[n] for i in range(len(neg_des_ns))]
                train_pos_u += [src_n for i in range(len(neg_des_ns))]
                # store index grouped by node
                train_index_dict[src_n] = [len(train_neg_v)-1-i for i in range(len(neg_des_ns))]

    # tranform to tensor
    test_pos_u, test_pos_v = torch.tensor(test_pos_u), torch.tensor(test_pos_v)
    test_neg_u, test_neg_v = torch.tensor(test_neg_u), torch.tensor(test_neg_v)
    train_pos_u, train_pos_v = torch.tensor(train_pos_u), torch.tensor(train_pos_v)
    train_neg_u, train_neg_v = torch.tensor(train_neg_u), torch.tensor(train_neg_v)
    test_eids = torch.tensor(test_eids)

    print ('==== Link Prediction Data ====')
    print ('  TrainPosEdges: ', len(train_pos_u))
    print ('  TrainNegEdges: ', len(train_neg_u))
    print ('  TestPosEdges: ', len(test_pos_u))
    print ('  TestNegEdges: ', len(test_neg_u))

    # Remove the edges in the test set from the original graph:
    # -  A subgraph will be created from the original graph by ``dgl.remove_edges``
    train_graph = dgl.remove_edges(graph, test_eids)

    # Construct positive graph and negative graph
    # -  Positive graph consists of all the positive examples as edges
    # -  Negative graph consists of all the negative examples
    # -  Both contain the same set of nodes as the original graph
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=graph.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=graph.number_of_nodes())

    return train_graph.to(device), features.to(device), \
           train_pos_g.to(device), train_neg_g.to(device), \
           test_pos_g.to(device), test_neg_g.to(device), \
           train_index_dict, test_index_dict
