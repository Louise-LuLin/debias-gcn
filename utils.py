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

from sklearn.metrics import roc_auc_score

import dgl.data
from create_dataset import MyDataset

def compute_entropy_loss(pos_score, neg_score):
    """Compute cross entropy loss
    """
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    """Compute AUC metric
    """
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def construct_link_data(data_type='cora'):
    """Construct dataset for link prediction
    
    Parameters
    ----------
    root :
        root directory where the variable should be saved
    name : str
        saved file name

    Returns
    -------
    None.
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
    adj_neg = 1 - adj.todense() - np.eye(graph.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), graph.number_of_edges() // 2)
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

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

    return train_graph, features, train_pos_g, train_neg_g, test_pos_g, test_neg_g
