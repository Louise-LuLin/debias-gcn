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
import os
import tqdm
from scipy.special import expit, softmax

import dgl.data
from sklearn.metrics import roc_auc_score, ndcg_score
from create_dataset import MyDataset

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size != k:
        raise ValueError('Ranking List length < k')    
    return np.sum((2**r - 1) / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(r, k):
    sort_r = sorted(r,reverse = True)
    idcg = dcg_at_k(sort_r, k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

dir = './embeddings'
dataset = 'movielens'
model = 'sage'


# load embeddings
print ('==== loading the learned embeddings ====')
path = '{}/{}_{}_embedding.bin'.format(dir, dataset, model)
with open(path, "rb") as in_file:
    embeddings = pkl.load(in_file)
print ('shape: ', embeddings.shape)

dataset = MyDataset(data_name=dataset)
graph = dataset[0]
ratings = graph.edata['weight']

u, v, eids = graph.edges(form='all')

# construct y_true, y_pred
scores = np.dot(embeddings, embeddings.T)
scores = softmax(scores, axis=1)

#random
rd_embeddings = np.random.rand(embeddings.shape[0], embeddings.shape[1])
rd_scores = np.dot(rd_embeddings, rd_embeddings.T)
rd_scores = softmax(rd_scores, axis=1)

adj = np.array(sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy()))).todense())
print (scores.shape, adj.shape, type(scores), type(adj))

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

auc = []
auc_rd = []
ndcg = []
ndcg_rd = []
myndcg = []
myndcg_rd = []
# only treat items as valid candidate set
mask = np.array([adj.shape[0] + i for i in range(adj.shape[1]-adj.shape[0])])
i = 0
for src_n, des_ns in tqdm.tqdm(edge_dict.items()):
    y_pred = scores[src_n, mask]
    y_pred2 = rd_scores[src_n, mask]
    y_true = adj[src_n, mask]

    # calc auc
    auc.append( roc_auc_score(y_true, y_pred) )
    auc_rd.append( roc_auc_score(y_true, y_pred2) )

    # calc Nan's ndcg
    myndcg.append( ndcg_at_k(y_pred, 10) )
    myndcg_rd.append( ndcg_at_k(y_pred2, 10) )

    # calc sklearn ndcg
    ndcg.append( ndcg_score(np.expand_dims(y_true, axis=0), np.expand_dims(y_pred, axis=0), k=10) )
    ndcg_rd.append( ndcg_score(np.expand_dims(y_true, axis=0), np.expand_dims(y_pred2, axis=0), k=10) )

    i += 1
    if i % 50 == 0:
        print ('  model  | auc  | myndcg | sklean ndcg ')
        print ('  sage   | {:.4f} | {:.4f} | {:.4f}'.format(np.mean(auc), np.mean(myndcg), np.mean(ndcg)))
        print ('  random | {:.4f} | {:.4f} | {:.4f} '.format(np.mean(auc_rd), np.mean(myndcg_rd), np.mean(ndcg_rd)))

print ('  model  | auc  | myndcg | sklean ndcg ')
print ('  sage   | {:.4f} | {:.4f} | {:.4f}'.format(np.mean(auc), np.mean(myndcg), np.mean(ndcg)))
print ('  random | {:.4f} | {:.4f} | {:.4f} '.format(np.mean(auc_rd), np.mean(myndcg_rd), np.mean(ndcg_rd)))
