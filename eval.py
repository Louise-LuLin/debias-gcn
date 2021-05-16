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

adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy()))).todense()

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
ndcg = []
# only treat items as valid candidate set
mask = np.array([adj.shape[0] + i for i in range(adj.shape[1]-adj.shape[0])])
i = 0
for src_n, des_ns in tqdm.tqdm(edge_dict.items()):
    y_pred = scores[src_n, mask]
    y_pred_flip = 1.0 - y_pred

    y_true = adj[src_n, mask]
    y_true_flip = 1 - y_true

    y_pred = np.stack([y_pred, y_pred_flip]).T
    y_true = np.stack([y_true, y_true_flip]).T

    auc.append(roc_auc_score(y_true, y_pred))
    ndcg.append(ndcg_score(y_true, y_pred, k=10))

    i+=1
    if i%50 == 0:
        print (i, np.mean(np.array(auc)), np.mean(np.array(ndcg)))

auc = np.mean(np.array(auc))
ndcg = np.mean(np.array(ndcg))

print (auc, ndcg)
