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

dir = './embeddings'
dataset = 'movielens'
model = 'sage'

# load graph
mydata = MyDataset(data_name=dataset)
graph = mydata[0]
ratings = graph.edata['weight'].numpy()
u, v, eids = graph.edges(form='all')
adj = np.array(sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy()))).todense())

# load embeddings
print ('==== loading the learned embeddings ====')
path = '{}/{}_{}_embedding.bin'.format(dir, dataset, model)
with open(path, "rb") as in_file:
    embeddings = pkl.load(in_file)
print ('shape: ', embeddings.shape)
# construct y_true, y_pred
scores = np.dot(embeddings, embeddings.T)
# scores = softmax(scores, axis=1)

# load bpr embeddings
print ('==== loading the bpr trained embeddings ====')
path = '{}/{}_{}_bpr_embedding.bin'.format(dir, dataset, model)
with open(path, "rb") as in_file:
    embeddings_bpr = pkl.load(in_file)
print ('shape: ', embeddings_bpr.shape)
# construct y_true, y_pred
scores_bpr = np.dot(embeddings_bpr, embeddings_bpr.T)
# scores_bpr = softmax(scores_bpr, axis=1)

# random embedding
embeddings_rd = np.random.rand(embeddings.shape[0], embeddings.shape[1])
scores_rd = np.dot(embeddings_rd, embeddings_rd.T)
# scores_rd = softmax(scores_rd, axis=1)

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

# retrieve ratings
adj_ratings = np.zeros_like(adj)
(row, col) = np.nonzero(adj)
for i in range(len(row)):
    adj_ratings[row[i]][col[i]] = ratings[eid_dict[(row[i], col[i])]]

auc = []
auc_bpr = []
auc_rd = []
ndcg = []
ndcg_bpr = []
ndcg_rd = []
ndcgrating = []
ndcgrating_bpr = []
ndcgrating_rd = []
# only treat items as valid candidate set
mask = np.array([adj.shape[0] + i for i in range(adj.shape[1]-adj.shape[0])])
i = 0
for src_n, des_ns in tqdm.tqdm(edge_dict.items()):
    y_pred = scores[src_n, mask]
    y_pred_rd = scores_rd[src_n, mask]
    y_pred_bpr = scores_bpr[src_n, mask]

    y_true = adj[src_n, mask]

    # retrieve ratings
    y_rating = adj_ratings[src_n, mask]

    # calc auc
    auc.append( roc_auc_score(y_true, y_pred) )
    auc_rd.append( roc_auc_score(y_true, y_pred_rd) )
    auc_bpr.append( roc_auc_score(y_true, y_pred_bpr) )

    # calc sklearn ndcg
    ndcg.append( ndcg_score(np.expand_dims(y_true, axis=0), np.expand_dims(y_pred, axis=0), k=10) )
    ndcg_rd.append( ndcg_score(np.expand_dims(y_true, axis=0), np.expand_dims(y_pred_rd, axis=0), k=10) )
    ndcg_bpr.append( ndcg_score(np.expand_dims(y_true, axis=0), np.expand_dims(y_pred_bpr, axis=0), k=10) )

    # calc sklearn ndcg based on ratings
    ndcgrating.append( ndcg_score(np.expand_dims(y_rating, axis=0), np.expand_dims(y_pred, axis=0), k=10) )
    ndcgrating_rd.append( ndcg_score(np.expand_dims(y_rating, axis=0), np.expand_dims(y_pred_rd, axis=0), k=10) )
    ndcgrating_bpr.append( ndcg_score(np.expand_dims(y_rating, axis=0), np.expand_dims(y_pred_bpr, axis=0), k=10) )

    i += 1
    if i % 500 == 0:
        print ('  model      |  auc   | ndcg | ndcg on rating ')
        print ('  sage       | {:.4f} | {:.4f} | {:.4f}'.format(np.mean(auc), np.mean(ndcg), np.mean(ndcgrating)))
        print ('  sage_bpr   | {:.4f} | {:.4f} | {:.4f}'.format(np.mean(auc_bpr), np.mean(ndcg_bpr), np.mean(ndcgrating_bpr)))
        print ('  random     | {:.4f} | {:.4f} | {:.4f} '.format(np.mean(auc_rd), np.mean(ndcg_rd), np.mean(ndcgrating_rd)))

print ('=== Final Result ===')
print ('  model      |  auc   | ndcg | ndcg on rating ')
print ('  sage       | {:.4f} | {:.4f} | {:.4f}'.format(np.mean(auc), np.mean(ndcg), np.mean(ndcgrating)))
print ('  sage_bpr   | {:.4f} | {:.4f} | {:.4f}'.format(np.mean(auc_bpr), np.mean(ndcg_bpr), np.mean(ndcgrating_bpr)))
print ('  random     | {:.4f} | {:.4f} | {:.4f} '.format(np.mean(auc_rd), np.mean(ndcg_rd), np.mean(ndcgrating_rd)))
