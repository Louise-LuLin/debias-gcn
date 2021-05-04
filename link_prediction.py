"""
Link Prediction Task: predicting the existence of an edge between two arbitrary nodes in a graph.
===========================================
-  Model: DGL-based graphsage and gat encoder (and many more)
-  Loss: cross entropy. You can modify the loss as you want
-  Metric: AUC
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
import os

from sklearn.metrics import roc_auc_score
import dgl.data

from models import GraphSAGE, GAT
from create_dataset import MyDataset
from predictors import DotLinkPredictor, MLPLinkPredictor
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, 
                    help='Random seed.')
parser.add_argument('--device', type=int, default=0, 
                    help='cuda')

parser.add_argument('--model', type=str, default='sage', 
                    choices=['sage', 'gat'], help='model variant')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dim1', type=int, default=64,
                    help='Number of first layer hidden units.')
parser.add_argument('--dim2', type=int, default=16,
                    help='Number of second layer hidden units.')

parser.add_argument('--predictor', type=str, default='dot',
                    choices=['dot', 'mlp'], help='Predictor of the output layer')

parser.add_argument('--dataset', type=str, default='cora', 
                    choices=['cora', 'citeseer', 'karate'], help='dataset')
parser.add_argument('--out_dir', type=str, default='./embeddings', 
                    help='output embedding foler')

args = parser.parse_args()

######################################################################
# Set up device and fix random seed
print ('==== Environment ====')
device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1) # limit cpu use
print ('  pytorch version: ', torch.__version__)
print ('  device: ', device)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

######################################################################
# Load and construct data for link prediction task
print ('==== Dataset ====')
graph, features, train_pos_g, train_neg_g, test_pos_g, test_neg_g = construct_link_prediction_data(data_type=args.dataset)
n_features = features.shape[1]

# Initialize embedding model
if args.model == 'sage':
    model = GraphSAGE(graph,
                      in_dim=n_features, 
                      hidden_dim=args.dim1, 
                      out_dim=args.dim2)
else:
    model = GAT(graph,
                in_dim=n_features,
                hidden_dim=args.dim1//8,
                out_dim=args.dim2,
                num_heads=8)

# Initialize link predictor
if args.predictor == 'dot':
    pred = DotLinkPredictor()
else:
    pred = MLPLinkPredictor(args.dim2) 

optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=args.lr)

######################################################################
print ('==== Training ====')
# Training loop
dur = []
cur = time.time()
for e in range(args.epochs):
    model.train()
    # forward propagation on training set
    h = model(features)
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_entropy_loss(pos_score, neg_score)
    # evaluation on test set
    model.eval()
    with torch.no_grad():
        test_pos_score = pred(test_pos_g, h)
        test_neg_score = pred(test_neg_g, h)
        test_auc = compute_auc(test_pos_score, test_neg_score)
        train_auc = compute_auc(pos_score, neg_score)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dur.append(time.time() - cur)
    cur = time.time()

    if e % 5 == 0:
        print("Epoch {:05d} | Loss {:.4f} | Train AUC {:.4f} | Test AUC {:.4f} | Time {:.4f}".format(
              e, loss.item(), train_auc, test_auc, dur[-1]))

######################################################################
# Save embedding
embeddings = h.detach().numpy()
print ('==== saving the learned embeddings ====')
print ('Shape: {} | Type: {}'.format(embeddings.shape, type(embeddings)))
path = '{}/{}_{}_embedding.bin'.format(args.out_dir, args.dataset, args.model)
os.makedirs(args.out_dir, exist_ok=True)
with open(path, "wb") as output_file:
    pkl.dump(embeddings, output_file)
print ('==== saved to {} ===='.format(path))
