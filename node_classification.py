"""
Node Classification Task: predicting labels of arbitrary nodes in a graph.
===========================================
-  Model: DGL-based graphsage and gat encoder (and many more)
-  Loss: cross entropy. You can modify the loss as you want
-  Metric: Accuracy
"""

from dgl import DGLGraph
import dgl.data 
import numpy as np
import time
import torch
import torch.nn.functional as F

import argparse
import os
import pickle as pkl

from models import GAT, GraphSAGE
from utils import * 


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, 
                    help='Random seed.')
parser.add_argument('--device', type=int, default=0, 
                    help='cuda')

parser.add_argument('--model', type=str, default='sage', 
                    choices=['sage', 'gat'], help='model variant')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dim1', type=int, default=64,
                    help='Number of first layer hidden units.')
parser.add_argument('--dim2', type=int, default=16,
                    help='Number of second layer hidden units.')

parser.add_argument('--dataset', type=str, default='cora', 
                    choices=['cora', 'citeseer', 'karate'], help='dataset')
parser.add_argument('--out_dir', type=str, default='./embeddings', 
                    help='output embedding foler')

args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1) # limit cpu use
print ('==== Environment ====')
print ('-- pytorch version: ', torch.__version__)
print ('-- device: ', device)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)


def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct =torch.sum(indices==labels)
        return correct.item() * 1.0 / len(labels) # Accuracy

graph, features, labels, train_mask, valid_mask, test_mask = load_node_classification_data(args.dataset)
n_features = features.shape[1]
n_labels = int(labels.max().item() + 1)

if args.model == 'sage':
    model = GraphSAGE(graph,
                      in_dim=n_features, 
                      hidden_dim=args.dim1, 
                      out_dim=n_labels)
else:
    model = GAT(graph,
                in_dim=n_features,
                hidden_dim=args.dim1//8,
                out_dim=n_labels,
                num_heads=8)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

dur = []
cur = time.time()
for e in range(args.epochs):
    model.train()
    # forward propagation using all nodes
    logits = model(features)
    # compute loss
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    # compute validation accuracy
    acc = evaluate(model, graph, features, labels, valid_mask)
    # backward propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dur.append(time.time() - cur)
    cur = time.time()
    
    if e % 5 == 0:
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time {:.4f}".format(
              e, loss.item(), acc, dur[-1]))

# ######################################################################
# # Save embedding
# embeddings = h.detach().numpy()
# print ('==== save the following embeddings ====')
# print (embeddings)
# path = '{}/{}_{}_embedding.bin'.format(args.out_dir, args.dataset, args.model)
# os.makedirs(args.out_dir, exist_ok=True)
# with open(path, "wb") as output_file:
#     pkl.dump(embeddings, output_file)
# print ('==== saved to {} ===='.format(path))