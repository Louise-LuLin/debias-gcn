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
from models import GAT, GraphSAGE

# Load dataset for node classification
def load_data(data_type='cora'):
    if data_type == 'cora':
        dataset = dgl.data.CoraGraphDataset()
    else:
        dataset = dgl.data.CiteseerGraphDataset()
    
    graph = dataset[0]
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    valid_mask = graph.ndata['val_mask']
    test_mask  = graph.ndata['test_mask']
    return graph, features, labels, train_mask, valid_mask, test_mask

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct =torch.sum(indices==labels)
        return correct.item() * 1.0 / len(labels) # Accuracy

model_type = 'sage'
data_type = 'cora'

graph, features, labels, train_mask, valid_mask, test_mask = load_data(data_type)
n_features = features.shape[1]
n_labels = int(labels.max().item() + 1)

if model_type == 'sage':
    model = GraphSAGE(graph,
                      in_dim=n_features, 
                      hidden_dim=64, 
                      out_dim=n_labels)
else:
    model = GAT(graph,
                in_dim=n_features,
                hidden_dim=8,
                out_dim=n_labels,
                num_heads=8)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

dur = []
cur = time.time()
for e in range(100):
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
