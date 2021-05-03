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
from sklearn.metrics import roc_auc_score
import dgl.data
from models import GraphSAGE

print('pytorch version: ', torch.__version__)

"""
Construct dataset for link prediction
"""
def construct_data(data_type='cora'):
    if data_type == 'cora':
        dataset = dgl.data.CoraGraphDataset()
    else:
        dataset = dgl.data.CiteseerGraphDataset()
    
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


"""
Dot product to compute the score of link 
The benefit of treating the pairs of nodes as a graph is that the score
on edge can be easily computed via the ``DGLGraph.apply_edges`` method
"""
import dgl.function as fn

# Dot product to predict the score of link
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]

# MLP to predict the score of link
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']



######################################################################
# Set dataset, model and predictor type
model_type = 'sage' # sage/gat
data_type = 'cora'  # cora/citeseer
predictor_type = 'dot' # dot/mlp

# Load and construct data for link prediction task
graph, features, train_pos_g, train_neg_g, test_pos_g, test_neg_g = construct_data(data_type)
n_features = features.shape[1]

# Initialize embedding model
if model_type == 'sage':
    model = GraphSAGE(graph,
                      in_dim=n_features, 
                      hidden_dim=64, 
                      out_dim=16)
else:
    model = GAT(graph,
                in_dim=n_features,
                hidden_dim=8,
                out_dim=16,
                num_heads=8)

# Initialize link predictor
# You can replace DotPredictor with MLPPredictor.
pred = DotPredictor() # or MLPPredictor(16)

optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=1e-3)

######################################################################
# Loss: cross entropy. You can modify here for debiasing.
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)
# Metric: AUC. You can add more metrics.
def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

######################################################################
# Training loop
dur = []
cur = time.time()
for e in range(200):
    model.train()
    # forward propagation on training set
    h = model(features)
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)
    # evaluation on test set
    model.eval()
    with torch.no_grad():
        test_pos_score = pred(test_pos_g, h)
        test_neg_score = pred(test_neg_g, h)
        test_acc = compute_auc(test_pos_score, test_neg_score)
        train_acc = compute_auc(pos_score, neg_score)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dur.append(time.time() - cur)
    cur = time.time()

    if e % 5 == 0:
        print("Epoch {:05d} | Loss {:.4f} | Train AUC {:.4f} | Test AUC {:.4f} | Time {:.4f}".format(
              e, loss.item(), train_acc, test_acc, dur[-1]))

######################################################################
# Save embedding
embeddings = h.detach().numpy()
print ('==== save the following embeddings ====')
print (embeddings)
path = './{}_{}_embedding.bin'.format(data_type, model_type)
with open(path, "wb") as output_file:
    pkl.dump(embeddings, output_file)
print ('==== saved to {} ===='.format(path))

