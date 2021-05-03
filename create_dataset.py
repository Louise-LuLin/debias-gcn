"""
Make our own datasets

We store three components for each dataset
-- node_feature.csv: store node feature
-- node_label.csv: store node label
-- edge.csv: store the edges
"""

import dgl
from dgl.data import DGLDataset
import torch
import os
import urllib.request
import pandas as pd

class MyDataset(DGLDataset):
    def __init__(self, data_name):
        self.data_name = data_name
        super().__init__(name='customized_dataset')

    def process(self):
        # Load the data as DataFrame
        node_features = pd.read_csv('./mydata/{}_node_feature.csv'.format(self.data_name))
        node_labels = pd.read_csv('./mydata/{}_node_label.csv'.format(self.data_name))
        edges = pd.read_csv('./mydata/{}_edge.csv'.format(self.data_name))

        # Transform from DataFrame to torch tensor
        node_features = torch.from_numpy(node_features.to_numpy()).float()
        node_labels = torch.from_numpy(node_labels['Label'].to_numpy()).int()
        edge_features = torch.from_numpy(edges['Weight'].to_numpy()).float()
        edges_src = torch.from_numpy(edges['Src'].to_numpy())
        edges_dst = torch.from_numpy(edges['Dst'].to_numpy())

        # construct DGL graph
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=node_features.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = node_features.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    # TODO: Define the other datasets as you like
    def process_raw_movielens(self):
        pass

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

def process_raw_karate():
        urllib.request.urlretrieve(
            'https://data.dgl.ai/tutorial/dataset/members.csv', './members.csv')
        urllib.request.urlretrieve(
            'https://data.dgl.ai/tutorial/dataset/interactions.csv', './interactions.csv')

        edges = pd.read_csv('./interactions.csv')
        nodes = pd.read_csv('./members.csv')
        nodes['Label'] = nodes['Club'].astype('category').cat.codes

        node_feature = nodes['Age']
        node_label = nodes[['Label', 'Club']]
        
        node_feature.to_csv('./mydata/karate_node_feature.csv', sep=',')
        node_label.to_csv('./mydata/karate_node_label.csv', sep=',')
        edges.to_csv('./mydata/karate_edge.csv', sep=',')