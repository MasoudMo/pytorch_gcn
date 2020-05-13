
from __future__ import division
from __future__ import print_function

import time

# numpy
import numpy as np
import scipy.sparse as sp

# PyTorch
import torch
import torch.optim as optim
import torch.nn as nn

# Layers
from source.layers import GraphConvolutionSparse
from source.layers import GraphConvolution
from source.layers import InnerProductDecoder

# data functions
from source.data import load_data
from source.data import mask_test_edges
from source.data import preprocess_graph

# Other functions
from source.functions import sparse_to_tuple
from source.functions import get_roc_score
from source.functions import tuples_to_torch_sparse

# Set random seed
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)


class GCNModel(nn.Module):
    """
    GCN Network class

    Attributes:
        adj (sparse torch FloatTensor):
        dropout (float): dropout rate
        input_dim (int): number of features
        output_dim (int): output feature dimension
        hidden_dim (int): hidden feature dimension
        features_nonzero (int): number of non-zero features
        graph_conv_sparse (GraphConvolutionSparse)
        graph_conv (GraphConvolution)
        inner_prod (InnerProductDecoder)

    """

    def __init__(self, adj, input_dim, output_dim,  dropout, hidden_dim, features_nonzero):
        """
        GCN Network constructor

        Parameters:
            adj (sparse torch FloatTensor):
            dropout (float): dropout rate
            input_dim (int): number of features
            output_dim (int): output feature dimension
            hidden_dim (int): hidden feature dimension
            features_nonzero (int): number of non-zero features

        """

        super(GCNModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.features_nonzero = features_nonzero
        self.adj = adj
        self.dropout = dropout

        self.graph_conv_sparse = GraphConvolutionSparse(input_dim=self.input_dim,
                                                        output_dim=self.hidden_dim,
                                                        adj=self.adj,
                                                        features_nonzero=self.features_nonzero,
                                                        act=nn.ReLU(),
                                                        dropout=self.dropout)

        self.graph_conv = GraphConvolution(input_dim=self.hidden_dim,
                                           output_dim=self.output_dim,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout)

        self.inner_prod = InnerProductDecoder(act=lambda x: x)
        
    def forward(self, x):
        """
        Forward path

        Parameters:
            x (sparse torch FloatTensor): input sparse FloatTensor

        Returns:
            (dense torch tensor): dense torch tensor for reconstructions
            (dense torch tensor): dense torch tensor for embeddings

        """

        hidden1 = self.graph_conv_sparse(x)

        embeddings = self.graph_conv(hidden1)

        reconstructions = self.inner_prod(embeddings)
  
        return reconstructions, embeddings


def cost_function(preds, labels, num_nodes, num_edges):
    """
    cost function

    Parameters:
        preds (torch tensor): predictions
        labels (torch tensor): labels
        num_nodes(int): number of nodes
        num_edges(int): number of edges

    returns:
        (float): cross entropy loss

    """

    pos_weight = torch.tensor(float(num_nodes**2 - num_edges) / num_edges, dtype=torch.float32, requires_grad=False)
    norm = num_nodes**2 / float((num_nodes**2 - num_edges) * 2)
    
    preds_sub = preds
    labels_sub = labels

    labels_sub = torch.reshape(labels_sub, [-1])

    cross_entropy_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight,
                                              reduction='mean')

    cost = norm * cross_entropy_loss(preds_sub, labels_sub)

    return cost


def main():

    adj = load_data('../data/yeast.edgelist')
    num_nodes = adj.shape[0]
    num_edges = adj.sum()

    # Featureless (one-hot encoding)
    features = sparse_to_tuple(sp.identity(num_nodes))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    features = tuples_to_torch_sparse(features)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, 2)
    adj = adj_train

    # Normalize adjacency matrix
    adj_norm = preprocess_graph(adj)
    adj_norm = tuples_to_torch_sparse(adj_norm)

    dropout = 0.1
    hidden_dim = 32
    output_dim = 16

    # Create model
    model = GCNModel(adj=adj_norm,
                     input_dim=num_features,
                     output_dim=output_dim,
                     dropout=dropout,
                     hidden_dim=hidden_dim,
                     features_nonzero=features_nonzero)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Learnable parameters are:")
    for name, param in model.named_parameters():
        print(name)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    adj_label = tuples_to_torch_sparse(adj_label)

    for epoch in range(20):
        t = time.time()
        optimizer.zero_grad()   # zero the gradient buffers
        reconstructions, embeddings = model(features)
        loss = cost_function(reconstructions, adj_label.to_dense(), num_nodes, num_edges)
        loss.backward()
        optimizer.step()    # Does the update

        # Performance on validation set
        with torch.no_grad():
            _, embeddings = model(features)
            roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false, adj_orig, embeddings)

        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(loss),
              "val_roc=", "{:.5f}".format(roc_curr),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t))

    with torch.no_grad():
        _, embeddings = model(features)
        roc_score, ap_score = get_roc_score(test_edges, test_edges_false, adj_orig, embeddings)
        print('Test ROC score: {:.5f}'.format(roc_score))
        print('Test AP score: {:.5f}'.format(ap_score))


if __name__ == '__main__':
    main()