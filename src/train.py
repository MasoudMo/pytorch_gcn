"""
Author: Masoud Mokhtari

This code is a PyTorch translation of http://snap.stanford.edu/deepnetbio-ismb/ipynb/Graph+Convolutional+Prediction+of+Protein+Interactions+in+Yeast.html

Simple implementation of vanilla GCN model on protein interactions in yeast

"""

from __future__ import division
from __future__ import print_function

import time

# Command line argument parse
import argparse

# numpy
import numpy as np
import scipy.sparse as sp

# PyTorch
import torch
import torch.optim as optim
import torch.nn as nn

# Layers
from layers import GraphConvolutionSparse
from layers import GraphConvolution
from layers import InnerProductDecoder

# data functions
from data import load_data
from data import mask_test_edges
from data import preprocess_graph

# Other functions
from functions import sparse_to_tuple
from functions import get_roc_score
from functions import tuples_to_torch_sparse

# Set random seed
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)


def dummy_func(x):
    return x


class GCNModel(nn.Module):
    """
    GCN Network class

    Attributes:
        adj (sparse torch FloatTensor): input sparse torch tensor
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
                                           act=dummy_func,
                                           dropout=self.dropout)

        self.inner_prod = InnerProductDecoder(act=dummy_func)
        
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
        labels (torch dense tensor): labels
        num_nodes(int): number of nodes
        num_edges(int): number of edges

    returns:
        (float): cross entropy loss

    """

    # If num_edges is small, pos_weight will be higher to reduce false negative predictions
    pos_weight = torch.tensor(float(num_nodes**2 - num_edges) / num_edges, dtype=torch.float32, requires_grad=False)

    # Normalization factor
    norm = num_nodes**2 / float((num_nodes**2 - num_edges) * 2)
    
    preds_sub = preds
    labels_sub = torch.reshape(labels, [-1])

    cross_entropy_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight,
                                              reduction='mean')

    cost = norm * cross_entropy_loss(preds_sub, labels_sub)

    return cost


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='GCN training on sparse graph in edgelist format')
    parser.add_argument('--input_path', type=str, required=True, help='path to edgelist file.')
    parser.add_argument('--dropout', type=float, default='0.1', help='dropout rate')
    parser.add_argument('--hidden_dim', type=int, default='32', help='hidden layer dimension')
    parser.add_argument('--output_dim', type=int, default='16', help='output feature dimension')
    parser.add_argument('--learning_rate', type=float, default='0.01', help='learning rate')
    parser.add_argument('--epochs', type=int, default='20', help='number of epochs')
    parser.add_argument('--model_path', type=str, default='../models/model.pt',
                        help='path to save the trained model to.')
    args = parser.parse_args()

    # Model parameters
    dropout = args.dropout
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    lr = args.learning_rate
    epochs = args.epochs

    # Load data
    adj = load_data(args.input_path)

    # Extract number of nodes and edges
    num_nodes = adj.shape[0]
    num_edges = adj.sum()

    # Print graph information
    print("Graph Statistics:")
    print("number of nodes: {:d}".format(num_nodes))
    print("number of edges: {:d} \n".format(num_edges))

    # Extract features (one-hot encoding since this is featureless)
    features = sparse_to_tuple(sp.identity(num_nodes))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    features = tuples_to_torch_sparse(features)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj_orig.eliminate_zeros()

    # Remove validation and test edges and create training graph
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, 2)
    adj = adj_train

    # Normalize adjacency matrix
    adj_norm = preprocess_graph(adj)
    adj_norm = tuples_to_torch_sparse(adj_norm)

    # Create model
    model = GCNModel(adj=adj_norm,
                     input_dim=num_features,
                     output_dim=output_dim,
                     dropout=dropout,
                     hidden_dim=hidden_dim,
                     features_nonzero=features_nonzero)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Print model parameters
    print("Learnable parameters are:")
    for name, param in model.named_parameters():
        print(name)

    # Create labels matrix
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    adj_label = tuples_to_torch_sparse(adj_label)

    # Start training the model
    for epoch in range(epochs):
        t = time.time()
        # Zero the gradient buffers
        optimizer.zero_grad()
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

    # Find test data accuracy
    with torch.no_grad():
        _, embeddings = model(features)
        roc_score, ap_score = get_roc_score(test_edges, test_edges_false, adj_orig, embeddings)
        print('Test ROC score: {:.5f}'.format(roc_score))
        print('Test AP score: {:.5f}'.format(ap_score))

    # Save the trained model
    torch.save(model, args.model_path)


if __name__ == '__main__':
    main()