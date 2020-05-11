
######################### Dependencies ############################

from __future__ import division
from __future__ import print_function

import time

# numpy and sparse matrix operations
import numpy as np
import scipy.sparse as sp

# Python package for complex networks
import networkx as nx

# Required metric functions
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set random seed
seed = 123
np.random.seed(seed)
torch.manual_seed(seed

def load_data():
    """
    Reads edge list representation and create the adjacency matrix
    """
    g = nx.read_edgelist('yeast.edgelist')
    adj = nx.adjacency_matrix(g)
    return adj

def weight_variable_glorot(input_dim, output_dim, name=""):
    """
    Create a weight variable with Glorot & Bengio initialization.

    Visit this page for more info on this initilization:
        https://jamesmccaffrey.wordpress.com/2017/06/21/neural-network-glorot-initialization/

    Paper is available at:
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    Alternatively, can directly use torch.nn.init.xavier_uniform_(tensor, gain=1.0)
    """

    init_range = np.sqrt(6.0 / (input_dim + output_dim))

    x = torch.empty(input_dim, output_dim, dtype=torch.float32).uniform_(-init_range, init_range)
    x.requires_grad = True

    return x

def sparse_to_tuple(sparse_mx):
    """ 
        Change a sparse matrix into tuple format:
        e.g.:
        coords: coords[i] shows coordinates for values[i]
    """

    # to learn more about coordinate matrix format use this link:
    # https://scipy-lectures.org/advanced/scipy_sparse/coo_matrix.html
    # in essence, it converts sparse matrix into 3 arrays: data, row and col
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape

    return coords, values, shape

class DropoutSparse(nn.Module):
    """
    Implements dropout on a torch.sparse_coo type sparce matrix 
    """  

    def __init__(self, keep_prob, num_nonzero_elems):

        super(DropoutSparse, self).__init__()

        self.keep_prob = keep_prob
        self.num_nonzero_elems = num_nonzero_elems

    def forward(self, x):

        noise_shape = [self.num_nonzero_elems]
        random_tensor = self.keep_prob
        random_tensor += torch.rand(noise_shape)
        dropout_mask = torch.floor(random_tensor).type(dtype=torch.bool)

        indices_to_keep = x._indices()[:,dropout_mask]
        values_to_keep = x._values()[dropout_mask]

        return torch.sparse.FloatTensor(indices_to_keep, 
                                        values_to_keep*(1./(keep_prob)), 
                                        x.size())

def preprocess_graph(adj):
    """
    Normalizes the adjacency matrix using node degrees

    This normalization is explained here:
    https://people.orie.cornell.edu/dpw/orie6334/Fall2016/lecture7.pdf
    """

    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    return sparse_to_tuple(adj_normalized)