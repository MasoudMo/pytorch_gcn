
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

def weight_variable_glorot(input_dim, output_dim):
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

    In essence, if two nodes have high degrees, a connection that is missing
    among them will have a lower value
    """

    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj, pos_link_percentage):
    """
    This function chooses pos_link_percentage of all edges as test and validation set
    """

    # Remove self and unconnected edges
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Since the graph is undirected, the upper triangle provides all the edges
    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)

    # Extract unique edges
    edges = adj_tuple[0]

    # Extract all edges
    edges_all = sparse_to_tuple(adj)[0]

    # Determine the number of test and validation edges
    num_test = int(np.floor((edges.shape[0]/ 100.)*pos_link_percentage))
    num_val = int(np.floor((edges.shape[0] / 100.)*pos_link_percentage))

    # Assign indices to unique edges and shuffle them
    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)

    # Choose validation and test edges
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]

    # Remove test and validation edges from the original graph
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b):
        """
            Determines if a is one of the edges present in b
        """
        
        # This line will cause the matched edge to become [True True]
        rows_close = np.all((a - b[:, None]) == 0, axis=-1)
        # Return true if any member is [True True] indicating that there was a match
        return np.any(rows_close)

    # Randomly choose unconnected edges and assign them to test_edges_false
    # The number of edges in test_edges_false will be equal to number of elements in
    # test_edges
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        n_rnd = len(test_edges) - len(test_edges_false)
        rnd = np.random.randint(0, adj.shape[0], size=2 * n_rnd)
        idxs_i = rnd[:n_rnd]                                        
        idxs_j = rnd[n_rnd:]
        for i in range(n_rnd):
            idx_i = idxs_i[i]
            idx_j = idxs_j[i]
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

    # Randomly choose unconnected edges. These can be chosen from test_edges
    # since our model is not aware of the test connections during training
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        n_rnd = len(val_edges) - len(val_edges_false)
        rnd = np.random.randint(0, adj.shape[0], size=2 * n_rnd)
        idxs_i = rnd[:n_rnd]                                        
        idxs_j = rnd[n_rnd:]
        for i in range(n_rnd):
            idx_i = idxs_i[i]
            idx_j = idxs_j[i]
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])

    # Re-build adj matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    # Since the graph is undirected, the transpose of the reconstructed graph should be added to it
    adj_train = adj_train + adj_train.T

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

# def get_roc_score(edges_pos, edges_neg):
#     feed_dict.update({placeholders['dropout']: 0})
#     emb = sess.run(model.embeddings, feed_dict=feed_dict)

#     def sigmoid(x):
#         return 1 / (1 + np.exp(-x))

#     # Predict on test set of edges
#     adj_rec = np.dot(emb, emb.T)
#     preds = []
#     pos = []
#     for e in edges_pos:
#         preds.append(sigmoid(adj_rec[e[0], e[1]]))
#         pos.append(adj_orig[e[0], e[1]])

#     preds_neg = []
#     neg = []
#     for e in edges_neg:
#         preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
#         neg.append(adj_orig[e[0], e[1]])

#     preds_all = np.hstack([preds, preds_neg])
#     labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
#     roc_score = roc_auc_score(labels_all, preds_all)
#     ap_score = average_precision_score(labels_all, preds_all)

#     return roc_score, ap_score

class GraphConvolution(nn.Module):
    """
    Basic graph convolution layer for undirected graph without edge labels.
    """

    def __init__(self, input_dim, output_dim, adj, dropout=0., act=nn.ReLU()):
        super(GraphConvolution, self).__init__()
        self.issparse = False
        self.act = act
        self.drop_layer = nn.Dropout(p=dropout)
        self.adj = adj
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.weights = weight_variable_glorot(input_dim, output_dim)
        self.fc1.weight.data = nn.Parameter(self.weights)
    def forward(self, x):       
        x = self.drop_layer(x)
        x = self.fc1(x)
        x = self.act(torch.sparse.mm(self.adj, x))
        return x

class GraphConvolutionSparse():
    """
    Graph convolution layer for sparse inputs.
    """

    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=nn.ReLU()):
        self.issparse = False
        self.weights = nn.parameter(weight_variable_glorot(input_dim, output_dim))
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def forward(self, x):
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = torch.sparse.mm(x, self.weights)
        x = torch.sparse.mm(self.adj, x)
        x = self.act(x)
        return x
