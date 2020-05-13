
from __future__ import division
from __future__ import print_function

# numpy and sparse matrix operations
import numpy as np
import scipy.sparse as sp

# Required metric functions
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# PyTorch
import torch


def sparse_to_tuple(sparse_mx):
    """
    Change a sparse matrix into tuple format.

    Parameters:
        sparse_mx (sparse matrix): sparse matrix of either coo format or not

    Returns:
        (list): list of tuples indicating coordinates of values
        (list): list of matrix values
        (tuple): shows size of matrix

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


def weight_variable_glorot(input_dim, output_dim):
    """
    Create a weight variable with Glorot & Bengio initialization which is based on fan-in
    and fan-out

    Visit this page for more info on this initilization:
        https://jamesmccaffrey.wordpress.com/2017/06/21/neural-network-glorot-initialization/

    Paper is available at:
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    Alternatively, can directly use torch.nn.init.xavier_uniform_(tensor, gain=1.0)

    Parameters:
        input_dim (int): number of rows of tensor
        output_dim (int): number of columns of tensor

    Returns:
        (Torch Tensor of Float32 Type): Tensor of specified dims

    """

    init_range = np.sqrt(6.0 / (input_dim + output_dim))

    x = torch.empty(input_dim, output_dim, dtype=torch.float32).uniform_(-init_range, init_range)

    return x


def get_roc_score(edges_pos, edges_neg, adj_orig, emb):
    """
    Computes Receiver Operating Characteristic score

    Parameters:
        edges_pos (list): list of tuples for edges that should be present
        edges_neg (list): list of tuples for edges that should not be present
        adj_orig (sparse matrix): sparse adjacency matrix
        emb (torch tensor): torch tensor for embeddings
    """

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []

    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def tuples_to_torch_sparse(adj_tuples, requires_grad=False):
    """
    Creates a torch parse FloatTensor from matrix tuples

    Parameters:
        adj_tuples (list): list of tuples for sparse matrix
        requires_grad (bool): indicates whether tensor requires grad or not

    Returns:
        (torch sparse tensor): torch sparse FloatTensor

    """

    return torch.sparse.FloatTensor(torch.LongTensor(adj_tuples[0].transpose()),
                                    torch.FloatTensor(adj_tuples[1]),
                                    torch.Size(adj_tuples[2])).requires_grad_(requires_grad)
