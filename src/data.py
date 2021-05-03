"""
Author: Masoud Mokhtari

This code is a PyTorch translation of http://snap.stanford.edu/deepnetbio-ismb/ipynb/Graph+Convolutional+Prediction+of+Protein+Interactions+in+Yeast.html

Data methods used for loading and preprocessing data and creating the test and validation set

"""

from __future__ import division
from __future__ import print_function

# numpy and sparse matrix operations
import numpy as np
import scipy.sparse as sp

# Python package for complex networks
import networkx as nx

# Import helper functions
from functions import sparse_to_tuple


def load_data(edge_list_file_path):
    """
    Reads edge list representation and creates the adjacency matrix

    Parameters:
        edge_list_file_path (str): path to edgelist file

    Return:
        (Sparse Matrix): adjacency matrix in sparse matrix format

    """

    g = nx.read_edgelist(edge_list_file_path)
    adj = nx.adjacency_matrix(g)
    return adj


def preprocess_graph(adj):
    """
    Normalizes the adjacency matrix using node degrees

    This normalization is explained here:
    https://people.orie.cornell.edu/dpw/orie6334/Fall2016/lecture7.pdf

    In essence, if two nodes have high degrees, a connection that is missing
    among them will have a lower value

    Parameters:
        adj (sparse matrix): input sparse matrix to be normalized

    Returns:
        (sparse matrix tuples): tuples of normalized sparse matrix

    """

    if not sp.isspmatrix_coo(adj):
        adj = sp.coo_matrix(adj)

    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())

    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    return sparse_to_tuple(adj_normalized)


def mask_test_edges(adj, pos_link_percentage):
    """
    This function chooses pos_link_percentage of all edges as test and validation set

    Parameters:
        adj (parse matrix): parse matrix choose test and val data from
        pos_link_percentage (float): number from 0 to 100 indicating percentage of val and test data

    Returns:
        (CSR matrix): train sparse matrix (with val and test edges removed)
        (list): list of training edges
        (list): list of validation edges
        (list): list of validation false edges
        (list): list of test edges
        (list): list of test false edges

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
    num_test = int(np.floor((edges.shape[0] / 100.) * pos_link_percentage))
    num_val = int(np.floor((edges.shape[0] / 100.) * pos_link_percentage))

    # Assign indices to unique edges and shuffle them
    all_edge_idx = np.arange(edges.shape[0])
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

            Parameters:
                a (list): one edge
                b (ndarray): list of edges

            Returns:
                (bool): indicates whether a is an element of b or not

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
