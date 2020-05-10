
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
    """Reads edge list representation and create the adjacency matrix"""
    g = nx.read_edgelist('yeast.edgelist')
    adj = nx.adjacency_matrix(g)
    return adj)

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



