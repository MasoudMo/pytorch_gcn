"""
Author: Masoud Mokhtari

This code is a PyTorch transalation of http://snap.stanford.edu/deepnetbio-ismb/ipynb/Graph+Convolutional+Prediction+of+Protein+Interactions+in+Yeast.html

Convolution and dropout layers used in the GCN model

"""

from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from source.functions import weight_variable_glorot


class DropoutSparse(nn.Module):
    """
    Implements dropout on a torch sparse matrix

    Attributes:
        keep_prob (float): probability to keep nodes
        num_nonzero_elems (int): number of non zero elements
    """

    def __init__(self, keep_prob, num_nonzero_elems):
        """
        Constructor for DropoutSparse class

        Parameters:
            keep_prob (float): probability to keep nodes
            num_nonzero_elems (int): number of non zero elements

        """

        super(DropoutSparse, self).__init__()

        self.keep_prob = keep_prob
        self.num_nonzero_elems = num_nonzero_elems

    def forward(self, x):
        """
        Forward path

        Parameters:
            x (Torch FloatTensor): input torch parse matrix

        Returns:
            (Torch FloatTensor): Torch FloatTensor after dropout

        """

        noise_shape = [self.num_nonzero_elems]
        random_tensor = self.keep_prob
        random_tensor += torch.rand(noise_shape)
        random_tensor = random_tensor.clone().detach()

        dropout_mask = torch.floor(random_tensor).type(dtype=torch.bool)

        indices_to_keep = x._indices()[:, dropout_mask]
        values_to_keep = x._values()[dropout_mask]

        return torch.sparse.FloatTensor(indices_to_keep,
                                        values_to_keep*(1./self.keep_prob),
                                        x.size())


class GraphConvolution(nn.Module):
    """
    Basic graph convolution layer for undirected graph without edge labels

    Attributes:
        input_dim (int): input dimension
        output_dim (int): output dimension
        adj (torch sparse FloatTensor): sparse adjacency matrix
        dropout (float): dropout rate
        act (function): activation function
    """

    def __init__(self, input_dim, output_dim, adj, dropout=0., act=nn.ReLU()):
        """
        Constructor for GraphConvolution class

        Parameters:
            input_dim (int): input dimension
            output_dim (int): output dimension
            adj (torch sparse FloatTensor): sparse adjacency matrix
            dropout (float): dropout rate
            act (function): activation function

        """

        super(GraphConvolution, self).__init__()
        self.adj = adj
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Layers
        self.drop_layer = nn.Dropout(p=self.dropout)
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)

        # Weight initialization
        self.weights = weight_variable_glorot(input_dim, output_dim)
        self.fc1.weight.data = nn.Parameter(self.weights.t())

    def forward(self, x):
        """
        Forward path

        Parameters:
            x (torch tensor): input dense torch tensor

        Returns:
            (dense torch tensor): dense torch tensor after convolution

        """

        x = self.drop_layer(x)
        x = self.fc1(x)
        x = torch.sparse.mm(self.adj, x)
        x = self.act(x)

        return x


class GraphConvolutionSparse(nn.Module):
    """
    Basic graph convolution layer for sparse input

    Attributes:
        input_dim (int): input dimension
        output_dim (int): output dimension
        adj (torch sparse FloatTensor): sparse adjacency matrix
        features_nonzero (int): number of non-zero features
        dropout (float): dropout rate
        act (function): activation function
    """

    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=nn.ReLU()):
        """
        Constructor for GraphConvolution class

        Parameters:
            input_dim (int): input dimension
            output_dim (int): output dimension
            adj (torch sparse FloatTensor): sparse adjacency matrix
            features_nonzero (int): number of non-zero features
            dropout (float): dropout rate
            act (function): activation function

        """

        super(GraphConvolutionSparse, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj = adj
        self.dropout = dropout
        self.act = act
        self.features_nonzero = features_nonzero

        self.weights = nn.Parameter(weight_variable_glorot(self.input_dim, self.output_dim), requires_grad=True)

        self.drop_layer = DropoutSparse(1-self.dropout, self.features_nonzero)

    def forward(self, x):
        """
        Forward path

        Parameters:
            x (sparse torch FloatTensor): input sparse torch FloatTensor

        Returns:
            (Torch dense tensor): Torch dense tensor after convolution

        """

        x = self.drop_layer(x)
        x = torch.sparse.mm(x, self.weights)
        x = torch.sparse.mm(self.adj, x)
        x = self.act(x)

        return x


class InnerProductDecoder(nn.Module):
    """
    Decoder model layer for link prediction

    Attributes:
        dropout (float): dropout rate
        act (function): activation function

    """
    def __init__(self, dropout=0., act=nn.Sigmoid()):

        """
        Constructor for GraphConvolution class

        Parameters:
            dropout (float): dropout rate
            act (function): activation function

        """

        super(InnerProductDecoder, self).__init__()

        self.dropout = dropout

        self.drop_layer = nn.Dropout(p=self.dropout)
        self.act = act

    def forward(self, x):
        """
        Forward path

        Parameters:
            x (dense torch tensor): input dense torch tensor

        Returns:
            (dense torch tensor): dense torch tensor after decoding

        """

        x = self.drop_layer(x)
        x_t = torch.t(x)
        x = torch.matmul(x, x_t)
        x = torch.reshape(x, [-1])
        x = self.act(x)

        return x
