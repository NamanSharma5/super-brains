import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from initializations import *
from preprocessing import normalize_adj_torch


class GSRLayer(nn.Module):
  
  def __init__(self,hr_dim):
    super(GSRLayer, self).__init__()
    
    self.weights = torch.from_numpy(weight_variable_glorot(hr_dim)).type(torch.FloatTensor)
    self.weights = torch.nn.Parameter(data=self.weights, requires_grad = True)

  def forward(self,A,X):
    # print('A shape: ', A.shape, ' X shape: ', X.shape)
    lr = A
    lr_dim = lr.shape[0]
    hr_dim = self.weights.shape[0]
    f = X
    eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO='U') # replaced deprecated lin
    U_lr = torch.abs(U_lr)
    eye_mat = torch.eye(lr_dim).type(torch.FloatTensor)
    s_d = torch.cat((eye_mat,eye_mat),0)
    
    a = torch.matmul(self.weights,s_d )
    b = torch.matmul(a ,torch.t(U_lr))
    f_d = torch.matmul(b ,f)
    f_d = torch.abs(f_d)
    self.f_d = f_d.fill_diagonal_(1)
    adj = normalize_adj_torch(self.f_d)
    X = torch.mm(adj, adj.t())
    X = (X + X.t())/2
    idx = torch.eye(hr_dim, dtype=bool)
    X[idx]=1
    return adj, torch.abs(X)
    


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    #160x320 320x320 =  160x320
    def __init__(self, in_features, out_features, dropout=0.3, act=F.leaky_relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        output = self.act(output)
        return output



class GINConvolution(nn.Module):
    """
    Simple GIN layer, as described in https://arxiv.org/abs/1810.00826

    H(l+1) = sigma ( (1+epsilon) H (l) + Aggregation of Neighbours Node Embeddings)

    sigma - non-linear activation function
    epsilon - learnable parameter controls mixing of node to neighbours
    Aggregation method - sum

    """

    def __init__(self, in_features, out_features, dropout=0.3, eps=0, act=F.leaky_relu):
        super(GINConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.eps = nn.Parameter(torch.Tensor([eps]))  # in pytorch docs, eps is not learnable by default - its; a hyperparm
        self.act = act
        self.linear = nn.Linear(in_features, out_features)  # final output transformation <- can replace with MLP
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        nn.init.constant_(self.eps, 0.0)  # Initializing eps to 0

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        # Assuming adj is an adjacency list for neighbors
        # Aggregation step, summing over neighbors
        neighbor_sum = torch.matmul(adj, input)

        # Combine the node's own features and its neighbors
        output = (1 + self.eps) * input + neighbor_sum

        # Apply linear transformation and activation function
        output = self.act(self.linear(output))
        return output


class GATLayer(nn.Module):
    """
    A basic implementation of the GAT layer.

    This layer applies an attention mechanism in the graph convolution process,
    allowing the model to focus on different parts of the neighborhood
    of each node.
    """
    def __init__(self, in_features, out_features, activation=None):
        super(GATLayer, self).__init__()
        # Initialize the weights, bias, and attention parameters as
        # trainable parameters
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.phi = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        stdv = 1. / np.sqrt(self.phi.size(1))
        self.phi.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        Forward pass of the GAT layer.

        Parameters:
        input (Tensor): The input features of the nodes.
        adj (Tensor): The adjacency matrix of the graph.

        Returns:
        Tensor: The output features of the nodes after applying the GAT layer.
        """
        N = input.size(0)

        # 1. Apply linear transform add bias
        H = torch.matmul(input, self.weight) + self.bias
        # init sim matrix
        S = torch.zeros((N,N), device=input.device)
        # 2. compute attention scores
        H_expanded = H.repeat(N, 1)  # repeat H [1, 2, 3] ->  [1,2,3,1,2,3] node gets repeated N times

        H_interleaved = H.repeat_interleave(N, dim=0) # repeats each elem N times H[1,2,3] -> [1,1,2,2,3,3]
        # Now concat should take elem H_expanded[i] and H_interleaved[j] which gives NxN of concatenated
        concat_feats = torch.cat([H_interleaved, H_expanded], dim=1)
        S = torch.matmul(concat_feats, self.phi).view(N, N)

        # Compute mask based on adjacency
        I = torch.eye(N, device=input.device)
        mask = (adj + I) != 0

        # 4 - Apply mask  to the pre-attention matrix
        S_masked = torch.where(mask, S, torch.tensor(float('-inf'), device=input.device))

        # 5 compute attention weights using softmax
        attention_weights = F.softmax(S_masked, dim=1)

        # 6 - Aggregate feature based on attention weights
        h = torch.matmul(attention_weights, H)

        return self.activation(h) if self.activation else h