import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import normalize_adj_torch

""" Adapted from github.com/basiralab/GSR-Net"""

def weight_variable_glorot(output_dim):

    input_dim = output_dim
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = np.random.uniform(-init_range, init_range,(input_dim, output_dim))
    
    return initial

class GSRLayer(nn.Module):
  
  def __init__(self, hr_dim):
    super(GSRLayer, self).__init__()
    
    self.weights = torch.from_numpy(weight_variable_glorot(hr_dim)).type(torch.FloatTensor)
    self.weights = torch.nn.Parameter(data=self.weights, requires_grad = True)

  def forward(self, A, X):
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
    def __init__(self, in_features, out_features, dropout=0.3, act=F.prelu): # changed to prelu 
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight_self = torch.nn.Parameter(torch.FloatTensor(in_features, out_features)) # added extra learnable params to determine self embedding
        self.weight_ne = torch.nn.Parameter(torch.FloatTensor(in_features, out_features)) 
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_self)
        torch.nn.init.xavier_uniform_(self.weight_ne)

    def forward(self, adj, input):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight_ne)
        
        output = torch.mm(adj, support)
        output += torch.mm(input, self.weight_self)
        output = self.act(output)
        return output


class GraphUnpool(nn.Module):

    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]])
        new_X[idx] = X
        return A, new_X

    
class GraphPool(nn.Module):

    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        scores = self.proj(X)
        scores = torch.abs(scores)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores/100)
        num_nodes = A.shape[0]
        values, idx = torch.topk(scores, int(self.k*num_nodes))
        new_X = X[idx, :]
        values = torch.unsqueeze(values, -1)
        new_X = torch.mul(new_X, values)
        A = A[idx, :]
        A = A[:, idx]
        return A, new_X, idx


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, p=0):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=p)

    def forward(self, A, X):
        X = self.drop(X)
        X = torch.matmul(A, X)
        X = self.proj(X)
        return X

class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim=320, p=0):
        super(GraphUnet, self).__init__()
        self.ks = ks
       
        self.start_gcn = GCN(in_dim, dim)
        self.bottom_gcn = GCN(dim, dim)
        self.end_gcn = GCN(2*dim, out_dim)
        self.down_gcns = []
        self.up_gcns = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, p))
            self.up_gcns.append(GCN(dim, dim, p))
            self.pools.append(GraphPool(ks[i], dim))
            self.unpools.append(GraphUnpool())

    def forward(self, A, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        X = self.start_gcn(A, X)
        start_gcn_outs = X
        org_X = X
        for i in range(self.l_n):
           
            X = self.down_gcns[i](A, X)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)
        X = self.bottom_gcn(A, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
           
            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, idx)
            X = self.up_gcns[i](A, X)
            X = X.add(down_outs[up_idx])
        X = torch.cat([X, org_X], 1)
        X = self.end_gcn(A, X)
        
        return X, start_gcn_outs