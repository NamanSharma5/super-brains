import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from initializations import *
from preprocessing import normalize_adj_torch



class GSRLayer(nn.Module):
  
  def __init__(self,hr_dim, device=torch.device('cpu')):
    super(GSRLayer, self).__init__()
    
    self.weights = torch.from_numpy(weight_variable_glorot(hr_dim)).type(torch.FloatTensor).to(device)
    self.weights = torch.nn.Parameter(data=self.weights, requires_grad = True).to(device)

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
    def __init__(self, in_features, out_features, dropout=0.3, act=F.prelu, device=torch.device('cpu')):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight_self = torch.nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        self.weight_ne = torch.nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_self)
        torch.nn.init.xavier_uniform_(self.weight_ne)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight_ne)
        
        output = torch.mm(adj, support)
        output += torch.mm(input, self.weight_self)
        output = self.act(output)
        return output