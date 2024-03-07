import torch
import torch.nn as nn
from layers import *
from ops import *
from preprocessing import normalize_adj_torch
import torch.nn.functional as F

class GSRNet(nn.Module):

  def __init__(self,ks,args):
    super(GSRNet, self).__init__()
    
    self.lr_dim = args.lr_dim
    self.hr_dim = args.hr_dim
    self.hidden_dim = args.hidden_dim
    self.layer = GSRLayer(self.hr_dim)
    self.net = GraphUnet(ks, self.lr_dim, self.hr_dim, self.hr_dim, args.p)
    self.net2 = GraphUnet(ks, self.hr_dim, self.hr_dim, self.hr_dim, args.p)
    self.gc1 = GraphConvolution(self.hr_dim, self.hidden_dim, args.p, act=F.selu)
    self.gc2 = GraphConvolution(self.hidden_dim, self.hidden_dim, args.p, act=F.selu)
    self.gc3 = GraphConvolution(self.hidden_dim, self.hidden_dim, args.p, act=F.selu)
    self.gc4 = GraphConvolution(self.hidden_dim, self.hr_dim, args.p, act=F.selu)

  def forward(self,lr):
    """
    Returns
    model_outputs: final super-resolution
    net_outs: node embeddings after first U-net
    start_gcn_outs: node embeddings after first GCN
    layer_outs: final node embeddings after GSRLayer
    """

    I = torch.eye(self.lr_dim).type(torch.FloatTensor) # LR node embeddings
    A = normalize_adj_torch(lr).type(torch.FloatTensor)

    # net_outs = learnt LR node embeddings , start_gcn_outs = embeddings of U-net after donwsampling
    self.net_outs, self.start_gcn_outs = self.net(A, I)
    
    self.outputs, self.Z = self.layer(A, self.net_outs)
    
    self.net_outs2, self.start_gcn_outs2 = self.net2(self.outputs, self.Z)
    
    self.hidden1 = self.gc1(self.net_outs2, self.outputs)
    self.hidden2 = self.gc2(self.hidden1, self.outputs)
    self.hidden3 = self.gc3(self.hidden2, self.outputs)
    self.hidden4 = self.gc4(self.hidden3, self.outputs)

    z = self.hidden4
    z = (z + z.t())/2
    idx = torch.eye(self.hr_dim, dtype=bool) 
    z[idx]=1
    
    return torch.abs(z), self.net_outs, self.start_gcn_outs, self.outputs
        #  model_outputs,net_outs,start_gcn_outs (hidden dimension),layer_outs = model(lr)