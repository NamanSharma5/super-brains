from typing import Union
import torch
import torch.nn.functional as F
import numpy as np

def compute_degree_nonzero(adj):
    return torch.count_nonzero(adj, axis=0)

def compute_degree_sum(adj):
    return torch.sum(adj, axis=0)

def compute_topological_MAE_loss(graph1,graph2:Union[np.ndarray,torch.Tensor]):
    return F.l1_loss(compute_degree_sum(graph1), compute_degree_sum(graph2))
