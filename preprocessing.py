import torch
import numpy as np

def pad_HR_adj(label, split):

  label=np.pad(label,((split,split),(split,split)),mode="constant")
  np.fill_diagonal(label,1)
  return torch.from_numpy(label).type(torch.FloatTensor)

def normalize_adj_torch(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx

def unpad(data, split):
  
  idx_0 = data.shape[0]-split
  idx_1 = data.shape[1]-split
  train = data[split:idx_0, split:idx_1]
  return train