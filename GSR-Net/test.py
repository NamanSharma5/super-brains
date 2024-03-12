import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from model import *

criterion = nn.L1Loss()
    
def test(model, test_adj, test_labels, args):

  test_error = []
  
  # TESTING
  for lr, hr in zip(test_adj, test_labels):

    all_zeros_lr = not np.any(lr)
    all_zeros_hr = not np.any(hr)

    if all_zeros_lr == False and all_zeros_hr == False: #choose representative subject
      lr = torch.from_numpy(lr).type(torch.FloatTensor)
      np.fill_diagonal(hr,1)
      hr = torch.from_numpy(hr).type(torch.FloatTensor)
      preds, _, _, _ = model(lr)
      preds = unpad(preds, args.padding)
      
      error = criterion(preds, hr)
      test_error.append(error.item())

  print ("Test error MAE: ", np.mean(test_error))
  return np.mean(test_error)
