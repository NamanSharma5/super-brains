import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from model import *

from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pickle


from MatrixVectorizer import *
from preprocessing import *
from model import *

criterion = nn.L1Loss()

def generate_submission_csv(model, args, data_path='./data/lr_test.csv', filename='testset-preds.csv'):
    lr_test_data = pd.read_csv(data_path, delimiter=',').to_numpy()
    lr_test_data[lr_test_data < 0] = 0
    np.nan_to_num(lr_test_data, copy=False)
    lr_test_data_vectorized = np.array([MatrixVectorizer.anti_vectorize(row, 160) for row in lr_test_data])

    model.eval()
    preds = []
    for lr in lr_test_data_vectorized:      
        lr = torch.from_numpy(lr).type(torch.FloatTensor)
        model_outputs, _, _, _ = model(lr)
        model_outputs  = unpad(model_outputs, args.padding)
        preds.append(MatrixVectorizer.vectorize(model_outputs.detach().numpy()))

    r = np.hstack(preds)
    meltedDF = r.flatten()
    n = meltedDF.shape[0]
    df = pd.DataFrame({'ID': np.arange(1, n+1),
                    'Predicted': meltedDF})
    df.to_csv(filename, index=False)

def train(model, train_data_loader, optimizer, criterion, args, name='model'): 
  
    all_epochs_loss = []
    all_epochs_error = []
    all_epochs_topoloss = []
    no_epochs = args.epochs

    for epoch in range(no_epochs):
        epoch_loss = []
        epoch_error = []
        epoch_topo = []

        model.train()
        for lr, hr in train_data_loader:  
            lr = lr.reshape(160, 160)
            hr = hr.reshape(268, 268)

            model_outputs,net_outs,start_gcn_outs,layer_outs = model(lr)
            model_outputs  = unpad(model_outputs, args.padding)

            padded_hr = pad_HR_adj(hr,args.padding)
            _, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')

            loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(model.layer.weights,U_hr) + criterion(model_outputs, hr) 
            topo = compute_topological_MAE_loss(hr, model_outputs)
            
            loss += args.lamdba_topo * topo

            error = criterion(model_outputs, hr)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            epoch_error.append(error.item())
            epoch_topo.append(topo.item())
        
    
        model.eval()
        print("Epoch: ",epoch+1, "Loss: ", np.mean(epoch_loss), "Error: ", np.mean(epoch_error),
            "Topo: ", np.mean(epoch_topo))
        all_epochs_loss.append(np.mean(epoch_loss))
        all_epochs_error.append(np.mean(epoch_error))
        all_epochs_topoloss.append(np.mean(epoch_topo))


    df = pd.DataFrame({'Epoch': np.arange(1, no_epochs+1),
                    'Total Loss': all_epochs_loss,
                    'Error': all_epochs_error,
                    'Topological loss': all_epochs_topoloss,
                    })

    df.to_csv(f'{name}-losses.csv', index=False)
    pickle.dump(model, open(f"{name}.sav", 'wb'))
  

def validate(model, val_loader, criterion, args, csv=False, filename=None):
    model.eval()
    val_loss = []
    val_error = []
    val_topo = []
    preds = []

    with torch.no_grad():
        for lr, hr in val_loader:
            lr = lr.reshape(160, 160)
            hr = hr.reshape(268, 268)

            model_outputs,net_outs,start_gcn_outs,layer_outs = model(lr)
            model_outputs  = unpad(model_outputs, args.padding)
            preds.append(MatrixVectorizer.vectorize(model_outputs.detach().numpy()))

            padded_hr = pad_HR_adj(hr,args.padding)
            eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')

            loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(model.layer.weights,U_hr) + criterion(model_outputs, hr) 

            topo = args.lamdba_topo * compute_topological_MAE_loss(hr, model_outputs)

            error = criterion(model_outputs, hr)

            val_loss.append(loss.item())
            val_error.append(error.item())
            val_topo.append(topo.item())

    print("Validation Loss: ", np.mean(val_loss), "Validation Error: ", np.mean(val_error),
          "Validation Topo: ", np.mean(val_topo))
    if csv:
        r = np.hstack(preds)
        meltedDF = r.flatten()
        n = meltedDF.shape[0]
        df = pd.DataFrame({'ID': np.arange(1, n+1),
                        'Predicted': meltedDF})
        df.to_csv(f"{filename}.csv", index=False)
    return np.mean(val_loss)
    
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
