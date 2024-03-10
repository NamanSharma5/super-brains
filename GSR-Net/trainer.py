#%%
from lightning.fabric import Fabric, seed_everything
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataloaders import brain_dataset
from preprocessing import *
from sklearn.model_selection import KFold
import argparse
from model import *
from train import test
import torch.optim as optim
import pandas as pd
from MatrixVectorizer import *
import networkx as nx
from typing import Union


#%%
def train(fabric, model, train_loader, optimizer, criterion, args):
    model.train()

    epoch_loss = []
    epoch_error = []
    epoch_topo = []

    for lr, hr in train_loader:      
        lr = lr.reshape(160, 160)
        hr = hr.reshape(268, 268)

        model_outputs,net_outs,start_gcn_outs,layer_outs = model(lr)
        model_outputs  = unpad(model_outputs, args.padding)

        padded_hr = pad_HR_adj(hr,args.padding)
        eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')

        loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(model.layer.weights,U_hr) + criterion(model_outputs, hr) 

        topo = args.lamdba_topo * compute_topological_MAE_loss(hr, model_outputs)

        error = criterion(model_outputs, hr)

        optimizer.zero_grad()
        fabric.backward(loss)
        optimizer.step()

        epoch_loss.append(loss.item())
        epoch_error.append(error.item())
        epoch_topo.append(topo.item())
        
    print("Loss: ", np.mean(epoch_loss), "Error: ", np.mean(epoch_error),
        "Topo: ", np.mean(epoch_topo))

    
def validate(fabric, model, val_loader, criterion, args):
    model.eval()
    val_loss = []
    val_error = []
    val_topo = []

    with torch.no_grad():
        for lr, hr in val_loader:
            lr = lr.reshape(160, 160)
            hr = hr.reshape(268, 268)

            model_outputs,net_outs,start_gcn_outs,layer_outs = model(lr)
            model_outputs  = unpad(model_outputs, args.padding)

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
    return np.mean(val_loss)

def hyperparameters():
    num_splt = 3
    epochs = 5
    lr = 0.00005 # try [0.0001, 0.0005, 0.00001, 0.00005]
    lmbda = 17 # should be around 15-20
    lamdba_topo = 1 # should be around 0.5-1.5
    lr_dim = 160
    hr_dim = 320
    hidden_dim = 320 # try smaller and larger - [160-512]
    padding = 26
    dropout = 0.1 # try [0., 0.1, 0.2, 0.3]
    args = argparse.Namespace()
    args.epochs = epochs
    args.lr = lr
    args.lmbda = lmbda
    args.lamdba_topo = lamdba_topo
    args.lr_dim = lr_dim
    args.hr_dim = hr_dim
    args.hidden_dim = hidden_dim
    args.padding = padding
    args.p = dropout
    args.seed = 123
    return args

#%%

def main():
    fabric = Fabric(accelerator='cpu')

    fabric.launch()
    args = hyperparameters()
    args.device = fabric.device
    ks = [0.9, 0.7, 0.6, 0.5]

    seed_everything(args.seed)

    kfold = KFold(n_splits=3, random_state=42, shuffle=True)
    models = [GSRNet(ks, args) for _ in range(kfold.n_splits)]
    optimizers = [optim.Adam(model.parameters(), lr=args.lr) for model in models]
    for i in range(kfold.n_splits):
        models[i], optimizers[i] = fabric.setup(models[i], optimizers[i])

    criterion = nn.L1Loss()

    for epoch in range(args.epochs):
        for fold, (train_ids, val_ids) in enumerate(kfold.split(brain_dataset)):
            print(f"Working on fold {fold}")

            train_dataloader = DataLoader(brain_dataset, batch_size=1, sampler=SubsetRandomSampler(train_ids))
            val_dataloader = DataLoader(brain_dataset, batch_size=1, sampler=SubsetRandomSampler(val_ids))

            train_loader, val_loader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

            model, optimizer = models[fold], optimizers[fold]

            # train and validate
            train(fabric, model, train_loader, optimizer, criterion, args)
            validate(fabric, model, val_loader, criterion, args)
    
    print(f"Running final evalutation")
    fold_losses = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(brain_dataset)):
        val_dataloader = DataLoader(brain_dataset, batch_size=1, sampler=SubsetRandomSampler(val_ids))
        val_loader = fabric.setup_dataloaders(val_dataloader)
        model = models[fold]
        fold_losses.append(validate(fabric, model, val_loader, criterion, args))

    print(fold_losses)




    

    train(fabric, model, train_loader, optimizer, criterion, args)
    validate(fabric, model, val_loader, criterion, args)


if __name__ == '__main__':
    main()
# %%
