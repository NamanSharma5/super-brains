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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import torch
import matplotlib.pyplot as plt


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

    
def final_validation(model, val_loader, args):
    model.eval()

    mae_bc = []
    mae_ec = []
    mae_pc = []

    pred_1d_list = []
    gt_1d_list = []

    for lr, hr in val_loader:
        lr = lr.reshape(160, 160)
        hr = hr.reshape(268, 268)

        model_outputs,net_outs,start_gcn_outs,layer_outs = model(lr)
        model_outputs  = unpad(model_outputs, args.padding)
        prediction = model_outputs.detach().numpy()
        prediction[prediction < 0] = 0

        gt = hr.detach().numpy()

        pred_graph = nx.from_numpy_array(prediction, edge_attr="weight")
        gt_graph = nx.from_numpy_array(gt, edge_attr="weight")

        # Compute centrality measures
        pred_bc = nx.betweenness_centrality(pred_graph, weight="weight")
        pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight")
        pred_pc = nx.pagerank(pred_graph, weight="weight")

        gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
        gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight")
        gt_pc = nx.pagerank(gt_graph, weight="weight")

        # Convert centrality dictionaries to lists
        pred_bc_values = list(pred_bc.values())
        pred_ec_values = list(pred_ec.values())
        pred_pc_values = list(pred_pc.values())

        gt_bc_values = list(gt_bc.values())
        gt_ec_values = list(gt_ec.values())
        gt_pc_values = list(gt_pc.values())

        # Compute MAEs
        mae_bc.append(mean_absolute_error(pred_bc_values, gt_bc_values))
        mae_ec.append(mean_absolute_error(pred_ec_values, gt_ec_values))
        mae_pc.append(mean_absolute_error(pred_pc_values, gt_pc_values))

        # Vectorize matrices
        pred_1d_list.append(MatrixVectorizer.vectorize(prediction))
        gt_1d_list.append(MatrixVectorizer.vectorize(gt))

    # Compute average MAEs
    avg_mae_bc = np.array(sum(mae_bc) / len(mae_bc))
    avg_mae_ec = np.array(sum(mae_ec) / len(mae_ec))
    avg_mae_pc = np.array(sum(mae_pc) / len(mae_pc))

    # Concatenate flattened matrices
    pred_1d = np.concatenate(pred_1d_list)
    gt_1d = np.concatenate(gt_1d_list)

    # Compute metrics
    mae = np.array(mean_absolute_error(pred_1d, gt_1d))
    pcc = np.array(pearsonr(pred_1d, gt_1d)[0])
    js_dis = np.array(jensenshannon(pred_1d, gt_1d))

    print("MAE: ", mae)
    print("PCC: ", pcc)
    print("Jensen-Shannon Distance: ", js_dis)
    print("Average MAE betweenness centrality:", avg_mae_bc)
    print("Average MAE eigenvector centrality:", avg_mae_ec)
    print("Average MAE PageRank centrality:", avg_mae_pc)

    return mae, pcc, js_dis, avg_mae_bc, avg_mae_ec, avg_mae_pc


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
    epochs = 2
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

def plot_fold_evaluation_colored(fold_mae, fold_pcc, fold_js_dis, fold_avg_mae_bc, fold_avg_mae_ec, fold_avg_mae_pc):
    num_folds = len(fold_mae)
    metrics = ['MAE', 'PCC', 'JSD', 'MAE (PC)', 'MAE (EC)', 'MAE (BC)']
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
    
    # Create a figure with subplots
    fig, axs = plt.subplots(1, num_folds + 1, figsize=(20, 5), constrained_layout=True)
    
    # Plot data for each fold
    for i in range(num_folds):
        data = [fold_mae[i], fold_pcc[i], fold_js_dis[i], fold_avg_mae_pc[i], fold_avg_mae_ec[i], fold_avg_mae_bc[i]]
        axs[i].bar(metrics, data, color=colors)
        axs[i].set_title(f'Fold {i+1}')
        axs[i].set_ylim([0, 1])  # Adjust the y-axis limit
    
    # Calculate the average across folds
    avg_data = [
        np.mean(fold_mae),
        np.mean(fold_pcc),
        np.mean(fold_js_dis),
        np.mean(fold_avg_mae_pc),
        np.mean(fold_avg_mae_ec),
        np.mean(fold_avg_mae_bc)
    ]
    
    # Calculate the standard error for error bars
    std_error = [
        np.std(fold_mae) / np.sqrt(num_folds),
        np.std(fold_pcc) / np.sqrt(num_folds),
        np.std(fold_js_dis) / np.sqrt(num_folds),
        np.std(fold_avg_mae_pc) / np.sqrt(num_folds),
        np.std(fold_avg_mae_ec) / np.sqrt(num_folds),
        np.std(fold_avg_mae_bc) / np.sqrt(num_folds)
    ]
    
    # Plot the average data with error bars and different colors
    axs[-1].bar(metrics, avg_data, color=colors, yerr=std_error, capsize=5)
    axs[-1].set_title('Avg. Across Folds')
    
    # Set common labels
    for ax in axs:
        ax.set_ylabel('Scores')
        ax.set_xticklabels(metrics, rotation=45, ha="right")
    
    # Show the plot
    plt.show()


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
    fold_mae = []
    fold_pcc = []
    fold_js_dis = []
    fold_avg_mae_bc = []
    fold_avg_mae_ec = []
    fold_avg_mae_pc = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(brain_dataset)):
        val_dataloader = DataLoader(brain_dataset, batch_size=1, sampler=SubsetRandomSampler(val_ids))
        val_loader = fabric.setup_dataloaders(val_dataloader)
        model = models[fold]

        mae, pcc, js_dis, avg_mae_bc, avg_mae_ec, avg_mae_pc = final_validation(model, val_loader, args)
        fold_mae.append(mae)
        fold_pcc.append(pcc)
        fold_js_dis.append(js_dis)
        fold_avg_mae_bc.append(avg_mae_bc)
        fold_avg_mae_ec.append(avg_mae_ec)
        fold_avg_mae_pc.append(avg_mae_pc)

    
    plot_fold_evaluation_colored(np.array(fold_mae), np.array(fold_pcc), np.array(fold_js_dis), np.array(fold_avg_mae_bc), np.array(fold_avg_mae_ec), np.array(fold_avg_mae_pc))

if __name__ == '__main__':
    sample_mae = [0.1, 0.2, 0.3]
    sample_pcc = [0.1, 0.2, 0.3]
    sample_js_dis = [0.1, 0.2, 0.3]
    sample_avg_mae_bc = [0.1, 0.2, 0.3]
    sample_avg_mae_ec = [0.1, 0.2, 0.3]
    sample_avg_mae_pc = [0.1, 0.2, 0.3]
    main()
# %%
