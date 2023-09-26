#!!! Change flag in main if small or large dataset is used
from models import *

from typing import Tuple, Optional
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, GATv2Conv, Sequential, GraphSAGE, BatchNorm
from torch_geometric.nn.pool import global_mean_pool
from torch.nn import Linear, Embedding, ModuleList
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm, trange

import numpy as np
import traceback
import torch
import torch_geometric
import torch.nn.functional as F
import random
import wandb

import sys
sys.path.append('../utils')

from helpers import load_data,\
                    load_stations,\
                    clean_data,\
                    normalize_data,\
                    create_data,\
                    compute_dist_matrix


def train_model(config):
    # !Fixed parameters not varied in this run
    n_epochs = 500  # max number of epochs
    patience = 30
    n_reps = 1
    lr = 0.0025 if config.model != 'GlobalInfo' else 0.001

    def train(batch):
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch_id=batch.batch)
        loss = crps(out, batch.y)
        loss.backward()
        optimizer.step()
        return loss

    @torch.no_grad()
    def valid(batch):
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch_id=batch.batch)
        loss = crps(out, batch.y)
        return loss

    model_list = []
    train_losses_models = []
    validation_losses_models = []

    for i in range(n_reps):
        train_losses = []
        validation_losses = []
        best_val_loss = float('inf')
        no_improvement = 0

        # loading bar
        epochs_pbar = trange(n_epochs, desc="Epochs")
        model, optimizer, scheduler = create_model(layer_type=config.model,
                                                   embed_dim=config.embed_dim,
                                                   in_channels=emb_num_features,
                                                   hidden_channels=config.hidden_channels,
                                                   num_layers=config.num_layers,
                                                   lr=lr,
                                                   heads=config.heads,
                                                   schedule_lr=False,
                                                   compile_model=True,
                                                   device=device)

        for epoch in epochs_pbar:
            # Train for one epoch
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                loss = train(batch)
                train_loss += loss.item() * batch.num_graphs
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            # Evaluate on the validation set
            model.eval()
            val_loss = 0.0

            for batch in valid_loader:
                loss = valid(batch)
                val_loss += loss.item() * batch.num_graphs
            val_loss /= len(valid_loader.dataset)
            validation_losses.append(val_loss)

            # Log to WandB
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})

            # Check if the validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
            else:
                no_improvement += 1

            epochs_pbar.set_postfix({"Train Loss": train_loss,
                                     "Val Loss": val_loss,
                                     "Best Loss": best_val_loss,
                                     "No Improvement": no_improvement})

            # Early stopping
            if no_improvement == patience:
                print('Early stopping.')
                break

        # Log WandB
        wandb.log({"best_val_loss": best_val_loss, "trained_epochs": epoch - patience})


if __name__ == '__main__':
    small = True  # !change here if the small or big dataset is used
    with wandb.init():
        # Set Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {device}")
        config = wandb.config
        # Get Data from feather
        print("Loading Data ...")
        data = load_data(indexed=False)

        # Clean Data
        print("Cleaning Data ...")
        data = clean_data(data, max_missing=121, max_alt=1000.0)

        # Cut Dataset to only 2015 and 2016 (2016 wont be used anyway)
        if small:
            data = data[data.date.dt.year >= 2015]
            data = data[data.date.dt.year < 2016]
        else:
            # Cut data to 2007-2015
            data = data[data.date.dt.year < 2016]

        # Normalize Data
        print("Normalizing Data ...")
        # last_obs is -365 since the last year is used for testing
        normalized_data = normalize_data(data, last_obs=-1, method="max")

        # Get List of stations with all stations -> will break further code if cut already
        print("Extracting Stations ...")
        stations = load_stations(data)

        # Create Torch Data #####
        dist_matrix = np.load('dist_matrix.npy')
        position_matrix = np.array(stations[['station', 'lon', 'lat']])

        max_dist = None
        nearest_k = None
        if config.generate_graph < 15:
            method = "nearest_k"
            nearest_k = config.generate_graph
        else:
            method = "max_dist"
            max_dist = config.generate_graph
        wandb.log({"graph_generation": method})

        torch_data = []
        for date in tqdm(data['date'].unique(), desc="Creating PyG Data"):
            torch_data.append(create_data(df=normalized_data,
                                          date=date,
                                          dist_matrix=dist_matrix,
                                          position_matrix=position_matrix,
                                          method=method,
                                          max_dist=max_dist,
                                          k=nearest_k,
                                          nearest_k_mode="in"))

        # Create Dataloader #####
        # Definition of train_loader and valid_loader
        # !Define Batch Size
        BS = 8
        # Move all the data directly to the GPU (should fit into memory)
        torch_data_train = [t.to(device) for t in torch_data]
        # shuffle data
        random.shuffle(torch_data_train)
        # Definition of train_loader and valid_loader
        if small:
            train_loader = DataLoader(torch_data_train[:183], batch_size=BS, shuffle=True)
            valid_loader = DataLoader(torch_data_train[183:], batch_size=BS, shuffle=True)
        else:
            train_loader = DataLoader(torch_data_train[:1899], batch_size=BS, shuffle=True)
            valid_loader = DataLoader(torch_data_train[1899:], batch_size=BS, shuffle=True)

        # get feature length
        num_features = torch_data[0].num_features
        emb_num_features = num_features + config.embed_dim - 1

        # Start Training #####
        try:
            train_model(config)
        except Exception as e:
            # exit gracefully, so wandb logs the problem
            print(traceback.print_exc())
            exit(1)
