import argparse
import time

import numpy as np
import torch.cuda
import yaml
from torch_geometric.loader import DataLoader
import sys
import os
import warnings
from models import *
from typing import Tuple, Optional
from tqdm import trange

sys.path.append('../utils')
from helpers import load_data, load_stations, clean_data, normalize_data, create_data, compute_dist_matrix


def create_model_from_config(config: dict, emb_num_features: int, device: torch.cuda.device)\
        -> Tuple[torch.nn.Module, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
    """
    Create a model using a config dict
    :param config: config dict
    :param emb_num_features: number of input features
    :param device: device to run the model on
    :return:
    """
    model, optimizer, scheduler = create_model(layer_type=config['model']['architecture'],
                                               embed_dim=config['model']['embed_dim'],
                                               in_channels=emb_num_features,
                                               hidden_channels=config['model']['hidden_channels'],
                                               num_layers=config['model']['num_layers'],
                                               lr=config['model']['lr'],
                                               heads=config['model']['heads'],
                                               schedule_lr=False,
                                               compile_model=config['model']['compile'],
                                               device=device
                                               )
    return model, optimizer, scheduler


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch):
    optimizer.zero_grad()
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch_id=batch.batch)
    loss = crps(out, batch.y)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def valid(model: torch.nn.Module, batch):
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch_id=batch.batch)
    loss = crps(out, batch.y)
    return loss


def main():
    start_time = time.time()
    # Suppress warning
    warnings.filterwarnings("ignore", category=UserWarning)
    # Argparse
    parser = argparse.ArgumentParser(description='Training of a single Model')
    parser.add_argument('--id', '-i', type=int, help='id of model run')
    parser.add_argument('--config', '-c', type=str, help='path to yaml config file')
    parser.add_argument('--device', '-d', type=str, help='device number')
    parser.add_argument('--small', '-s', action=argparse.BooleanOptionalAction,
                        help='use small dataset for model estimation')
    args = parser.parse_args()

    # Hyperparameters stored in yaml file
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    print(f"Model {args.id} start")
    # Load Data
    # Get Data from feather
    data = load_data(indexed=False)

    if not os.path.exists("dist_matrix.npy"):
        stations = load_stations(data)  # This needs to be done here because we need all stations
        dist_matrix = compute_dist_matrix(stations)
        np.save('dist_matrix.npy', dist_matrix)

    # Create correlation matrix for further graph generating methods
    if not os.path.exists("corr_matrix.npy"):
        # Creation of new Graph using Correlation of observations
        data_cut = data.iloc[:-1460]
        time_series = data_cut.pivot(index='date', columns='station', values='obs')
        time_series = time_series.interpolate()
        corr_matrix = time_series.corr().clip(lower=0)
        corr_matrix = 1 - corr_matrix
        corr_matrix = np.nan_to_num(corr_matrix, nan=1)
        np.save('corr_matrix.npy', corr_matrix)

    # Clean Data
    data = clean_data(data, max_missing=121, max_alt=1000.0)

    # Cut Dataset to only 2015 and 2016 (2016 wont be used anyway)
    if args.small:
        data = data[data.date.dt.year >= 2015]

    # Normalize Data
    normalized_data = normalize_data(data, last_obs=-len(data[data.date.dt.year == 2016]), method="max")

    # Get List of stations with all stations -> will break further code if cut already
    stations = load_stations(data)

    # Create Dataset
    if config['data']['use_corr']:
        dist_matrix = np.load('corr_matrix.npy')
    else:
        dist_matrix = np.load('dist_matrix.npy')
    position_matrix = np.array(stations[['station', 'lon', 'lat']])

    torch_data = []
    dates = data['date'].unique()[:-367]  # last 366 days are used for testing
    for date in dates:
        torch_data.append(create_data(df=normalized_data,
                                      date=date,
                                      dist_matrix=dist_matrix,
                                      position_matrix=position_matrix,
                                      method=config['data']['method'],
                                      max_dist=config['data']['max_dist'],
                                      k=config['data']['nearest_k'],
                                      nearest_k_mode=config['data']['nearest_k_mode']))

    # Set device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)

    # Create Dataloader
    # Move all the data directly to the GPU (should fit into memory)
    #torch_data_train = [tensor.to(device) for tensor in torch_data]
    torch_data_train = torch_data

    # Definition of train_loader and valid_loader
    train_loader = DataLoader(torch_data_train, batch_size=config['model']['batch_size'], shuffle=True)
    len(f"Days to train on {train_loader.dataset}")
    # Model Creation
    # Number of Features
    num_features = torch_data[0].num_features
    emb_num_features = num_features + config['model']['embed_dim'] - 1

    model, optimizer, _ = create_model_from_config(config, emb_num_features, device)

    # Model to GPU
    model.to(device)

    # Train Loop
    train_losses = []
    pbar = trange(config['training']['n_epochs'])
    for epoch in pbar:
        # Train for one epoch
        model.train()
        train_loss = 0.0
        # Train loop
        for batch in train_loader:
            batch.to(device)
            loss = train(model, optimizer, batch)
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        pbar.set_postfix({"Train_Loss": train_loss})
        if torch.isnan(loss).any():
            print("Loss is NaN. Training terminated.")
            sys.exit(1)

    # path
    save_path = os.path.dirname(args.config)
    if not os.path.exists(f"{save_path}/checkpoints"):
        os.makedirs(f"{save_path}/checkpoints")

    # Save model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{save_path}/checkpoints/model_{args.id}.pt")

    elapsed_time = time.time()-start_time
    print(f"Model {args.id} trained in: {int(elapsed_time // 60)}min {int(elapsed_time % 60)}s ")


if __name__ == '__main__':
    main()
