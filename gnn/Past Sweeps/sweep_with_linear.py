# Important to import helpers
import sys

sys.path.append('../utils')

from helpers import load_data, load_stations, clean_data, normalize_data, create_data, visualize_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, Sequential, summary
from torch_geometric.utils import to_networkx
from torch.nn import Linear, Embedding, Dropout, ModuleList
from tqdm import tqdm, trange

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopy.distance
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import traceback
import wandb


def build_dataloaders(max_dist: int, batch_size: int):
    dist_matrix = np.load('dist_matrix.npy')

    # Create a boolean mask indicating which edges to include
    mask = (dist_matrix <= max_dist) & (dist_matrix != 0)

    torch_data = []
    for date in tqdm(data['date'].unique(), desc="Building dataset"):
        torch_data.append(create_data(df=normalized_data, date=date, mask=mask, dist_matrix=dist_matrix))

    # Definition of train_loader and valid_loader
    train_loader = DataLoader(torch_data[:-730], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(torch_data[-730:-365], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(torch_data[-365:], batch_size=batch_size, shuffle=True)
    train_loader_small = DataLoader(torch_data[-1460:-730], batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader, train_loader_small


def crps(mu: torch.tensor, sigma: torch.tensor, y: torch.tensor):
    """Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed df

    Args:
        mu (torch.tensor): mean
        sigma (torch.tensor): standard deviation
        y (torch.tensor): observed df

    Returns:
        torch.tensor: CRPS value
    """
    y = y.view((-1, 1))  # make sure y has the right shape
    PI = np.pi  # 3.14159265359
    omega = (y - mu) / sigma
    # PDF of normal distribution at omega
    pdf = 1 / (torch.sqrt(torch.tensor(2 * PI))) * torch.exp(-0.5 * omega ** 2)

    # Source: https://stats.stackexchange.com/questions/187828/how-are-the-error-function-and-standard-normal-distribution-function-related
    cdf = 0.5 * (1 + torch.erf(omega / torch.sqrt(torch.tensor(2))))

    crps = sigma * (omega * (2 * cdf - 1) + 2 * pdf - 1 / torch.sqrt(torch.tensor(PI)))
    return torch.mean(crps)


class Convolution(torch.nn.Module):
    def __init__(self, out_channels, hidden_channels, heads, num_layers: int = None):
        super(Convolution, self).__init__()
        # Make sure either hidden_channels is a list, heads is a list or num_layer is supplied
        assert isinstance(hidden_channels, list) or isinstance(heads, list) or num_layers is not None, \
            "If hidden_channels is not a list, num_layers must be specified."
        # both are a list
        if isinstance(hidden_channels, list) and isinstance(heads, list):
            assert len(hidden_channels) == len(heads), \
                f"Lengths of lists {len(hidden_channels)} and {len(heads)} do not match."
        # only hidden_channels is list
        if isinstance(hidden_channels, list) and not isinstance(heads, list):
            heads = [heads] * len(hidden_channels)
        # only heads is list
        if isinstance(heads, list) and not isinstance(hidden_channels, list):
            hidden_channels = [hidden_channels] * len(heads)
        # none is list
        if not isinstance(heads, list) and not isinstance(hidden_channels, list):
            heads = [heads] * num_layers
            hidden_channels = [hidden_channels] * num_layers

        # definition of Layers
        self.convolutions = ModuleList()
        for c, h in zip(hidden_channels, heads):
            self.convolutions.append(GATv2Conv(in_channels=-1, out_channels=c, heads=h, edge_dim=1))
        # Last Layer to match shape of output
        self.lin = Linear(in_features=hidden_channels[-1] * heads[-1], out_features=out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = x.float()
        edge_attr = edge_attr.float()

        for conv in self.convolutions:
            x = F.relu(conv(x, edge_index, edge_attr))

        x = F.relu(self.lin(x))
        return x


class EmbedStations(torch.nn.Module):
    def __init__(self, num_stations_max, embedding_dim):
        super(EmbedStations, self).__init__()
        self.embed = Embedding(num_embeddings=num_stations_max, embedding_dim=embedding_dim)

    def forward(self, x):
        station_ids = x[:, 0].long()
        emb_station = self.embed(station_ids)
        x = torch.cat((emb_station, x[:, 1:]), dim=1)  # Concatenate embedded station_id to rest of the feature vector
        return x


class MakePositive(torch.nn.Module):
    def __init__(self):
        super(MakePositive, self).__init__()

    def forward(self, x):
        mu, sigma = torch.split(x, 1, dim=-1)
        sigma = F.softplus(sigma)  # ensure that sigma is positive
        return mu, sigma


class ResGnn(torch.nn.Module):
    def __init__(self, out_channels, num_layers, hidden_channels, heads):
        super(ResGnn, self).__init__()
        assert num_layers > 0, "num_layers must be > 0."

        # Create Layers
        self.convolutions = ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convolutions.append(GATv2Conv(-1, hidden_channels, heads=heads, edge_dim=1))
            else:
                self.convolutions.append((GATv2Conv(-1, hidden_channels, heads=heads, edge_dim=1)))
        self.lin = Linear(hidden_channels * heads, out_channels)  # hier direkt 2 testen

    def forward(self, x, edge_index, edge_attr):
        x = x.float()
        edge_attr = edge_attr.float()
        for i, conv in enumerate(self.convolutions):
            if i == 0:
                # First Layer
                x = conv(x, edge_index, edge_attr)
                x = F.relu(x)
            else:
                x = x + F.relu(conv(x, edge_index, edge_attr))  # Residual Layers

        x = self.lin(x)
        x = F.relu(x)
        return x


def build_model(embed_dim: int, hidden_channels: int, heads: int, num_layers: int, linear_size: int, type: str):
    """Builds  a model with the specified parameters

    Args:
        embed_dim (int): embedding dimension of the station id
        hidden_channels (int): number of hidden channels used by the convolution layers
        heads (int): number of heads used for the attention of the convolution layers
        num_layers (int): depth of the convolution layers
        linear_size (int): size of the linear layer
        type (str): type of the model, either 'ResGNNv2' or 'GATConvv2'


    Returns:
        _type_: returns a model with the specified parameters
    """
    torch.cuda.empty_cache()

    if type == 'ResGNNv2':
        conv = (ResGnn(out_channels=linear_size, hidden_channels=hidden_channels, heads=heads, num_layers=num_layers),
                'x, edge_index, edge_attr -> x')
    elif type == 'GATConvv2':
        conv = (Convolution(out_channels=linear_size, hidden_channels=hidden_channels, heads=heads, num_layers=num_layers),
                'x, edge_index, edge_attr -> x')

    model = Sequential('x, edge_index, edge_attr',
                       [
                           (EmbedStations(num_stations_max=535, embedding_dim=embed_dim), 'x -> x'),
                           conv,
                           (Linear(linear_size, 2),'x -> x'),
                           (MakePositive(), 'x -> mu, sigma')
                       ])
    model.to(device)

    return model


def build_optimizer(model, learning_rate: float) -> torch.optim.Optimizer:
    """Defines the optimizer for the model


    Args:
        model (_type_): model for which the optimizer is defined
        learning_rate (float): learning rate

    Returns:
        torch.optim.Optimizer: returns the optimizer for the model
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer


def eval(model, test_loader):
    model.eval()
    mu_list = []
    sigma_list = []
    err_list = []
    y_list = []

    for batch in test_loader:
        batch.to(device)
        mu, sigma = model(batch.x, batch.edge_index, batch.edge_attr)
        y = batch.y
        err = crps(mu, sigma, y)
        mu = mu.detach().cpu().numpy().flatten()
        sigma = sigma.detach().cpu().numpy().flatten()
        y = y.cpu().numpy()
        err = err.detach().cpu().numpy()

        mu_list.append(mu)
        sigma_list.append(sigma)
        y_list.append(y)
        err_list.append(err * len(batch))

    err = sum(err_list) / len(test_loader.dataset)
    return err


def train_model():
    with wandb.init():
        config = wandb.config
        train_loader, valid_loader, test_loader, train_loader_small = build_dataloaders(max_dist=config.max_dist, batch_size=config.batch_size)

        model = build_model(embed_dim=config.embed_dim,
                            hidden_channels=config.hidden_channels,
                            heads=config.heads,
                            num_layers=config.num_layers,
                            linear_size=config.linear_size,
                            type=config.type)

        optimizer = build_optimizer(model=model, learning_rate=config.learning_rate)

        best_val_loss = float('inf')

        def train(batch):
            batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            mu, sigma = out
            loss = crps(mu, sigma, batch.y)
            loss.backward()
            optimizer.step()
            return loss

        @torch.no_grad()
        def valid(batch):
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            mu, sigma = out
            loss = crps(mu, sigma, batch.y)
            return loss

        epochs_pbar = trange(config.max_epochs, desc="Epochs")
        for epoch in epochs_pbar:
            # Train for one epoch
            model.train()
            train_loss = 0.0
            for batch in train_loader_small:
                loss = train(batch)
                train_loss += loss.item() * batch.num_graphs
            train_loss /= len(train_loader.dataset)

            # Evaluate on the validation set
            model.eval()
            val_loss = 0.0
            for batch in valid_loader:
                loss = valid(batch)
                val_loss += loss.item() * batch.num_graphs
            val_loss /= len(valid_loader.dataset)

            # Check if the validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
                # Save model checkpoint
                wandb.log({"best_val_loss": best_val_loss})
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, "checkpoint.pt")
            else:
                no_improvement += 1

            # Log to WandB
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})
            epochs_pbar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss, "Best Loss": best_val_loss,
                                     "No Improvement": no_improvement})
            # Early stopping
            if no_improvement == config.patience:
                print('Early stopping.')
                break

        # Load weights from model checkpoint
        checkpoint = torch.load("checkpoint.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        test_error = eval(model=model, test_loader=test_loader)

        wandb.log({"best_val_loss": best_val_loss,
                   "trained_epochs": epoch - config.patience,
                   "evaluation_error": test_error})

        # Free memory
        model.to('cpu')
        torch.cuda.empty_cache()


def train_model_catch_errors():
    try:
        train_model()
    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc())
        exit(1)


if __name__ == '__main__':
    # Get Data from feather
    data = load_data(indexed=False)
    # Get List of stations with all stations -> will break further code if cut already
    stations = load_stations(data)
    # Clean Data
    data = clean_data(data, max_missing=121, max_alt=1000.0)
    # Normalize Data
    normalized_data = normalize_data(data, last_obs=-365)  # last_obs is -365 since the last year is used for testing
    # Set Device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # Start Training
    train_model_catch_errors()
