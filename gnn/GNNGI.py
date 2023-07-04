import sys
sys.path.append('../utils')

from helpers import load_data, load_stations, clean_data, normalize_data, create_data, visualize_graph, visualize_attention

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, GCN, GCNConv, GATConv, GATv2Conv, Sequential, summary, MLP, BatchNorm, GraphSAGE, ResGatedGraphConv, GINConv, GINEConv
from torch_geometric.utils import to_networkx, add_self_loops
from torch_geometric.nn.aggr import MeanAggregation, MLPAggregation
from torch_geometric.nn.pool import global_mean_pool
from torch.nn import Linear, Embedding, Dropout, ModuleList
from torch.optim.lr_scheduler import MultiStepLR, StepLR
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
import torch_geometric
import torch.nn.functional as F
import wandb
import warnings
import traceback

plt.style.use('default')


# Define Loss Function #####
def crps(mu: torch.tensor, sigma: torch.tensor, y: torch.tensor):
    """Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed df

    :param torch.tensor mu: mean
    :param torch.tensor sigma: standard deviation
    :param torch.tensor y: observed df

    :return tensor: CRPS value
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


class EmbedStations(torch.nn.Module):
    def __init__(self, num_stations_max, embedding_dim):
        super(EmbedStations, self).__init__()
        self.embed = Embedding(num_embeddings=num_stations_max, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor):
        station_ids = x[:, 0].long()
        emb_station = self.embed(station_ids)
        x = torch.cat((emb_station, x[:, 1:]), dim=1) # Concatenate embedded station_id to rest of the feature vector
        return x


class MakePositive(torch.nn.Module):
    def __init__(self):
        super(MakePositive, self).__init__()

    def forward(self, x: torch.Tensor):
        mu, sigma = torch.split(x, 1, dim=-1)
        sigma = F.softplus(sigma) # ensure that sigma is positive
        return mu, sigma


class MPWGI(MessagePassing):
    """
    Message Passing with Global Information
    """
    def __init__(self, in_channels, edge_channels, global_channels, hidden_channels, out_channels):
        super().__init__(aggr='mean')
        self.psi = Linear(2 * in_channels + edge_channels + global_channels, hidden_channels)
        self.phi = Linear(in_channels + hidden_channels + global_channels, out_channels)
        self.rho = Linear(global_channels + out_channels, out_channels)
        self.pool = global_mean_pool

    def reset_parameters(self)  -> None:
        """
        Reset all parameters
        :return: None
        """
        self.psi.reset_parameters()
        self.phi.reset_parameters()
        self.rho.reset_parameters()

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor, global_features_i: torch.Tensor) -> torch.Tensor:
        """
        Corresponds to psi.
        Constructs a message from node j to node i for each edge in edge_index.
        Tensors passed to propagate() can be mapped to the respective nodes i and j and by appending _i or _j
        :param x_i: features of node i
        :param x_j: features of node j
        :param edge_attr: feature form node i to node j
        :param global_features: global feature of the entire graph
        :return: message passed from node j to node i
        """
        #global_features = global_features.repeat(x_i.shape[0], 1)
        message_input = torch.cat([x_i, x_j, edge_attr, global_features_i], dim=-1)
        message_transformed = self.psi(message_input)
        return message_transformed

    def update(self, inputs: torch.Tensor, x: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """
        Corresponds to phi.
        Updates node embeddings.
        Takes in the output of aggregation as first argument and any argument which was initially passed to propagate().
        :param inputs: output of aggregation
        :param x_i: features of node i
        :param global_features: global feature of the entire graph
        :return: updated feature of node i
        """
        update_input = torch.cat([inputs, x, global_features], dim=-1)
        return self.phi(update_input)

    def forward(self, x, edge_index, edge_attr, global_features, batch_id):
        # Add self-loops to the adjacency matrix.
        # edge_index, edge_attr = add_self_loops(edge_index, edge_attr=torch.Tensor([0]))

        # Every node should have a global feature which is the same for every node in each batch
        # The batch number is stores in batch_id
        global_features_per_node = global_features[batch_id]

        # Propagate Messages
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr, global_features=global_features_per_node)

        # Update global feature
        # Corresponds to rho but only takes all nodes into consideration and omits messages
        aggregated_nodes = self.pool(x, batch=batch_id)
        global_input = torch.cat([global_features, aggregated_nodes], dim=-1)
        global_features = self.rho(global_input)

        return x, global_features


class GNNGI(torch.nn.Module):
    """
    Graph Neural Network with global Information
    """
    def __init__(self, in_channels, edge_channels, global_channels, hidden_channels, out_channels, num_layers, embedding_dim):
        """
        Create an instance of a Graph Neural Network with global Information sharing
        :param in_channels: dimension of node attributes before embedding
        :param edge_channels: dimension of edge attributes
        :param global_channels: dimension of global information vector, should be the same as in_channels+edge_channels
        :param hidden_channels: dimension of the message passed along the edges
        :param out_channels: describes the dimension of the new node features
        :param num_layers: number of layers
        :param embedding_dim: dimension to use of node embedding
        """
        super(GNNGI, self).__init__()
        self.convolutions = ModuleList()
        assert num_layers > 0, "num_layers must be greater than 0"
        for _ in range(num_layers-1):
            self.convolutions.append(MPWGI(in_channels, edge_channels, global_channels, hidden_channels, out_channels))
            in_channels = out_channels
        self.convolutions.append(MPWGI(in_channels, edge_channels, global_channels, hidden_channels, out_channels))  # Last Layer

        self.emb = EmbedStations(num_stations_max=535, embedding_dim=embedding_dim)
        self.norm_edge = BatchNorm(in_channels=1)
        self.pool = global_mean_pool
        self.lin = Linear(out_channels,2)
        self.make_pos = MakePositive()

    def forward(self, x, edge_index, edge_attr, batch_id, global_features=None):
        x = self.emb(x)
        edge_attr = self.norm_edge(edge_attr)

        if global_features is None:
            global_features = self.pool(x, batch=batch_id)  # Do batch-wise pooling

        x, global_feature = self.convolutions[0](x, edge_index, edge_attr, global_features, batch_id)
        x = F.relu(x)
        global_feature = F.relu(global_feature)

        if len(self.convolutions) > 1:
            for conv in self.convolutions[1:]:
                x_new, global_features_new = conv(x, edge_index, edge_attr, global_features, batch_id)
                x = x + F.relu(x_new)  # Resnet
                global_feature = global_feature + F.relu(global_features_new)  #Resnet

        x = self.lin(x)

        mu, sigma = self.make_pos(x)
        return mu, sigma

def create_model(EMBED_DIM, emb_num_features, hidden_channels, num_layers, LR):
    # Clear Cache
    torch.cuda.empty_cache()
    model = GNNGI(in_channels=emb_num_features,
                  edge_channels=1,
                  global_channels=emb_num_features,
                  hidden_channels=hidden_channels,
                  out_channels = emb_num_features,
                  num_layers=num_layers,
                  embedding_dim=EMBED_DIM)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR) #momentum anschauen
    # Learning Rate Scheduler
    scheduler = MultiStepLR(optimizer, milestones=[70, 90, 100, 110], gamma=0.5)
    return model, optimizer, scheduler


def eval(model_list, test_loader):
    mu_list_model = []
    sigma_list_model = []

    for model in model_list:
        mu_list_batch = []
        sigma_list_batch = []
        y_list = []

        for batch in test_loader:
            batch.to(device)
            mu, sigma = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y = batch.y
            y = y.cpu().numpy()
            y_list.append(y)

            mu = mu.detach().cpu().numpy().flatten()
            mu_list_batch.append(mu)
            sigma = sigma.detach().cpu().numpy().flatten()
            sigma_list_batch.append(sigma)

        mu = np.concatenate(mu_list_batch)
        mu_list_model.append(mu)
        sigma = np.concatenate(sigma_list_batch)
        sigma_list_model.append(sigma)
        y = np.concatenate(y_list)
    mu = np.array(mu_list_model).T.mean(axis=1).reshape(-1, 1)
    sigma = np.array(sigma_list_model).T.mean(axis=1).reshape(-1, 1)

    mu = torch.tensor(mu).to(device)
    sigma = torch.tensor(sigma).to(device)
    y = torch.tensor(y).to(device)
    err = crps(y=y, mu=mu, sigma=sigma)
    return err


def train_model(config):

    n_epochs = 200
    patience = 30
    n_reps = 1

    model_list = []
    train_losses_models = []
    validation_losses_models = []

    for i in range(n_reps):
        num_features = torch_data[0].num_features
        emb_num_features = num_features + config.embed_dim - 1

        train_losses = []
        validation_losses = []
        best_val_loss = float('inf')
        no_improvement = 0

        # loading bar
        epochs_pbar = trange(n_epochs, desc="Epochs")
        model, optimizer, scheduler = create_model(EMBED_DIM=config.embed_dim,
                                                   emb_num_features=emb_num_features,
                                                   hidden_channels=config.hidden_channels,
                                                   num_layers=config.num_layers,
                                                   LR=0.002)

        def train(batch):
            batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch_id=batch.batch)
            mu, sigma = out
            loss = crps(mu, sigma, batch.y)
            loss.backward()
            optimizer.step()
            return loss

        @torch.no_grad()
        def valid(batch):
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch_id=batch.batch)
            mu, sigma = out
            loss = crps(mu, sigma, batch.y)
            return loss

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
                # Save model checkpoint
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"checkpoints/checkpoint_model_{i}.pt")
            else:
                no_improvement += 1

            epochs_pbar.set_postfix({"Train Loss": train_loss, "Val Loss": val_loss, "Best Loss": best_val_loss,
                                     "No Improvement": no_improvement, "Learning Rate": scheduler.get_last_lr()})
            # Early stopping
            if no_improvement == patience:
                print('Early stopping.')
                break
            # Update the Learning Rate
            scheduler.step()

        # Log WandB
        wandb.log({"best_val_loss": best_val_loss, "trained_epochs": epoch - patience})

        # Load weights from model checkpoint
        checkpoint = torch.load(f"checkpoints/checkpoint_model_{i}.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Set model to eval mode
        model.eval()
        model_list.append(model)
        train_losses_models.append(train_losses)
        validation_losses_models.append(validation_losses)

        test_error = eval(model_list=model_list, test_loader=test_loader)
        # Log WandB
        wandb.log({"best_val_loss": best_val_loss,
                   "trained_epochs": epoch - patience,
                   "evaluation_error": test_error})

        # Free memory
        model.to('cpu')
        torch.cuda.empty_cache()


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

    with wandb.init():
        config = wandb.config
        # Get Data from feather
        data = load_data(indexed=False)
        # Get List of stations with all stations -> will break further code if cut already
        stations = load_stations(data)
        # Clean Data
        data = clean_data(data, max_missing=121, max_alt=1000.0)
        # Normalize Data
        normalized_data = normalize_data(data, last_obs=-1460) # last_obs is -365 since the last year is used for testing
        # Set Device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Create Torch Data #####
        dist_matrix = np.load('dist_matrix.npy')

        torch_data = []
        max_dist = None
        nearest_k = None
        if config.generate_graph < 15:
            method = "nearest_k"
            nearest_k = config.generate_graph
        else:
            method = "max_dist"
            max_dist = config.generate_graph

        wandb.log({"graph_generation": method})
        for date in tqdm(data['date'].unique()):
            torch_data.append(
                create_data(df=normalized_data, date=date, dist_matrix=dist_matrix, method=method,
                            max_dist=max_dist, k=nearest_k))

        # Create Dataloader #####
        BS = config.batch_size

        # Definition of train_loader and valid_loader
        train_loader = DataLoader(torch_data[:-1460], batch_size=BS, shuffle=True, num_workers=4)
        valid_loader = DataLoader(torch_data[-1460:-365], batch_size=BS, shuffle=True, num_workers=4)
        test_loader = DataLoader(torch_data[-365:], batch_size=BS, shuffle=False, num_workers=4)

        # Start Training
        try:
            train_model(config)
        except Exception as e:
            # exit gracefully, so wandb logs the problem
            print(traceback.print_exc())
            exit(1)
