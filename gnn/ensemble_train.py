import argparse
import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F
import random
import yaml
from typing import Tuple
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, GATv2Conv, Sequential, GraphSAGE, BatchNorm
from torch_geometric.nn.pool import global_mean_pool
from torch.nn import Linear, Embedding, ModuleList
import sys
import os
import warnings
sys.path.append('../utils')
from helpers import load_data, load_stations, clean_data, normalize_data, create_data, visualize_graph,\
    visualize_attention, compute_dist_matrix, visualize_explanation


def create_model(config, emb_num_features):
    if config['model']['architecture'] == 'GNNGI':
        model = GNNGI(in_channels=emb_num_features,
                      edge_channels=1,
                      global_channels=emb_num_features,
                      hidden_channels=config['model']['hidden_channels'],
                      out_channels=config['model']['hidden_channels'],
                      global_out_channels=emb_num_features,
                      num_layers=config['model']['num_layers'],
                      embedding_dim=config['model']['embed_dim'])
    elif config['model']['architecture'] == 'ResGnn':
        model = Sequential('x, edge_index, edge_attr, batch_id',
                           [
                               (EmbedStations(num_stations_max=535, embedding_dim=config['model']['embed_dim']), 'x -> x'),
                               (ResGnn(in_channels=-1,
                                       out_channels=2,
                                       hidden_channels=config['model']['hidden_channels'],
                                       heads=config['model']['heads'],
                                       num_layers=config['model']['num_layers']), 'x, edge_index, edge_attr -> x'),
                               (MakePositive(), 'x -> mu_sigma')
                           ])
    elif config['model']['architecture'] == 'GraphSAGE':
        model = Sequential('x, edge_index, edge_attr, batch_id',
                           [
                               (EmbedStations(num_stations_max=535, embedding_dim=config['model']['embed_dim']), 'x -> x'),
                               (GraphSAGE(in_channels=emb_num_features,
                                          hidden_channels=config['model']['hidden_channels'],
                                          num_layers=config['model']['num_layers'], out_channels=2,
                                          project=True,
                                          aggr='mean'), 'x, edge_index -> x'),
                               (MakePositive(), 'x -> mu_sigma')
                           ])
    else:
        print(f"Invalid model type {config['model']['architecture']}")
        exit(1)
    return model


def crps(mu_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed df
    :param torch.Tensor mu_sigma: tensor of mean and standard deviation
    :param torch.Tensor y: observed df

    :return tensor: CRPS value
    :rtype torch.Tensor
    """
    mu, sigma = torch.split(mu_sigma, 1, dim=-1)
    y = y.view((-1, 1))  # make sure y has the right shape
    pi = np.pi  # 3.14159265359
    omega = (y - mu) / sigma
    # PDF of normal distribution at omega
    pdf = 1 / (torch.sqrt(torch.tensor(2 * pi))) * torch.exp(-0.5 * omega ** 2)

    # Source:
    # https://stats.stackexchange.com/questions/187828/how-are-the-error-function-and-standard-normal-distribution-function-related
    cdf = 0.5 * (1 + torch.erf(omega / torch.sqrt(torch.tensor(2))))

    crps_score = sigma * (omega * (2 * cdf - 1) + 2 * pdf - 1 / torch.sqrt(torch.tensor(pi)))
    return torch.mean(crps_score)


def train(model, optimizer, batch):
    optimizer.zero_grad()
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch_id=batch.batch)
    loss = crps(out, batch.y)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def valid(model, batch):
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch_id=batch.batch)
    loss = crps(out, batch.y)
    return loss

class EmbedStations(torch.nn.Module):
    def __init__(self, num_stations_max, embedding_dim):
        super(EmbedStations, self).__init__()
        self.embed = Embedding(num_embeddings=num_stations_max, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor):
        station_ids = x[:, 0].long()
        emb_station = self.embed(station_ids)
        x = torch.cat((emb_station, x[:, 1:]),
                      dim=1)  # Concatenate embedded station_id to rest of the feature vector
        return x


class MakePositive(torch.nn.Module):
    def __init__(self):
        super(MakePositive, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, sigma = torch.split(x, 1, dim=1)
        sigma = F.softplus(sigma)  # ensure that sigma is positive
        mu_sigma = torch.cat([mu, sigma], dim=1)
        return mu_sigma


class ResGnn(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, hidden_channels: int, heads: int):
        super(ResGnn, self).__init__()
        assert num_layers > 0, "num_layers must be > 0."

        # Create Layers
        self.convolutions = ModuleList()
        for _ in range(num_layers):
            self.convolutions.append(GATv2Conv(-1, hidden_channels, heads=heads, edge_dim=1, add_self_loops=True,
                                               fill_value=0.0))  # TODO small positive or negative number can be tested
        self.lin = Linear(hidden_channels * heads, out_channels)
        self.norm = BatchNorm(heads * hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = x.float()
        edge_attr = edge_attr.float()
        for i, conv in enumerate(self.convolutions):
            if i == 0:
                # First Layer
                x = conv(x, edge_index, edge_attr)
                x = F.relu(x)
                x = self.norm(x)
            else:
                x = x + F.relu(conv(x, edge_index, edge_attr))  # Residual Layers

        x = self.lin(x)
        return x

    @torch.no_grad()
    def get_attention(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor)\
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs a forward Pass for the given graph only though the ResGNN layer.
        NOTE: the data that is given to this method must first pass through the layers before this layer in the Graph

        :param torch.Tensor x: Tensor of Node Features (NxD)
        :param torch.Tensor edge_index: Tensor of Edges (2xE)
        :param torch.Tensor edge_attr: Edge Attributes (ExNum_Attr)
        :return x, edge_index_attention, attention_weights: Tensor of Node Features (NxD), Tensor of Edges with self loops (2xE), Tensor of Attention per edge (ExNum_Heads)
        """
        x = x.float()
        edge_attr = edge_attr.float()

        # Pass Data though Layer to get the Attention
        attention_list = []
        edge_index_attention, attention_weights = None, None  # Note: edge_index_attention has to be added since we have self loops now

        for i, conv in enumerate(self.convolutions, ):
            if i == 0:
                # First Layer
                x, (edge_index_attention, attention_weights) = conv(x, edge_index, edge_attr,
                                                                    return_attention_weights=True)
                attention_list.append(attention_weights)
                x = F.relu(x)
                x = self.norm(x)
            else:
                x_conv, (edge_index_attention, attention_weights) = conv(x, edge_index, edge_attr,
                                                                         return_attention_weights=True)
                attention_list.append(attention_weights)
                x = x + F.relu(x_conv)  # Residual Layers
        x = self.lin(x)

        # TODO Average the attention across all layers
        attention_weights = attention_weights.mean(dim=1)

        return x, edge_index_attention, attention_weights


class MPWGI(MessagePassing):
    """
    Message Passing with Global Information
    """
    def __init__(self, in_channels, edge_channels, global_channels, hidden_channels, out_channels, global_out_channels):
        super().__init__(aggr='mean') # use mean aggregation
        self.W1 = Linear(in_channels, out_channels) # used in update
        self.W2 = Linear(hidden_channels, out_channels, bias=False)  # used in update
        self.W3 = Linear(global_channels, out_channels, bias=False)  # used in update
        self.W4 = Linear(2*in_channels + edge_channels + global_channels, hidden_channels)  # used in message
        self.W5 = Linear(hidden_channels + global_channels, global_out_channels)  # used for global features
        self.pool = global_mean_pool

    def reset_parameters(self) -> None:
        """
        Reset all parameters
        :return: None
        """
        self.W1.reset_parameters()
        self.W2.reset_parameters()
        self.W3.reset_parameters()
        self.W4.reset_parameters()
        self.W5.reset_parameters()

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor, global_features_i: torch.Tensor) -> torch.Tensor:
        """
        Corresponds to psi.
        Constructs a message from node j to node i for each edge in edge_index.
        Tensors passed to propagate() can be mapped to the respective nodes i and j and by appending _i or _j
        :param x_i: features of node i
        :param x_j: features of node j
        :param edge_attr: feature form node i to node j
        :param global_features_i: global feature of the entire graph (here i is added since multiple graphs can be processed in parallel, then each of the graphs should have their own global attribute)
        :return: message passed from node j to node i
        """
        message_input = torch.cat([x_i, x_j, edge_attr, global_features_i], dim=-1)
        message_transformed = self.W4(message_input)
        return message_transformed

    def update(self, inputs: torch.Tensor, x: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """
        Corresponds to phi.
        Updates node embeddings.
        Takes in the output of aggregation as first argument and any argument which was initially passed to propagate().
        :param inputs: output of aggregation
        :param x: features of node i
        :param global_features: global feature of the entire graph
        :return: updated feature of node i
        """
        own_hidden_feat = self.W1(x)
        neighbours_hidden_feat = self.W2(inputs)
        global_feat_transformed = self.W3(global_features)
        return own_hidden_feat + neighbours_hidden_feat + global_feat_transformed

    def forward(self, x, edge_index, edge_attr, global_features, batch_id):
        # Every node should have a global feature which is the same for every node in each graph
        # The batch number is stores in batch_id
        global_features_per_node = global_features[batch_id]

        # Propagate Messages
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr, global_features=global_features_per_node)

        # Update global feature
        # Corresponds to rho but only takes all nodes into consideration and omits messages
        aggregated_nodes = self.pool(x, batch=batch_id)
        global_input = torch.cat([global_features, aggregated_nodes], dim=-1)
        global_features = self.W5(global_input)

        return x, global_features


class GNNGI(torch.nn.Module):
    """
    Graph Neural Network with global Information
    """
    def __init__(self, in_channels, edge_channels, global_channels, hidden_channels, out_channels, global_out_channels, num_layers, embedding_dim):
        """
        Create an instance of a Graph Neural Network with global Information sharing
        :param in_channels: dimension of node attributes before embedding
        :param edge_channels: dimension of edge attributes
        :param global_channels: dimension of global information vector, should be the same as in_channels+edge_channels
        :param hidden_channels: dimension of the message passed along the edges
        :param out_channels: dimension of the new node features
        :param global_out_channels: dimension of the new global feature
        :param num_layers: number of layers
        :param embedding_dim: dimension to use of node embedding
        """
        super(GNNGI, self).__init__()
        self.convolutions = ModuleList()

        assert num_layers > 0, "num_layers must be greater than 0"
        for _ in range(num_layers-1):
            self.convolutions.append(MPWGI(in_channels, edge_channels, global_channels, hidden_channels, out_channels, global_out_channels))  # in_channels is first the feature size of the nodes
            in_channels = out_channels  # now in_channels is the size of out_channels of the last layer
            global_channels = global_out_channels  # same for global_channels
        self.convolutions.append(MPWGI(in_channels, edge_channels, global_channels, hidden_channels, out_channels, global_out_channels))  # Last Layer

        self.emb = EmbedStations(num_stations_max=535, embedding_dim=embedding_dim)
        self.pool = global_mean_pool
        self.lin = Linear(out_channels,2)
        self.make_pos = MakePositive()

    def forward(self, x, edge_index, edge_attr, batch_id, global_feature=None):
        x = self.emb(x)  # embed station id

        if global_feature is None:
            global_feature = self.pool(x, batch=batch_id)  # Do batch-wise pooling to create initial global feature

        x, global_feature = self.convolutions[0](x, edge_index, edge_attr, global_feature, batch_id)
        x = F.relu(x)
        global_feature = F.relu(global_feature)

        if len(self.convolutions) > 1:
            for conv in self.convolutions[1:]:
                x_new, global_features_new = conv(x, edge_index, edge_attr, global_feature, batch_id)
                x = x + F.relu(x_new)  # Resnet
                global_feature = global_feature + F.relu(global_features_new)  # Resnet

        x = self.lin(x)

        mu_sigma = self.make_pos(x)
        return mu_sigma



def main():
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
    # last_obs is -366 since the last year is used for testing
    normalized_data = normalize_data(data, last_obs=-366, method="max")

    # Get List of stations with all stations -> will break further code if cut already
    stations = load_stations(data)

    # Create Dataset
    dist_matrix = np.load('dist_matrix.npy')
    corr_matrix = np.load('corr_matrix.npy')
    position_matrix = np.array(stations[['station', 'lon', 'lat']])

    torch_data = []
    dates = data['date'].unique()[:-366]  # last 366 days are used for testing
    for date in dates:
        torch_data.append(create_data(df=normalized_data,
                                      date=date,
                                      dist_matrix=dist_matrix if not config['data']['use_corr'] else corr_matrix,
                                      position_matrix=position_matrix,
                                      method=config['data']['method'],
                                      max_dist=config['data']['max_dist'],
                                      k=config['data']['nearest_k'],
                                      nearest_k_mode=config['data']['nearest_k_mode']))
    # Set device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Create Dataloader
    # Move all the data directly to the GPU (should fit into memory)
    torch_data_train = [tensor.to(device) for tensor in torch_data]

    # Definition of train_loader and valid_loader
    train_loader = DataLoader(torch_data_train, batch_size=config['model']['batch_size'], shuffle=True)

    # Model Creation
    # Number of Features
    num_features = torch_data[0].num_features
    emb_num_features = num_features + config['model']['embed_dim'] - 1

    model = create_model(config, emb_num_features)

    # Model to GPU
    model.to(device)
    # Compile model
    if config['model']['compile']:
        torch_geometric.compile(model)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['model']['lr'])

    # Train Loop
    train_losses = []

    for epoch in range(config['training']['n_epochs']):
        # Train for one epoch
        model.train()
        train_loss = 0.0
        # Train loop
        for batch in train_loader:
            loss = train(model, optimizer, batch)
            train_loss += loss.item() * batch.num_graphs
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

    # path
    save_path = os.path.dirname(args.config)
    if not os.path.exists(f"{save_path}/checkpoints"):
        os.makedirs(f"{save_path}/checkpoints")

    # Save model checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{save_path}/checkpoints/model_{args.id}.pt")

    print(f"Stopped {args.id}")


if __name__ == '__main__':
    main()
