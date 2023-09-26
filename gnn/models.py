import sys
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import torch_geometric
import numpy as np
from typing import Tuple, Optional
from torch_geometric.nn import Sequential, MessagePassing, GATv2Conv, GraphSAGE
from torch_geometric.nn.pool import global_mean_pool
from torch.nn import Linear, Embedding, ModuleList, ReLU


def crps_no_avg(mu_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed df

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
    return crps_score


def crps(mu_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates the Continuous Ranked Probability Score (CRPS) assuming normally distributed df

    :param torch.Tensor mu_sigma: tensor of mean and standard deviation
    :param torch.Tensor y: observed df

    :return tensor: CRPS value
    :rtype torch.Tensor
    """
    crps_score = crps_no_avg(mu_sigma=mu_sigma, y=y)
    return torch.mean(crps_score)


class Convolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, heads, num_layers: int = None):
        super(Convolution, self).__init__()
        # Make sure either hidden_channels is a list, heads is a list or num_layer is supplied
        assert isinstance(hidden_channels, list) or isinstance(heads, list) or num_layers is not None, \
            "If hidden_channels is not a list, num_layers must be specified."
        # both are a list
        if isinstance(hidden_channels, list) and isinstance(heads, list):
            assert len(hidden_channels) == len(
                heads), f"Lengths of lists {len(hidden_channels)} and {len(heads)} do not match."
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
            self.convolutions.append(GATv2Conv(in_channels=-1, out_channels=c, heads=h))
        # Last Layer to match shape of output
        self.lin = Linear(in_features=hidden_channels[-1] * heads[-1], out_features=out_channels)
        self.relu = ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for conv in self.convolutions:
            x = self.relu(conv(x, edge_index, edge_attr))
        x = self.lin(x)
        return x


class EmbedStations(torch.nn.Module):
    def __init__(self, num_stations_max, embedding_dim):
        super(EmbedStations, self).__init__()
        self.embed = Embedding(num_embeddings=num_stations_max, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor):
        station_ids = x[:, 0].long()
        emb_station = self.embed(station_ids)
        x = torch.cat((emb_station, x[:, 1:]), dim=1)  # Concatenate embedded station_id to rest of the feature vector
        return x


class MakePositive(torch.nn.Module):
    def __init__(self):
        super(MakePositive, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, sigma = torch.split(x, 1, dim=1)
        sigma = F.softplus(sigma)  # ensure that sigma is positive
        mu_sigma = torch.cat([mu, sigma], dim=1)
        return mu_sigma


class UseGlobalContext(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(UseGlobalContext, self).__init__()
        self.aggr = global_mean_pool
        self.lin = Linear(2 * in_features, out_features)

    def forward(self, x: torch.Tensor, batch_id: torch.Tensor) -> torch.Tensor:
        global_features = self.aggr(x, batch_id)
        global_features_per_node = global_features[batch_id]
        input_vec = torch.cat([x, global_features_per_node], dim=-1)
        return self.lin(input_vec)


class ResGnn(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, hidden_channels: int, heads: int):
        super(ResGnn, self).__init__()
        assert num_layers > 0, "num_layers must be > 0."

        # Create Layers
        self.convolutions = ModuleList()
        for _ in range(num_layers):
            self.convolutions.append(GATv2Conv(-1, hidden_channels, heads=heads, edge_dim=1, add_self_loops=True,
                                               fill_value=0.01))
        self.lin = Linear(hidden_channels * heads, out_channels)
        self.relu = ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = x.float()
        edge_attr = edge_attr.float()
        for i, conv in enumerate(self.convolutions):
            if i == 0:
                # First Layer
                x = conv(x, edge_index, edge_attr)
                x = self.relu(x)
            else:
                x = x + self.relu(conv(x, edge_index, edge_attr))  # Residual Layers

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
        :return x, edge_index_attention, attention_weights: Tensor of Node Features (NxD), Tensor of Edges with
        self loops (2xE), Tensor of Attention per edge (ExNum_Heads)
        """
        x = x.float()
        edge_attr = edge_attr.float()

        # Pass Data though Layer to get the Attention
        attention_list = []
        # Note: edge_index_attention has to be added since we have self loops now
        edge_index_attention, attention_weights = None, None

        for i, conv in enumerate(self.convolutions, ):
            if i == 0:
                # First Layer
                x, (edge_index_attention, attention_weights) = conv(x, edge_index, edge_attr,
                                                                    return_attention_weights=True)
                attention_list.append(attention_weights)
                x = self.relu(x)
                x = self.norm(x)
            else:
                x_conv, (edge_index_attention, attention_weights) = conv(x, edge_index, edge_attr,
                                                                         return_attention_weights=True)
                attention_list.append(attention_weights)
                x = x + self.relu(x_conv)  # Residual Layers
        x = self.lin(x)

        # Attention weights of first layer
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
        self.relu = ReLU()

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

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor, global_features_i: torch.Tensor)\
            -> torch.Tensor:
        """
        Corresponds to psi.
        Constructs a message from node j to node i for each edge in edge_index.
        Tensors passed to propagate() can be mapped to the respective nodes i and j and by appending _i or _j
        :param x_i: features of node i
        :param x_j: features of node j
        :param edge_attr: feature form node i to node j
        :param global_features_i: global feature of the entire graph (here i is added since multiple graphs can be
        processed in parallel, then each of the graphs should have their own global attribute)
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
        update_input = torch.cat([inputs, x, global_features], dim=-1)
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
        self.emb = EmbedStations(num_stations_max=535, embedding_dim=embedding_dim)
        self.convolutions = ModuleList()

        assert num_layers > 0, "num_layers must be greater than 0"
        for _ in range(num_layers-1):
            self.convolutions.append(MPWGI(in_channels, edge_channels, global_channels, hidden_channels, out_channels, global_out_channels))  # in_channels is first the feature size of the nodes
            in_channels = out_channels  # now in_channels is the size of out_channels of the last layer
            global_channels = global_out_channels  # same for global_channels
        self.convolutions.append(MPWGI(in_channels, edge_channels, global_channels, hidden_channels, out_channels, global_out_channels))  # Last Layer

        self.pool = global_mean_pool
        self.lin = Linear(out_channels, 2)
        self.make_pos = MakePositive()
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch_id, global_feature=None):
        x = self.emb(x)  # embed station id

        if global_feature is None:
            global_feature = self.pool(x, batch=batch_id)  # Do batch-wise pooling to create initial global feature

        x, global_feature = self.convolutions[0](x, edge_index, edge_attr, global_feature, batch_id)
        x = self.relu(x)
        global_feature = self.relu(global_feature)

        if len(self.convolutions) > 1:
            for conv in self.convolutions[1:]:
                x_new, global_features_new = conv(x, edge_index, edge_attr, global_feature, batch_id)
                x = x + self.relu(x_new)  # Resnet
                global_feature = global_feature + self.relu(global_features_new)  #Resnet

        x = self.lin(x)

        mu_sigma = self.make_pos(x)
        return mu_sigma


def create_model(layer_type: str,
                 embed_dim: int,
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 lr: float,
                 heads: int = 0,
                 schedule_lr: bool = False,
                 compile_model: bool = False,
                 device: torch.cuda.device = "cuda")\
        -> Tuple[torch.nn.Module, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
    """
    Create a graph neural network model for node level regression.

    :param layer_type: type of layer used in the model
    :param embed_dim: The dimensionality of the node embeddings.
    :param in_channels: The number of input features for the node embeddings.
    :param hidden_channels: The number of hidden channels in the Convolution layers.
    :param num_layers: The number Convolution layers in the model.
    :param lr: The max learning rate for the optimizer.
    :param heads: number of heads used by the GAT.
    :param schedule_lr: Whether to use a learning rate scheduler. Defaults to False.
    :param compile_model: Whether to compile the model with torch_geometric. Defaults to False.
    :param device: Device to transfer the model to. Defaults to "cuda".

    :return: A tuple containing the created model, the optimizer
    and an optional learning rate scheduler if schedule_lr is True.
    :rtype: Tuple[torch.nn.Module, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]
    """
    # Clear Cache
    torch.cuda.empty_cache()

    if layer_type == 'GraphSAGE':
        mod = Sequential('x, edge_index, edge_attr, batch_id',
                         [
                             (EmbedStations(num_stations_max=535, embedding_dim=embed_dim), 'x -> x'),
                             (GraphSAGE(in_channels=in_channels,
                                        hidden_channels=hidden_channels,
                                        num_layers=num_layers,
                                        out_channels=2,
                                        project=True,
                                        aggr='mean'), 'x, edge_index -> x'),
                             (MakePositive(), 'x -> mu_sigma')
                         ])
    elif layer_type == 'ResGnn':
        mod = Sequential('x, edge_index, edge_attr, batch_id',
                         [
                             (EmbedStations(num_stations_max=535, embedding_dim=embed_dim), 'x -> x'),
                             (ResGnn(in_channels=in_channels,
                                     out_channels=2,
                                     hidden_channels=hidden_channels,
                                     heads=heads,
                                     num_layers=num_layers), 'x, edge_index, edge_attr -> x'),
                             (MakePositive(), 'x -> mu_sigma')
                         ])
    elif layer_type == 'GlobalInfo':
        mod = GNNGI(in_channels=in_channels,
                    edge_channels=1,
                    global_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,
                    global_out_channels=in_channels,
                    num_layers=num_layers,
                    embedding_dim=embed_dim)
    else:
        print(f"Invalid model architecture {layer_type}")
        sys.exit(1)

    mod.to(device)
    # Compile model
    if compile_model:
        torch_geometric.compile(mod)
    # Optimizer
    opt = torch.optim.AdamW(mod.parameters(), lr=lr)

    # Learning Rate Scheduler
    sched = None
    if schedule_lr:
        sched = ExponentialLR(opt, gamma=0.99)

    return mod, opt, sched
