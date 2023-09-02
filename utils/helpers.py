from torch_geometric.data import Data

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopy.distance
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import torch
import torch_geometric


def load_data(indexed: bool = True) -> pd.DataFrame:
    """
    Load the data from the specified file and preprocess it.

    :param indexed: Whether to add a DateTimeIndex to the DataFrame. Defaults to True.
    :type indexed: bool, optional
    :return: The preprocessed DataFrame.
    :rtype: pd.DataFrame
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_RL18.feather'))
    df = pd.read_feather(path)
    # convert station to integer and subtract 1 to make it 0-based
    df.station = pd.to_numeric(df.station, downcast='integer') - 1
    df = df.sort_values(by=['date', 'station'])  # sort by date and station
    df["doy"] = df["date"].apply(lambda x: math.sin(((x.day_of_year-105)/366)*2*math.pi))  # Sin transformed day of year
    if indexed:
        df.index = df.date  # add DatetimeIndex
        df.index = df.index.tz_convert(None)  # remove timezone
    return df


def load_stations(df: pd.DataFrame) -> pd.DataFrame:
    """
     Create a DataFrame containing station-specific data from the input DataFrame.

     :param df: The DataFrame created by load_data.
     :type df: pd.DataFrame
     :return: The DataFrame containing station-specific data.
     :rtype: pd.DataFrame
     """
    stations = df.groupby(by='station')[['lat', 'lon', 'alt', 'orog']].first().reset_index()
    stations.station = pd.to_numeric(stations.station, downcast='integer')
    return stations


def clean_data(df: pd.DataFrame, max_missing: int = 121, max_alt: float = 1000.0) -> pd.DataFrame:
    """
    Cleans the DataFrame by removing outliers and stations with a high number of missing values.

    :param df: The DataFrame to be cleaned.
    :type df: pd.DataFrame
    :param max_missing: The maximum number of rows with missing values allowed for each station. Defaults to 121.
    :type max_missing: int, optional
    :param max_alt: The maximum altitude of stations to keep. Stations with altitudes above this value will be dropped. Defaults to 1000.0.
    :type max_alt: float, optional
    :return: The cleaned DataFrame.
    :rtype: pd.DataFrame
    """
    
    # drop stations with altitude > max_alt
    df = df[df['alt'] < max_alt]
    # drop stations with more than max_missing missing values completely
    stations_missing_data = df.station[df.sm_mean.isna()].to_numpy()
    stations_missing_data, counts = np.unique(stations_missing_data, return_counts=True)
    stations_to_drop = stations_missing_data[counts > max_missing]
    df = df[~df['station'].isin(stations_to_drop)]
    # drop all rows with missing values
    df = df.dropna()
    return df


def normalize_data(df: pd.DataFrame, last_obs: int, method: str = None) -> pd.DataFrame:
    """
    Normalize the data in a DataFrame, excluding date, obs, station and doy.

    :param df: The DataFrame containing the data to be normalized.
    :type df: pd.DataFrame
    :param last_obs: The index of the last observation to use for normalization.
    :type last_obs: int
    :param method: The normalization method to use. Available options are 'normal' (default) and 'max'.
    If None or 'normal', the data will be normalized by subtracting the mean and dividing by the standard deviation.
    If 'max', the data will be normalized by dividing by the maximum value.
    :type method: str, optional
    :return: The normalized DataFrame.
    :rtype: pd.DataFrame
    """
    # normalize the DataFrame, excluding the columns to exclude
    exclude_cols = ['date', 'obs', 'station', 'doy']
    normalized_data = df.copy()  # create a copy of the original DataFrame to avoid modifying it directly
    if method is None or method == 'normal':
        for col in df.columns:
            if col not in exclude_cols:
                normalized_data[col] = (df[col] - df[col][:last_obs].mean()) / df[col][:last_obs].std()
    
    if method == 'max':
        for col in df.columns:
            if col not in exclude_cols:
                normalized_data[col] = df[col] / df[col][:last_obs].max()
    
    return normalized_data


def dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Returns the distance between two stations in kilometers using the WGS-84 ellipsoid.

    :param lat1: Latitude of the first station.
    :type lat1: float
    :param lat2: Latitude of the second station.
    :type lat2: float
    :param lon1: Longitude of the first station.
    :type lon1: float
    :param lon2: Longitude of the second station.
    :type lon2: float

    :return: The distance between the two stations in kilometers.
    :rtype: float
    """
    return geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).km


def compute_dist_matrix(df: pd.DataFrame) -> np.array:
    """Returns a distance matrix between stations.

    :param df: dataframe with stations

    :return: distance matrix
    :rtype: np.array
    """
    coords_df = df[['lat', 'lon']].copy()

    # create numpy arrays for latitudes and longitudes
    latitudes = np.array(coords_df['lat'])
    longitudes = np.array(coords_df['lon'])

    # create a meshgrid of latitudes and longitudes
    lat_mesh, lon_mesh = np.meshgrid(latitudes, longitudes)

    # calculate distance matrix using vectorized distance function
    distance_matrix = np.vectorize(dist_km)(lat_mesh, lon_mesh, lat_mesh.T, lon_mesh.T)
    return distance_matrix


def get_mask(dist_matrix_sliced: np.array,
             method: str = "max_dist",
             k: int = 3,
             max_dist: int = 50,
             nearest_k_mode: str = "in") -> np.array:
    """
    Generate mask which specifies which edges to include in the graph
    :param dist_matrix_sliced: distance matrix with only the reporting stations
    :param method: method to compute included edges. max_dist includes all edges wich are shorter than max_dist km,
    k_nearest includes the k nearest edges for each station. So the out_degree of each station is k,
    the in_degree can vary.
    :param k: number of connections per node
    :param max_dist: maximum length of edges
    :param nearest_k_mode: "in" or "out". If "in" the every node has k nodes passing information to it,
    if "out" every node passes information to k nodes.
    :return: return boolean mask of which edges to include
    :rtype: np.array
    """
    mask = None
    if method == "max_dist":
        mask = (dist_matrix_sliced <= max_dist) & (dist_matrix_sliced != 0)
    elif method == 'nearest_k':
        nearest_indices = np.argsort(dist_matrix_sliced, axis=1)[:, 1:k+1]
        # Create an empty boolean array with the same shape as distances
        nearest_k = np.zeros_like(dist_matrix_sliced, dtype=bool)
        # Set the corresponding indices in the nearest_k array to True
        row_indices = np.arange(dist_matrix_sliced.shape[0])[:, np.newaxis]
        nearest_k[row_indices, nearest_indices] = True
        if nearest_k_mode == "in":
            mask = nearest_k.T
        elif nearest_k_mode == "out":
            mask = nearest_k

    return mask


def create_data(df: pd.DataFrame, date: pd.Timestamp, dist_matrix: np.array, position_matrix: np.array, **kwargs) -> Data:
    """
    Create a PyTorch Geometric Data object from the provided DataFrame, distance matrix, and position matrix.

    :param df: The DataFrame containing the data.
    :type df: pd.DataFrame
    :param date: The specific date for which to create the data.
    :type date: pd.Timestamp
    :param dist_matrix: The pairwise distance matrix.
    :type dist_matrix: np.array
    :param position_matrix: The position matrix.
    :type position_matrix: np.array
    :param kwargs: Additional keyword arguments for get_mask function.
    Arguments are passed to geet_mask. These arguments include method (can be max_dist or k_nearest), with max_dist or k
    accordingly specified. Also for k_nearest the method nearest_k_mode "in" or "out" must be specified

    :return: The PyTorch Geometric Data object.
    :rtype: torch_geometric.data.Data
    """

    # Get the rows of the dataframe corresponding to the current date
    df = df[df['date'] == date]

    # Get the stations that reported for the current date
    reporting_stations = df['station'].to_numpy()
    assert any(df.station.value_counts() != 1) is False  # Any Station that reports df should appear only once
    assert np.all(reporting_stations[:-1] < reporting_stations[1:])  # Array of Stations should be in ascending order

    # Create feature tensor (stations are ordered in the df)
    node_features = df.drop(['date', 'obs'], axis=1).to_numpy()
    x = torch.tensor(node_features, dtype=torch.float)

    # Create target tensor
    target = df['obs'].to_numpy()
    y = torch.tensor(target, dtype=torch.float)

    # Create a pairwise distance matrix (omit non reporting stations)
    mesh = np.ix_(reporting_stations, reporting_stations)
    dist_matrix_sliced = dist_matrix[mesh]
    mask = get_mask(dist_matrix_sliced=dist_matrix_sliced, **kwargs)
    mask_sliced = mask  # already sliced

    # Get the indices of the edges to include
    edges = np.argwhere(mask_sliced)
    edges = edges.T

    edge_index = torch.tensor(edges, dtype=torch.int64)
    assert edge_index.shape[0] == 2

    # Create edge_attr tensor
    dist_matrix_sliced = dist_matrix[mesh]
    edge_attr = dist_matrix_sliced[edges[0], edges[1]]
    edge_attr = edge_attr.reshape(-1, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    # normalize
    max_len = torch.max(edge_attr)
    standardized_edge_attr = edge_attr / max_len

    # Additional attributes
    # Position
    station_indices = np.isin(position_matrix[:, 0], reporting_stations)  # get position of reporting stations
    pos = position_matrix[station_indices][:, -2:]
    pos = torch.tensor(pos, dtype=torch.float)
    # Actual length of edges
    dist = edge_attr.flatten()

    # Create a PyTorch Geometric Data object
    df = Data(x=x, y=y, edge_index=edge_index, edge_attr=standardized_edge_attr, distances=edge_attr, pos=pos)
    return df


def plot_map() -> plt.Axes:
    """
    Plot a map using PlateCarree projection.

    :return: The Axes object containing the map.
    :rtype: matplotlib.axes.Axes
    """
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.coastlines()
    ax.set_extent([5, 16, 47, 56], crs=proj)
    ax.add_feature(cfeature.LAND, color="lightgrey")
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN)
    return ax


def visualize_graph(d: Data) -> None:
    """
        Visualize the Generated Data as a graph.

    :param d: torch_geometric data object
    :type d: Data
    """
    if d.is_directed():
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # to cpu for further calculations and plotting
    edge_index, dist = d.edge_index.cpu().numpy(), d.distances.cpu().numpy()

    # NOTE: edge_index_att holds the Edges of the new graph,
    # however they are labeled consecutively instead of the ordering from stations DataFrame
    station_ids = np.array(d.x[:, 0])
    edge_index = station_ids[edge_index]  # now the same indexes as in the stations Dataframe are used

    # Add nodes (stations) to the graph
    for i in range(d.num_nodes):
        G.add_node(int(d.x[i,0]), lon=float(d.pos[i,0]), lat=float(d.pos[i,1]))  # Add station with ID, LAT and LON

    pos = {node: (data['lon'], data['lat']) for node, data in
           G.nodes(data=True)}  # Create a positions dict

    # Add edges with edge_length as an attribute
    if d.is_directed():
        for edge, a in zip(edge_index.T.tolist(), dist.flatten().tolist()):  # Add all edges
            G.add_edge(edge[0], edge[1], length=a)  # Add all Edges with distance Attribute
    else:
        for edge, a in zip(edge_index.T.tolist(), dist.flatten().tolist()):
            if not (G.has_edge(edge[0], edge[1]) or G.has_edge(edge[1], edge[0])):  # Edge only needs to be added once
                G.add_edge(edge[0], edge[1], length=a)  # Add all Edges with distance Attribute

    # Colors
    degrees = G.degree if d.is_undirected() else G.in_degree

    node_colors = [deg for _, deg in degrees]
    cmap_nodes = plt.get_cmap('jet', max(node_colors) - min(node_colors) + 1)
    norm = plt.Normalize(min(node_colors), max(node_colors))
    colors_n = [cmap_nodes(norm(value)) for value in node_colors]
    sm_nodes = plt.cm.ScalarMappable(cmap=cmap_nodes, norm=norm)

    # Edge Colors
    color_values = [attr['length'] for _, _, attr in G.edges(data=True)]
    cmap = mpl.colormaps.get_cmap('Blues_r')
    # Normalize the values to range between 0 and 1
    norm = plt.Normalize(min(color_values), max(color_values))
    # Generate a list of colors based on the normalized values
    colors = [cmap(norm(value)) for value in color_values]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Plot map
    ax = plot_map()

    # Colorbar Node degrees
    if not all(node_colors[0] == col for col in node_colors[1:]):  # only add colorbar if there are different degrees
        colorbar = plt.colorbar(sm_nodes, ax=ax)
        ticks_pos = np.linspace(min(node_colors) + 1, max(node_colors), max(node_colors) - min(node_colors) + 1) - 0.5
        colorbar.set_ticks(ticks_pos)
        ticks = np.arange(min(node_colors), max(node_colors) + 1)
        colorbar.set_ticklabels(ticks)
        colorbar.ax.set_ylabel(f'Node{"_in" if d.is_directed() else ""} Degree', rotation=270, labelpad=20)
    # Colormap for Edges
    colorbar_e = plt.colorbar(sm, ax=ax)
    colorbar_e.ax.set_ylabel('Distance in km', rotation=270, labelpad=20)

    # Plot Graph
    nx.draw_networkx(G,
                     pos=pos,
                     node_size=20,
                     node_color=colors_n,
                     ax=ax,
                     with_labels=False,
                     edge_color=colors,
                     edgecolors="black")

    # Fix the aspect ratio of the map
    lat_center = (ax.get_extent()[2] + ax.get_extent()[3]) / 2
    ax.set_aspect(1 / np.cos(np.radians(lat_center)))

    ax.set_title("Active weather stations in Germany")
    plt.savefig("stations.eps", format="eps")
    plt.show()


def visualize_explanation(subgraph: Data, fullgraph: Data, stations: pd.DataFrame) -> None:
    """
    Visualize the explanation provided by an Explainer.

    :param subgraph: torch_geometric data object produced by explanation.get_explanation_subgraph()
    :type subgraph: Data
    :param fullgraph: the entire graph which is given to the explainer
    :type fullgraph: Data
    :param stations: dataframe used to construct the torch data
    :type stations: pd.DataFrame
    """
    if subgraph.is_directed():
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    # Holds the Station_ids like in stations DataFrame
    station_ids = np.array(subgraph.x[..., 0].cpu(), dtype=int)

    # to cpu for further calculations and plotting
    edge_index, dist = subgraph.edge_index.cpu().numpy(), subgraph.edge_mask.cpu().numpy()

    # NOTE: edge_index_att holds the Edges of the new graph,
    # however they are labeled consecutively instead of the ordering from stations DataFrame
    edge_index = station_ids[edge_index]  # now the same indexes as in the stations Dataframe are used
    if not isinstance(subgraph.index, list): subgraph.index = [subgraph.index]
    full_station_ids = np.array(fullgraph.x[..., 0].cpu(), dtype=int)
    exp_node_id = full_station_ids[subgraph.index]
    # Filter latitude and longitude using station IDs
    stations_cut = stations[stations['station'].isin(station_ids)]

    # Add nodes (stations) to the graph
    for station in stations_cut.itertuples():
        G.add_node(station.station, lat=station.lat, lon=station.lon)  # Add station with ID, LAT and LON

    pos = {node: (data['lon'], data['lat']) for node, data in
           G.nodes(data=True)}  # Create A Positions Dict for every node

    # Add edges with edge_length as an attribute
    if subgraph.is_directed():
        for edge, a in zip(edge_index.T.tolist(), dist.flatten().tolist()):  # Add all edges
            G.add_edge(edge[0], edge[1], length=a)  # Add all Edges with distance Attribute
    else:
        for edge, a in zip(edge_index.T.tolist(), dist.flatten().tolist()):
            if not (G.has_edge(edge[0], edge[1]) or G.has_edge(edge[1], edge[0])): # Edge only needs to be added once
                G.add_edge(edge[0], edge[1], length=a)  # Add all Edges with distance Attribute

    # Colors
    degrees = G.degree if subgraph.is_undirected() else G.in_degree

    # Node Colors
    node_colors = ["#00FF00" if node_id in exp_node_id else "black" for node_id in  G.nodes]

    # Edge Colors
    color_values = [attr['length'] for _, _, attr in G.edges(data=True)]
    cmap = mpl.colormaps.get_cmap('Purples')
    # Normalize the values to range between 0 and 1
    norm = plt.Normalize(min(color_values), max(color_values))
    # Generate a list of colors based on the normalized values
    colors = [cmap(norm(value)) for value in color_values]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Plot map
    ax = plot_map()

    # Colormap for Edges
    colorbar_e = plt.colorbar(sm, ax=ax)
    colorbar_e.ax.set_ylabel('Influence', rotation=270, labelpad=20)

    # Plot Graph
    nx.draw_networkx(G,
                     pos=pos,
                     node_size=20,
                     node_color=node_colors,
                     ax=ax,
                     with_labels=False,
                     edge_color=colors,
                     edgecolors="black")

    # Fix the aspect ratio of the map
    lat_center = (ax.get_extent()[2] + ax.get_extent()[3]) / 2
    ax.set_aspect(1 / np.cos(np.radians(lat_center)))

    ax.set_title(f"Explainability")

    plt.show()


def visualize_attention(test_sample: Data,
                        device: torch.device, model,
                        stations: pd.DataFrame,
                        min_attention: float = 0.0) -> None:
    """
    Visualize the graph and color Edges based on attention of the GATConv Layer.

    :param test_sample: The sample to plot.
    :type test_sample: torch_geometric.data.data.Data
    :param device: Device to run the forward pass on.
    :type device: torch.device
    :param model: The model with the GATConv Layer.
    :type model: Any
    :param stations: DataFrame of all stations.
    :type stations: pd.DataFrame
    :param min_attention: Minimum attention to plot the edge.
    :type min_attention: float
    """
    test_sample = test_sample.to(device)
    test_station_ids = np.array(test_sample.x[..., 0].cpu(),
                                dtype=int)  # Holds the Station_ids like in stations DataFrame

    test_x = test_sample.x
    test_edge_index = test_sample.edge_index
    test_edge_attr = test_sample.edge_attr

    # Pass data through Layers before Convolution
    with torch.no_grad():
        test_x = model[0](test_x)  # Embedding

    # Get ResGnn Layer and retrieve the attention weights
    resgnn_layer = model[1]
    x, edge_index_att, att = resgnn_layer.get_attention(x=test_x,
                                                        edge_index=test_sample.edge_index,
                                                        edge_attr=test_edge_attr)

    # to cpu for further calculations and plotting
    edge_index_att, att = edge_index_att.cpu().numpy(), att.cpu().numpy()

    # NOTE: edge_index_att holds the Edges of the new graph,
    # however they are labeled consecutively instead of the ordering from stations DataFrame
    edge_index_att = test_station_ids[edge_index_att]  # now the same indexes as in the stations Dataframe are used
    # Filter latitude and longitude using station IDs
    stations_cut = stations[stations['station'].isin(test_station_ids)]
    # ========================================================================================================== #
    # Create a graph using networkx
    G = nx.DiGraph()

    # Add nodes (stations) to the graph
    for station in stations_cut.itertuples():
        G.add_node(station.station, lat=station.lat, lon=station.lon)  # Add station with ID, LAT and LON

    # Add edges with edge_attention as an attribute
    for edge, a in zip(edge_index_att.T.tolist(), att.flatten().tolist()):
        G.add_edge(edge[0], edge[1], attention=a)  # Add all Edges with attention attribute

    # Extract coordinates from nodes' attributes
    # Create A Positions Dict for every node
    pos = {node: (data['lon'], data['lat']) for node, data in G.nodes(data=True)}

    # Remove self edges and add self_attention to the nodes' attributes
    for node, _, d in nx.selfloop_edges(G, data=True):
        G.nodes[node]['self_attention'] = d["attention"]
    G.remove_edges_from(nx.selfloop_edges(G))

    # Create a list of self Attention to color the Nodel later
    nodes_att = [d['self_attention'] for _, d in G.nodes(data=True)]

    # Remove all edges with an attention less than 0.1
    # Iterate over all edges and store the ones that need to be removed
    edges_to_remove = []
    for u, v, data in G.edges(data=True):
        if data["attention"] < min_attention:
            edges_to_remove.append((u, v))
    # Remove the identified edges from the graph
    G.remove_edges_from(edges_to_remove)

    # Extract edge attentions from edges' attributes
    edge_att = nx.get_edge_attributes(G, 'attention').values()

    # Define Color for the Edges and Nodes
    cmap = mpl.colormaps.get_cmap('Reds')
    # Normalize the values to range between 0 and 1
    norm = plt.Normalize(min([min(edge_att), min(nodes_att)]),
                         max([max(edge_att), max(nodes_att)]))
    # Generate a list of colors based on the normalized values
    colors = [cmap(norm(value)) for value in edge_att]
    colors_node = [cmap(norm(value)) for value in nodes_att]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # ========================================================================================================== #
    # Plot map
    ax = plot_map()
    # Draw the graph using networkx
    nx.draw_networkx(G,
                     pos=pos,
                     with_labels=False,
                     node_color=colors_node,
                     node_size=20,
                     edge_color=colors,
                     edgecolors="black",
                     ax=ax)
    # Add colorbar
    plt.colorbar(sm, ax=ax)
    # Add labels
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    # Fix the aspect ratio of the map
    lat_center = (ax.get_extent()[2] + ax.get_extent()[3]) / 2
    ax.set_aspect(1 / np.cos(np.radians(lat_center)))
    # Show the plot
    plt.show()
