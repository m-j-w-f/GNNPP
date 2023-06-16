from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from tqdm import tqdm, trange

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
from typing import List


def load_data(indexed:bool = True) -> pd.DataFrame:
    """
    Loads the data from the feather file and converts the station column to an integer.
    :return: Dataframe with the data.
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_RL18.feather'))
    df = pd.read_feather(path)
    df.station = pd.to_numeric(df.station, downcast='integer') - 1 # convert station to integer and subtract 1 to make it 0-based
    df = df.sort_values(by=['date', 'station'])  # sort by date and station
    df["doy"] = df["date"].apply(lambda x: math.sin(((x.day_of_year-105)/366)*2*math.pi))  # Sin transformed day of year
    if indexed:
        df.index = df.date # add DatetimeIndex
        df.index = df.index.tz_convert(None) # remove timezone
    return df


def load_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Create Stations Dataframe which only holds station specific data

    Args:
        df (pd.DataFrame): DataFrame as created by load_data

    Returns:
        pd.DataFrame: Station specific data
    """
    stations = df.groupby(by='station')[['lat','lon','alt','orog']].first().reset_index()
    stations.station = pd.to_numeric(stations.station, downcast='integer')
    return stations


def clean_data(df: pd.DataFrame, max_missing: int = 121, max_alt: float = 1000.0) -> pd.DataFrame:
    """Cleans the DataFrame from outliers and stations that have a lot of missig values

    Args:
        max_missing (int, optional): max number of rows with missing values. Defaults to 120.
        max_alt (float, optional): max altitude of station. Defaults to 1000.

    Returns:
        pd.DataFrame: Cleaned Dataframe
    """
    
    # drop stations with altitude > max_alt
    df = df[df['alt'] < max_alt]
    # drop stations with more than max_missing missing values completely
    stations_missing_data = df.station[df.sm_mean.isna()].to_numpy()
    stations_missing_data, counts = np.unique(stations_missing_data, return_counts=True)
    stations_to_drop  = stations_missing_data[counts > max_missing]
    df = df[~df['station'].isin(stations_to_drop)]
    # drop all rows with missing values
    df = df.dropna()
    return df


def normalize_data(df: pd.DataFrame, last_obs: int, method: str=None) -> pd.DataFrame:
    # normalize the DataFrame, excluding the columns to exclude
    exclude_cols = ['date', 'obs', 'station','doy']
    normalized_data = df.copy()  # create a copy of the original DataFrame to avoid modifying it directly
    if method == None or method == 'normal':
        for col in df.columns:
            if col not in exclude_cols:
                normalized_data[col] = (df[col] - df[col][:last_obs].mean()) / df[col][:last_obs].std()
    
    if method == 'max':
        for col in df.columns:
            if col not in exclude_cols:
                normalized_data[col] = df[col] / df[col][:last_obs].max()
    
    return normalized_data


def dist_km(lat1, lon1, lat2, lon2):
    """Returns distance between two stations in km using the the WGS-84 ellipsoid.

    Args:
        lat1 (float): latitude of first station
        lat2 (float): latitude of second station
        lon1 (float): longitude of first station
        lon2 (float): longitude of second station

    Returns:
        float: distance in km
    """
    return geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).km


def compute_dist_matrix(df: pd.DataFrame) -> np.array:
    """Returns a distance matrix between stations.

    Args:
        df (pd.DataFrame): dataframe with stations

    Returns:
        np.array: distance matrix
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


def get_mask(dist_matrix_sliced: np.array, method: str = "max_dist", k: int=3, max_dist: int=50):
    """
    Generate mask which specifies which edges to include in the graph
    :param dist_matrix_sliced: distance matrix with only the reporting stations
    :param method: method to compute included edges. max_dist includes all edges wich are shorter than max_dist km,
    k_nearest includes the k nearest edges for each station. So the out_degree of each station is k, the in_degree can vary.
    :param k: number of connections per node
    :param max_dist: maximum lengh of edges
    :return: return boolean mask of which edges to include
    """
    mask = None
    if method == "max_dist":
        mask = (dist_matrix_sliced <= max_dist) & (dist_matrix_sliced != 0)
    elif method == 'nearest_k':
        nearest_indices = np.argsort(dist_matrix_sliced, axis=1)[:, 1:+k]
        # Create an empty boolean array with the same shape as distances
        nearest_k = np.zeros_like(dist_matrix_sliced, dtype=bool)
        # Set the corresponding indices in the nearest_k array to True
        row_indices = np.arange(dist_matrix_sliced.shape[0])[:, np.newaxis]
        nearest_k[row_indices, nearest_indices] = True
        mask = nearest_k

    return mask


def create_data(df: pd.DataFrame, date: pd.Timestamp, dist_matrix: np.array, **kwargs) -> Data:
    """
    Create a PyTorch Geometric Data object from a given date.

    Args:
        df (pd.DataFrame): dataframe with stations and observations
        date (pd.Timestamp): date to create data for
        dist_matrix (np.array): distance matrix between stations
        **kwargs: Arguments are passed to geet_mask. These arguments include method (can be max_dist or k_nearest),
        k, max_dist


    Returns:
        Data: PyTorch Geometric Data object
    """
    # Get the rows of the dataframe corresponding to the current date
    df = df[df['date'] == date]

    # Get the stations that reported for the current date
    reporting_stations = df['station'].to_numpy()
    assert any(df.station.value_counts() != 1) is False  # Any Station that reports df should appear only once
    assert np.all(reporting_stations[:-1] < reporting_stations[1:])  # Array of Stations should be in ascending order

    # Create feature tensor (stations are orderd in the df)
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
    # Create a PyTorch Geometric Data object
    df = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    return df


def plot_map():
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.coastlines()
    ax.set_extent([5, 16, 47, 56], crs=proj)
    ax.add_feature(cfeature.LAND, color="lightgrey")
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN)
    return ax


def visualize_graph(d: Data, stations: pd.DataFrame):
    """Visualize the Generated Data as a graph

    Args:
        d (Data): torch_geometric data object
        stations (pd.DataFrame): dataframe used to construct the torch data
    """
    if d.is_directed():
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    # Holds the Station_ids like in stations DataFrame
    station_ids = np.array(d.x[..., 0].cpu(), dtype=int)

    # to cpu for further calculations and plotting
    edge_index, dist = d.edge_index.cpu().numpy(), d.edge_attr.cpu().numpy()

    # NOTE: edge_index_att holds the Edges of the new graph, however they are labeled consecutively instead of the ordering from stations DataFrame
    edge_index = station_ids[edge_index]  # now the same indexes as in the stations Dataframe are used
    # Filter latitude and longitude using station IDs
    stations_cut = stations[stations['station'].isin(station_ids)]

    # Add nodes (stations) to the graph
    for station in stations_cut.itertuples():
        G.add_node(station.station, lat=station.lat, lon=station.lon)  # Add station with ID, LAT and LON

    pos = {node: (data['lon'], data['lat']) for node, data in
           G.nodes(data=True)}  # Create A Positions Dict for every node

    # Add edges with edge_length as an attribute
    if d.is_directed():
        for edge, a in zip(edge_index.T.tolist(), dist.flatten().tolist()):  # Add all edges
            G.add_edge(edge[0], edge[1], length=a)  # Add all Edges with distance Attribute
    else:
        for edge, a in zip(edge_index.T.tolist(), dist.flatten().tolist()):
            if not (G.has_edge(edge[0], edge[1]) or G.has_edge(edge[1], edge[0])): # Edge only needs to be added once
                G.add_edge(edge[0], edge[1], length=a)  # Add all Edges with distance Attribute

    # Colors
    degrees = G.degree() if d.is_undirected() else G.in_degree()

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
    nx.draw_networkx(G, pos=pos, node_size=20, node_color=colors_n, ax=ax, with_labels=False, edge_color=colors, edgecolors="black")

    # Fix the aspect ratio of the map
    lat_center = (ax.get_extent()[2] + ax.get_extent()[3]) / 2
    ax.set_aspect(1 / np.cos(np.radians(lat_center)))

    ax.set_title("Active weather stations in Germany")

    plt.show()


def visualize_attention(test_sample: torch_geometric.data.data.Data, device: torch.device, model, stations: pd.DataFrame, min_attention: float=0.0):
    """
    Visualize the graph and color Edges based on attention of the GATConv Layer
    :param torch_geometric.data.data.Data test_sample: The sample to plot
    :param torch.device device: device to run the forward pass on
    :param model: The model with the GATConv Layer
    :param pd. Dataframe stations: DataFrame of all stations
    :param float min_attention: minimum attention to plot the edge
    """
    test_sample = test_sample.to(device)
    test_station_ids = np.array(test_sample.x[..., 0].cpu(),
                                dtype=int)  # Holds the Station_ids like in stations DataFrame

    test_x = test_sample.x
    test_edge_index = test_sample.edge_index
    test_edge_attr = test_sample.edge_attr

    # Pass data through Layers before Convolution
    with torch.no_grad():
        test_x = model[0](test_x)
        test_edge_attr = model[1](test_edge_attr)

    # Get ResGnn Layer and retrieve the attention weights
    resgnn_layer = model[2]
    x, edge_index_att, att = resgnn_layer.get_attention(x=test_x,
                                                        edge_index=test_sample.edge_index,
                                                        edge_attr=test_edge_attr)

    # to cpu for further calculations and plotting
    edge_index_att, att = edge_index_att.cpu().numpy(), att.cpu().numpy()

    # NOTE: edge_index_att holds the Edges of the new graph, however they are labeled consecutively instead of the ordering from stations DataFrame
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
    pos = {node: (data['lon'], data['lat']) for node, data in G.nodes(data=True)} # Create A Positions Dict for every node

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
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 10))
    ax = plot_map()
    # Draw the graph using networkx
    nx.draw_networkx(G, pos=pos, with_labels=False, node_color=colors_node, node_size=20, edge_color=colors, edgecolors="black", ax=ax)
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
