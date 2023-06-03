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
import pandas as pd
import torch
from typing import List


def  load_data(indexed:bool = True) -> pd.DataFrame:
    """
    Loads the data from the feather file and converts the station column to an integer.
    :return: Dataframe with the data.
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_RL18.feather'))
    df = pd.read_feather(path)
    df.station = pd.to_numeric(df.station, downcast='integer') - 1 # convert station to integer and subtract 1 to make it 0-based
    df = df.sort_values(by=['date','station']) # sort by date and station
    df["doy"] = df["date"].apply(lambda x: math.sin(((x.day_of_year-105)/366)*2*math.pi)) # Sin transformed day of year
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
    df = df.drop(df[df['alt'] > max_alt].index)
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


def create_data(df: pd.DataFrame, date: pd.Timestamp, mask: np.array, dist_matrix: np.array) -> Data:
    """
    Create a PyTorch Geometric Data object from a given date.

    Args:
        df (pd.DataFrame): dataframe with stations and observations
        date (pd.Timestamp): date to create data for
        mask (np.array): Boolean array that represents the edges with distance less than max_dist
        dist_matrix (np.array): distance matrix between stations


    Returns:
        Data: PyTorch Geometric Data object
    """
    # Get the rows of the dataframe corresponding to the current date
    df = df[df['date'] == date]

    # Get the stations that reported for the current date
    reporting_stations = df['station'].to_numpy()
    assert any(df.station.value_counts() != 1) == False # Any Station that reports df should appear only once
    assert np.all(reporting_stations[:-1] < reporting_stations[1:]) # Array of Stations should be in ascending order

    # Create feature tensor (stations are orderd in the df)
    node_features = df.drop(['date', 'obs'], axis=1).to_numpy()
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create target tensor
    target = df['obs'].to_numpy()
    y = torch.tensor(target, dtype=torch.float)

    # Create a pairwise distance matrix (omit non reporting stations)
    mesh = np.ix_(reporting_stations, reporting_stations)
    mask_sliced = mask[mesh]

    # Get the indices of the edges to include
    edges = np.argwhere(mask_sliced)
    edges = edges.T

    edge_index = torch.tensor(edges, dtype=torch.int64)
    assert edge_index.shape[0] == 2
    
    #Create edge_attr tensor
    dist_matrix_sliced = dist_matrix[mesh]
    edge_attr = dist_matrix_sliced[edges[0], edges[1]]
    edge_attr = edge_attr.reshape(-1, 1)
    edge_attr = torch.tensor(edge_attr)
    # Create a PyTorch Geometric Data object
    df = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
    return df


def visualize_graph(d: Data, df: pd.DataFrame, last_obs: int, rescaling_method: str = 'std'):
    """Visualize the Generated Data as a graph

    Args:
        d (Data): torch_geometric data object
        df (pd.DataFrame): dataframe used to construct the torch data
        last_obs (int): last observation used in nomalizing the data to rescale it here
    """
    G = to_networkx(d, to_undirected=True)
    pos =  d.x[:,-5:-3].detach().numpy() # TODO this does not work if new features are added (add dict featureToIndex)
    pos = np.transpose([pos[:, 1], pos[:, 0]]) # Switch latitude and longitude
    # Rescale lat and long
    if rescaling_method == 'std':
        fac_lat = df["lat"][:last_obs].std()
        fac_lon = df["lon"][:last_obs].std()
        mean_lat = df["lat"][:last_obs].mean()
        mean_lon = df["lon"][:last_obs].mean()
        pos = pos * np.array([fac_lon, fac_lat]) + np.array([mean_lon, mean_lat])
    else:
        fac_lat = df["lat"][:last_obs].max()
        fac_lon = df["lon"][:last_obs].max()
        pos = pos * np.array([fac_lon, fac_lat])

    dict_pos = {i: p.tolist() for i, p in enumerate(pos)}

    # Plot map
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.coastlines()
    ax.set_extent([5, 16, 47, 56], crs=proj)
    ax.add_feature(cfeature.LAND, color="lightgrey")
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN)

    # Colors
    color_mask = np.isin(d.edge_index.numpy().T, np.array(G.edges)).all(axis=1)
    color_values = d.edge_attr.flatten()[color_mask]

    cmap = mpl.colormaps.get_cmap('Greens_r')
    # Normalize the values to range between 0 and 1
    norm = plt.Normalize(min(color_values), max(color_values))
    # Generate a list of colors based on the normalized values
    colors = [cmap(norm(value)) for value in color_values]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax)

    # Plot Graph
    nx.draw_networkx(G, pos=dict_pos, node_size=10, node_color="black", ax=ax, with_labels=False, edge_color=colors)

    # Fix the aspect ratio of the map
    lat_center = (ax.get_extent()[2] + ax.get_extent()[3]) / 2
    ax.set_aspect(1 / np.cos(np.radians(lat_center)))

    ax.set_title("Active weather stations in Germany")

    plt.show()