import pandas as pd
import math
import os
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

def load_stations(df: pd.DataFrame = None) -> pd.DataFrame:
    stations = df.groupby(by='station')[['lat','lon','alt','orog']].first().reset_index()
    stations.station = pd.to_numeric(stations.station, downcast='integer')
    return stations