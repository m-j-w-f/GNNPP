import pandas as pd
import os
def  load_data() -> pd.DataFrame:
    """
    Loads the data from the feather file and converts the station column to an integer.
    :return: Dataframe with the data.
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_RL18.feather'))
    df = pd.read_feather(path)
    df.station = pd.to_numeric(df.station, downcast='integer') # convert station to integer
    df.index = df.date # add DatetimeIndex
    df.index = df.index.tz_convert(None) # remove timezone
    return df

def load_stations(df: pd.DataFrame = None) -> pd.DataFrame:
    stations = df.groupby(by='station')[['lat','lon','alt','orog']].first().reset_index()
    stations.station = pd.to_numeric(stations.station, downcast='integer')
    return stations