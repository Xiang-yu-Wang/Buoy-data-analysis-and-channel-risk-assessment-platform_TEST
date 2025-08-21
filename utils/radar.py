import os
from re import I, S
from typing import List
import numpy as np
from numpy.typing import NDArray
from pandas import pandas
from streamlit_folium import st

from utils.helpers import DatasetCategory, StationDate, StationMetadata, get_data_path, list_station_dates

class Radar:
    def __init__(
            self,
            metadata: StationMetadata,
            resolution: float
        ):
        """
        Initializes a Radar object with the given parameters.
        :param name: Name of the radar
        :param latitude: Latitude of the radar location
        :param longitude: Longitude of the radar location
        :param resolution: Resolution of the radar in meters
        """
        self.id = metadata["StationID"]
        self.name = metadata["StationNameLocal"]
        self.metadata = metadata
        self.latitude = metadata["CenterLatitude"] 
        self.longitude = metadata["CenterLongitude"]
        self.resolution= resolution
        self.path = get_data_path(DatasetCategory.RADAR,self.id)

    def list_date(self) -> List[StationDate]:
        return list_station_dates(DatasetCategory.RADAR, self.path)

    def load_data(self, date: str) -> NDArray[np.float32]:
        """
        Loads radar data for a specific date.
        :param date: Date in the format 'YYYY-MM-DD'
        :return: A np array of radar data for the specified date (2D array)
        """
        return load_data(self.path, date)

    def prepare_data(self, path: str) -> NDArray[np.float32]:
        return prepare_data(path)


@st.cache_data
def load_data(station_path: str, date: str) -> NDArray[np.float32]:
    """
    Loads radar data for a specific date.
    :return: A np array of radar data for the specified date (2D array)
    """
    dates = list_station_dates(DatasetCategory.RADAR, station_path)

    path: str = ""
    for d in dates:
        if d["date"] != date: continue
        path = d["path"]
        break

    if not path:
        raise ValueError(f"Date {date} not found for radar ({path})")

    # Check if data not prepared
    npy_path = os.path.join(path, "prepared.npy")
    if not os.path.exists(npy_path):
        return prepare_data(path)

    return np.load(npy_path).astype(np.float32)

@st.cache_resource
def prepare_data(path: str) -> NDArray[np.float32]:
    """
    Prepares radar data from a specified path.
    :param path: Path to the data files
    :return: A message indicating that data has been prepared
    """
    # load all .csv files in the path
    # sort the files by filename
    files = sorted([f for f in os.listdir(path) if f.endswith('.txt')])
    if not files:
        raise ValueError(f"No TXT files found in path: {path}")

    # create a 2D np array to hold the data
    data = np.empty((0, 6), dtype=np.float32)

    # load each file
    # split by 5space (but negative will less 1 space)
    # the file have 6 columns
    for file in files:
        file_path = os.path.join(path, file)
        colspecs = [(0, 13), (13, 26), (26, 39), (39, 52), (52, 65), (65, None)]
        df = pandas.read_fwf(file_path, colspecs=colspecs, header=None)
        df = df.fillna(0.0).astype(np.float32)
        data = np.hstack((data, df.values)) if data.size else df.values


    # save as gray image with float32 2d array
    print(f"Preparing data for radar ({path})")
    np.save(os.path.join(path, "prepared.npy"), data)
    
    return data
