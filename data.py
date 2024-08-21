from datetime import datetime
import numpy as np
import pandas as pd

from dmi_open_data import DMIOpenDataClient, Parameter

import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv('API_KEY')
#---------------------------------------------------------------
#---------------------------------------------------------------
def loadDataset():

    levelDsetName = "52.75_Vandstand_Minut.csv"
    tempDsetName = "52.75_Vandtemperatur_Minut.csv"

    level = pd.read_csv(levelDsetName, 
                        delimiter = ";", 
                        skiprows = 12,
                        encoding = "unicode-escape")
    
    level["Dato (DK normaltid)"] = pd.to_datetime(level["Dato (DK normaltid)"])
    x = level["Dato (DK normaltid)"]
    y = level["Vandstand (m DVR90)"]

    return x, y

#---------------------------------------------------------------
def get_train_test_data():
    level = pd.read_csv("52.75_Vandstand_Minut.csv",
                        delimiter = ";",
                        skiprows = 12,
                        encoding = "unicode-escape")

    temp = pd.read_csv("52.75_Vandtemperatur_Minut.csv",
                        delimiter = ";",
                        skiprows = 12,
                        encoding = "unicode-escape")

    # Train data from Jan 1 2019 to Dec 30 2019
    level["Dato (DK normaltid)"] = \
        pd.to_datetime(level["Dato (DK normaltid)"], dayfirst = True)
    start_date = "2019-01-01 07:00:00"
    end_date = "2019-12-30 07:00:00"
    mask = (level['Dato (DK normaltid)'] > start_date) & \
        (level['Dato (DK normaltid)'] <= end_date)
    train_data = level.loc[mask]
    train_data.set_index('Dato (DK normaltid)', inplace = True)
    train_data = train_data.resample('24H').mean()
    train_y = train_data["Vandstand (m DVR90)"].values
    train_x_orig = np.asarray(train_data.index, dtype = object)
    n = train_y.shape[0]
    train_x = np.arange(0, n, 1).astype(float)

    # Test data from Jan 1 2020 to Dec 30 2020
    start_date = "2020-01-01 07:00:00"
    end_date = "2020-03-30 07:00:00"
    mask = (level['Dato (DK normaltid)'] > start_date) & \
        (level['Dato (DK normaltid)'] <= end_date)
    test_data = level.loc[mask]
    test_data.set_index('Dato (DK normaltid)', inplace = True)
    test_data = test_data.resample('24H').mean()
    test_y = test_data["Vandstand (m DVR90)"].values
    test_x_orig = np.asarray(test_data.index, dtype = object)
    m = test_y.shape[0]
    test_x = np.arange(n, n + m, 1).astype(float)

    return train_x, train_y, test_x, test_y, train_x_orig, test_x_orig


#---------------------------------------------------------------
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

#---------------------------------------------------------------
def get_weather_data():
    '''
    Get Weather data - not tested!

    Get precipitation data from DMI
    '''
    # Get 10 stations
    client = DMIOpenDataClient(api_key)
    stations = client.get_stations(limit=10)

    # Get closest station
    closest_station = client.get_closest_station(
        latitude=55.40,
        longitude=12.34)

    # Get available parameters
    parameters = client.list_parameters()

    # Get temperature observations from DMI station in given time period
    observations = client.get_observations(
        parameter=Parameter.PrecipDurPast1h,
        station_id=closest_station['properties']['stationId'],
        from_time=datetime(2019, 1, 1),
        to_time=datetime(2019, 2, 4))
        #limit=300)

    n = len(observations)
    precip = [d["properties"]['value'] for d in observations]
    avg_prec = moving_average(precip, int(0.965 * n))

#---------------------------------------------------------------