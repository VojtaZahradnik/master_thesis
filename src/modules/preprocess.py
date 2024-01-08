import glob
import math
import os
from feature_engine.timeseries.forecasting import WindowFeatures
from scipy.ndimage import uniform_filter1d
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import MathFeatures
from gpx_converter import Converter
import haversine as hs
import warnings
from datetime import datetime, timedelta
from typing import Any, Union

import fitparse
import numpy as np
import pandas as pd
from meteostat import Point, Hourly
from src.modules import conf, log
from pandas import DataFrame
from tqdm import tqdm

"""
Load values from fit files and add some new variables from external libraries
"""

def get_meteo(lat: str, long: str, alt: str, day: str) -> Union[list[Any], DataFrame]:
    """
    Retrieves meteorological data for a specified location and time using the Meteostat library.

    Args:
        lat (str): Latitude of the start location.
        long (str): Longitude of the start location.
        alt (str): Altitude of the start location.
        day (str): Date and time of the activity start.

    Returns:
        Union[list[Any], DataFrame]: Weather information about the start location,
        or basic values when Meteostat does not have data for the location.
    """
    lst = [15, 0, 0, 0, 0]
    if lat is not None and long is not None:

        transfer_num = 11930465
        if lat > 100:
            loc = Point(lat/transfer_num, long/transfer_num, int(alt))
        else:
            loc = Point(lat, long, int(alt))

        day = datetime(
            day.year,
            day.month,
            day.day,
            day.hour,
        )

        data = Hourly(loc, day - timedelta(hours=2), day + timedelta(hours=2))
        data = data.fetch()
        if len(data != 0):
            lst = [
                data["temp"][0],
                data["rhum"][0],
                data["prcp"][0],
                data["snow"][0],
                data["wspd"][0],
            ]
        else:
            print(f"WARN: {day} has basic weather values")

        lst = [x if not math.isnan(x) else 0 for x in lst]
    return lst


def calc_slope_steep(df: pd.DataFrame, threshold=45) -> list[int]:
    """
    Calculates the slope steepness for each row in a dataframe.

    Args:
       df (pd.DataFrame): DataFrame containing activity data.
       threshold (int, optional): Threshold for steepness to push a bike or walk in running. Defaults to 45.

    Returns:
       list[int]: A list representing the slope steepness for each data point.
    """
    slope_steep = [0] * len(df)
    for i in range(1, len(df)):
        dist = df["distance"][i] - df["distance"][i - 1]
        elev = np.max([df["slope_descent"][i], df["slope_ascent"][i]])
        tmp = - 1 if df["slope_descent"][i] > df["slope_ascent"][i] else 1
        if dist != 0:
            steep = tmp* (elev / dist) * 100
            if steep > threshold:
                steep = threshold
            elif steep < -threshold:
                steep = -threshold
            slope_steep[i] = steep
        else:
            slope_steep[i] = 0
    return slope_steep


def calc_ascent_descent(df: pd.DataFrame, threshold=10) -> tuple[list[int], list[int]]:
    """
    Calculates the ascent and descent steepness in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing activity data.
        threshold (int, optional): Threshold for considering an elevation change significant. Defaults to 10.

    Returns:
        tuple[list[int], list[int]]: Two lists representing the ascent and descent for each data point.
    """
    slope_ascent = [0] * len(df)
    slope_descent = [0] * len(df)
    for i in range(1, len(df)):
        elev = df["enhanced_altitude"][i] - df["enhanced_altitude"][i - 1]
        if elev > threshold:
            elev = threshold
        elif elev < -threshold:
            elev = -threshold
        if elev > 0:
            slope_ascent[i] = np.abs(elev)
        elif elev < 0:
            slope_descent[i] = np.abs(elev)

    return slope_ascent, slope_descent


def parse_fit(filename: str, df_columns: list) -> tuple[DataFrame, Any]:
    """
    Parses a FIT file into a DataFrame.

    Args:
        filename (str): Name of the FIT file to be parsed.
        df_columns (list): List of column names to include in the DataFrame.

    Returns:
        tuple[DataFrame, Any]: DataFrame containing data from the FIT file and the last record.
    """
    fitfile = fitparse.FitFile(filename)
    activity = []
    for record in fitfile.get_messages("record"):
        r = {}
        for col in df_columns:
            r[col] = record.get_value(col)
        activity.append(r)

    return pd.DataFrame(activity), record


def basic_outlier_detect(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Performs basic outlier detection on a DataFrame.

    This method applies simple rules to detect and handle outliers in the cadence, heart rate, and enhanced speed columns.

    Args:
        df (pd.DataFrame): DataFrame containing activity data.

    Returns:
        pd.DataFrame: The DataFrame after outlier detection and correction.
    """

    df["cadence"] = df.cadence.apply(lambda i: i if 30 < i < 110 else np.mean(df.cadence))
    df["heart_rate"] = df.heart_rate.apply(lambda i: i if 50 < i < 210 else np.mean(df.heart_rate))
    df["enhanced_speed"] = df.enhanced_speed.apply(lambda i: i if 5 < i < 30 else np.mean(df.enhanced_speed))
    return df

def calc_dist(pos1: tuple,pos2: tuple) -> int:
    """
    Calculates the Haversine distance between two points.

    Args:
        pos1 (tuple): The first position (latitude, longitude).
        pos2 (tuple): The second position (latitude, longitude).

    Returns:
        int: The distance in meters between the two points.
    """
    return hs.haversine(pos1, pos2, unit=hs.Unit.METERS)

def calc_delayed(lst: pd.Series, window = 1) -> pd.Series:
    """
    Calculates a delayed version of a given Series.

    Args:
        lst (pd.Series): Series of variable values.
        window (int, optional): The window size for rolling mean calculation. Defaults to 1.

    Returns:
        pd.Series: The delayed series.
    """
    return lst.rolling(window, min_periods=1).mean()

def get_hr_zone(hr: pd.Series) -> pd.DataFrame:
    """
    Determines the heart rate zone based on the average heart rate.

    Args:
        hr (pd.Series): Series containing heart rate data.

    Returns:
        int: The heart rate zone.
    """
    mean_hr = np.mean(hr)
    if mean_hr < 140:
        zone = 1
    elif mean_hr > 140 and mean_hr < 156:
        zone = 2
    elif mean_hr > 156 and mean_hr < 166:
        zone = 3
    elif mean_hr > 166 and mean_hr < 175:
        zone = 4
    elif mean_hr > 175:
        zone = 5
    return zone


def calc_windows(df, lagged, cols) -> pd.DataFrame:
    """
    Calculates window features for specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        lagged (int): The number of lags to consider for window features.
        cols (list): List of column names for which to calculate window features.

    Returns:
        pd.DataFrame: The DataFrame with window features added.
    """
    for lag in range(1,lagged):
        wft = WindowFeatures(variables=cols,
                             window =lag)
        df = wft.fit_transform(df)
        df.fillna(0, inplace=True)
    return df

def calc_dt(df,cols, date) -> pd.DataFrame:
    """
    Calculates datetime features for a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        cols (list): List of datetime features to extract.
        date (datetime): The reference datetime for feature extraction.

    Returns:
        pd.DataFrame: The DataFrame with datetime features added.
    """
    df["date"] = date
    dtf = DatetimeFeatures(features_to_extract=cols)
    return dtf.fit_transform(df)

def calc_moving(df,col, max_range) -> pd.DataFrame:
    """
    Calculates moving averages for a specified column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        col (str): The column name for which to calculate moving averages.
        max_range (int): The maximum range for moving averages.

    Returns:
        pd.DataFrame: The DataFrame with moving averages added.
    """
    for x in range(10,max_range,10):
        df[f"moved_{col}_{x}"] = uniform_filter1d(df[col], size=x)
    return df

def preprocessing(activity_type: str,
                  athlete_name:str,
                  df_columns: list,
                  path_to_load="fit_files",
                  name= None
                  ) -> list:
    """
    Preprocesses data from FIT files and applies various transformations.

    Args:
        activity_type (str): The type of activity (e.g., running, cycling).
        athlete_name (str): The name of the athlete.
        df_columns (list): List of columns to include in the DataFrame.
        path_to_load (str, optional): Path to the directory containing FIT files. Defaults to "fit_files".
        name (str, optional): Optional name for the processed data. Defaults to None.

    Returns:
        list: A list of processed DataFrames.
    """
    warnings.filterwarnings("ignore")

    files = glob.glob(os.path.join(path_to_load, athlete_name, activity_type, "*.fit"))

    for file in tqdm(files):
        df, record = parse_fit(file, df_columns)
        df.dropna(inplace=True)

        if len(df) != 0 and (set(df.columns) == set(df_columns)):
            df_len = len(df)
            df.set_index(["timestamp"], inplace=True)
            df["enhanced_speed"] = df["enhanced_speed"].apply(lambda i: i * 3.6)

            df["temp"], df["humidity"], df["rain"], df["snow"], df["wind_speed"] = get_meteo(
                lat=record.get_value("position_lat"),
                long=record.get_value("position_long"),
                alt=df["enhanced_altitude"][0],
                day=df.index[0],
            )

            df["dist_diff"] = df.distance.diff(1)

            df["slope_ascent"], df["slope_descent"] = calc_ascent_descent(df)

            df["slope_steep"] = calc_slope_steep(df)

            df = basic_outlier_detect(
                df=df
            )

            for fce in ["sum", "mean", "min", "max"]:
                df = MathFeatures(variables=["heart_rate", "cadence"], func=fce).fit(df).transform(df)

            df["enhanced_altitude_delayed"] = calc_delayed(df.enhanced_altitude, window=1)
            df["cadence_altitude_delayed"] = calc_delayed(df.cadence, window=1)

            df["hr_zone"] = get_hr_zone(df.heart_rate)

            df = calc_dt(df, cols=['month','week','hour','minute','second'], date=df.index)

            df = calc_windows(df=df,
                              lagged=18,
                              cols=["slope_steep", "slope_ascent", "slope_descent", "heart_rate", "cadence"])

            df = calc_moving(df=df, col="heart_rate",
                             max_range=110)
            df = calc_moving(df=df, col="cadence",
                             max_range=110)

            df["peak"] = [0] + segment_elev(df)

            df.dropna(inplace=True)

            if(df_len != len(df)): print("Nerovná se", df_len, len(df))

            save_pcl(df,
                     activity_type=activity_type,
                     athlete_name=athlete_name,
                     path_to_save=conf["Paths"]["pcl"],
                     name=name)


def save_pcl(
    df: list,
    activity_type: str,
    athlete_name: str,
    path_to_save: str,
    name: str
):
    """
    Saves processed data into pickle files.

    Args:
        df (list): List of DataFrames to save into pickle files.
        activity_type (str): Type of the activity.
        athlete_name (str): Name of the athlete.
        path_to_save (str): Save path for the pickle files.
        name (str): Name for the saved pickle file.
    """

    if name is None:
        path = os.path.join(path_to_save, athlete_name, activity_type)
        name = df.index[0].strftime("%Y%m%d%H%M")
    else:
        path = os.path.join("src","test_activities", athlete_name, activity_type)

    if path_to_save not in os.listdir():
        os.mkdir(path_to_save)

    if athlete_name not in os.listdir(path_to_save):
        os.mkdir(os.path.join(path_to_save, athlete_name))

    if activity_type not in os.listdir(os.path.join(path_to_save, athlete_name)):
        os.mkdir(os.path.join(path_to_save, athlete_name, activity_type))

    df.to_pickle(
        os.path.join(path, name + ".pcl")
        )

    log.info(f"{len(df)} files saved into pickles")


def segment_elev(df: pd.DataFrame) -> list[int]:
    """
    Segments elevation data in a DataFrame to identify peaks.

    Args:
        df (pd.DataFrame): DataFrame containing elevation data.

    Returns:
        list[int]: A list indicating peaks in the elevation data.
    """
    def isMonotonic(A):
        return all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or all(A[i] >= A[i + 1] for i in range(len(A) - 1))

    altitude = df.enhanced_altitude

    segments = [(0, 0)]
    peak_signs = []

    for x in range(0, len(altitude)-1):
        if not isMonotonic(altitude[segments[-1][1]:x]):
            sign = -1 * np.sign(altitude[segments[-1][1]] - altitude[x])
            peak_signs.append(sign)
            segments[-1] = (segments[-1][0], x)
        else:
            peak_signs.append(0)

    return peak_signs

def load_test_activity(path: str, race_day: str) -> pd.DataFrame:
    """
    Loads test activity data from a GPX file.

    Args:
        path (str): Path to the GPX file.
        race_day (str): The date of the race or activity in 'YYYY-MM-DD-HH-MM' format.

    Returns:
        pd.DataFrame: The processed DataFrame containing activity data.
    """
    df = Converter(input_file=path).gpx_to_dataframe()

    df["timestamp"] = df.time

    df.drop("time",axis=1,inplace=True)
    df.set_index("timestamp",inplace=True)

    if "altitude" in df.columns:
        df['enhanced_altitude'] = df.altitude
        df.drop("altitude", axis=1, inplace=True)
    else:
        df['enhanced_altitude'] = 0
        df.drop("altitude", axis=1, inplace=True)

    df['dist_diff'] = [0] + [calc_dist(
        (df['latitude'].iloc[x], df['longitude'].iloc[x]),
        (df['latitude'].iloc[x + 1], df['longitude'].iloc[x + 1])) for x in range(len(df) - 1)]

    df['distance'] = df['dist_diff'].cumsum()

    df["temp"], df["humidity"], df["rain"], df["snow"], df["wind_speed"] = get_meteo(
        df.latitude.iloc[0],
        df.longitude.iloc[0],
        df.enhanced_altitude.iloc[0],
        datetime.strptime(race_day, '%Y-%m-%d-%H-%M')
    )
    df.drop(['latitude', 'longitude'], axis=1, inplace=True)

    df["slope_ascent"], df["slope_descent"] = calc_ascent_descent(df)

    df["slope_steep"] = calc_slope_steep(df)

    df['hr_zone'] = 5

    df["enhanced_altitude_delayed"] = calc_delayed(df.enhanced_altitude, window=3)

    df["peak"] = [0] + segment_elev(df)

    df = calc_windows(df=df,
                           lagged=15,
                           cols=["slope_steep", "slope_ascent", "slope_descent"])


    df = calc_dt(df, cols=['month', 'week', 'hour', 'minute', 'second'], date=df.index)

    return df

def segment_data(data: list) -> tuple:
    """
    Segments activity data into low and high distance categories.

    Args:
        data (list): List of DataFrames containing activity data.

    Returns:
        tuple: A tuple of two lists, one for low distance activities and one for high distance activities.
    """
    low_dist = []
    high_dist = []
    for act in data:
        if np.max(act.distance) > 10000:
            high_dist.append(act)
        else:
            low_dist.append(act)

    return low_dist, high_dist

