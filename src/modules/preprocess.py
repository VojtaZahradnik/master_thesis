import glob
import math
import os
from feature_engine.timeseries.forecasting import WindowFeatures
from scipy.ndimage import uniform_filter1d
from feature_engine.datetime import DatetimeFeatures

import warnings
from datetime import datetime, timedelta
from typing import Any, Union

import fitparse
import numpy as np
import pandas as pd
import scipy.interpolate
from meteostat import Daily, Point, Hourly
from src.modules import conf, df_columns, log
from pandas import DataFrame
from tqdm import tqdm
import shutil

"""
Load values from fit files and add some new variables from external libraries
"""

def get_meteo(lat: str, long: str, alt: str, day: str) -> Union[list[Any], DataFrame]:
    """
    Implementation of meteostat library to get data about weather on start location of activity
    :param lat: latitude of start location
    :param long: longitude of start location
    :param alt: altitude of start location
    :param day: day of the activity start
    :return: weather information about start location or basic values, when meteostat don't have data about location
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
    Calculation of slope steep for every row of data
    :param df: activity dataframe
    :param threshold: steep needed to push a bike or walk in running
    :return: slope steep
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
    Calculation of ascent and descent of steep
    :param df: activity dataframe
    :param threshold: steep needed to push a bike or walk in running
    :return: Two lists of ascent and descent
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
    Function for parsing fit file
    :param filename: name of the file
    :param df_columns: list of column names
    :return: dataframe from fit file
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
    Method for basic outlier detection based just on values
    """

    df["cadence"] = df.cadence.apply(lambda i: i if 30 < i < 110 else np.mean(df.cadence))
    df["heart_rate"] = df.heart_rate.apply(lambda i: i if 50 < i < 210 else np.mean(df.heart_rate))
    df["enhanced_speed"] = df.enhanced_speed.apply(lambda i: i if 5 < i < 30 else np.mean(df.enhanced_speed))
    return df


def calc_delayed(lst: pd.Series, window = 1) -> pd.Series:
    """
    Method for calculation of delayed variable in time t=-1
    :param lst: list of variable
    """
    return lst.rolling(window, min_periods=1).mean()

def get_hr_zone(hr: pd.Series):
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

def calc_windows(df, lagged, cols):
    for lag in range(1,lagged):
        wft = WindowFeatures(variables=cols,
                             window =lag)
        df = wft.fit_transform(df)
        df.fillna(0, inplace=True)
    return df

def calc_dt(df,cols, date):
    df["date"] = date
    dtf = DatetimeFeatures(features_to_extract=cols)
    return dtf.fit_transform(df)

def calc_moving(df,col, max_range):
    for x in range(10,max_range,10):
        df[f"moved_{col}_{x}"] = uniform_filter1d(df[col], size=x)
    return df

def preprocessing(activity_type: str,
                  athlete_name:str,
                  df_columns: list,
                  path_to_load="fit_files",
                  ) -> list:

    warnings.filterwarnings("ignore")
    # path=os.path.join(conf["Paths"]["pcl"], athlete_name, activity_type)
    # if os.path.exists(path):
    #     shutil.rmtree(path)

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

            df.dropna(inplace=True)

            if(df_len != len(df)): print("Nerovná se", df_len, len(df))

            save_pcl(df,
                     activity_type=activity_type,
                     athlete_name=athlete_name,
                     path_to_save=conf["Paths"]["pcl"])


def save_pcl(
    df: list,
    activity_type: str,
    athlete_name: str,
    path_to_save: str,
):
    """
    Save loaded dataframes into pickles
    :param df: list of dataframes to save into pickles
    :param athlete_name: name of the athlete
    :param activity_type: type of the activity
    :param path_to_save: save path of pickles
    """

    path = os.path.join(path_to_save, athlete_name, activity_type)
    if path_to_save not in os.listdir():
        os.mkdir(path_to_save)

    if athlete_name not in os.listdir(path_to_save):
        os.mkdir(os.path.join(path_to_save, athlete_name))

    if activity_type not in os.listdir(os.path.join(path_to_save, athlete_name)):
        os.mkdir(os.path.join(path_to_save, athlete_name, activity_type))

    df.to_pickle(
        os.path.join(path, df.index[0].strftime("%Y%m%d%H%M") + ".pcl")
        )

    log.info(f"{len(df)} files saved into pickles")


def segment_elev(df: pd.DataFrame) -> list[int]:
    # deprecated
    def is_monotonic(A: list) -> bool:
        return all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or all(
            A[i] >= A[i + 1] for i in range(len(A) - 1)
        )

    def get_elev():
        if len(segments_desc) == len(segments_asc):
            elev = [x for y in zip(segments_desc, segments_asc) for x in y]
        elif len(segments_asc) > len(segments_desc):
            elev = [x for y in zip(segments_asc, segments_desc) for x in y] + [
                segments_asc[-1]
            ]
        else:
            elev = [x for y in zip(segments_desc, segments_asc) for x in y] + [
                segments_desc[-1]
            ]
        return elev

    segments = []
    segments_asc = []
    segments_desc = []

    xnew = np.linspace(0, 20, len(df.enhanced_altitude))
    spl = scipy.interpolate.UnivariateSpline(xnew, df.enhanced_altitude)
    altitude = spl(xnew)

    tmp = 0
    for x in range(len(altitude)):
        if is_monotonic(altitude[tmp:x]):
            continue
        else:
            segments.append((tmp, x))
            tmp = x
    segments.append((tmp, x))
    for i in segments:
        for _ in i:
            sign = -1 * np.sign(altitude[i[0]] - altitude[i[1]])
        if sign == -1:
            segments_desc.append(i)
        else:
            segments_asc.append(i)
    return get_elev()
