import glob
import math
import os
import time
import warnings
from datetime import datetime
from typing import Any, Union

import fitparse
import numpy as np
import pandas as pd
import scipy.interpolate
from meteostat import Daily, Point
from src.modules import conf, df_columns, log
from pandas import DataFrame
from tqdm import tqdm

"""
Load values from fit files and add some new variables from external libraries
"""


def input_cli():
    """
    Solution for CLI input without __init__
    """
    if conf["Athlete"]["activity_type"] == "run":
        df_columns.append("cadence")

    data = preprocessing(
        activity_type=conf["Athlete"]["activity_type"],
        athlete_name=conf["Athlete"]["name"],
        df_columns=df_columns,
        path_to_load=conf["Paths"]["fit"],
    )
    save_pcls(
        dfs=data,
        activity_type=conf["Athlete"]["activity_type"],
        athlete_name=conf["Athlete"]["name"],
        path_to_save=conf["Paths"]["pcl"],
    )


def get_meteo(lat: str, long: str, alt: str, day: str) -> Union[list[Any], DataFrame]:
    """
    Implementation of meteostat library to get data about weather on start location of activity
    :param lat: latitude of start location
    :param long: longitude of start location
    :param alt: altitude of start location
    :param day: day of the activity start
    :return: weather information about start location or basic values, when meteostat don't have data about location
    """
    lst = [15, 0, 0, 0]
    if lat is not None and long is not None:

        transfer_num = 11930465
        if not (isinstance(alt, int)):
            alt = 200
        loc = Point(int(lat) / transfer_num, int(long) / transfer_num, int(alt))
        day = datetime(
            day.year,
            day.month,
            day.day,
        )
        data = Daily(loc, day, day)
        data = data.fetch()
        if len(data != 0):
            lst = [
                data["tavg"][0],
                data["wspd"][0],
                data["wdir"][0],
                data["prcp"][0],
            ]

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
        elev = np.max([np.abs(df["slope_descent"][i]), df["slope_ascent"][i]])
        tmp = -1 if df["slope_descent"][i] < 0 else 1
        if dist != 0:
            steep = tmp * (elev / dist) * 100
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
    activity_type: str,
    threshold_cycling: [],
    threshold_running: [],
) -> pd.DataFrame:
    """
    Method for basic outlier detection based just on values
    :param df: dataframe for detection of outliers
    :param activity_type: name of the activity type
    :param threshold_cycling: Threshold for cycling
    :param threshold_running: Threshold for running
    """

    columns = [
        x for x in df_columns if x not in ["timestamp", "distance", "enhanced_altitude"]
    ]
    if activity_type == "cycling":
        threshold = threshold_cycling
    else:
        threshold = threshold_running
    for i, j in zip(columns, threshold):
        df = df[df[i] > j[0]]
        df = df[df[i] < j[1]]

    if "cadence" in df_columns:
        df["cadence"] = [i if i > 1 else 1 for i in df["cadence"]]
        df["cadence_delayed"] = [i if i > 1 else 1 for i in df["cadence_delayed"]]
    return df


def calc_delayed(lst: pd.Series) -> pd.Series:
    """
    Method for calculation of delayed variable in time t=-1
    :param lst: list of variable
    """
    return lst.rolling(1, min_periods=1).mean()


def calc_dist_diff(distance: pd.Series) -> list:
    diff = [0.00]
    for x in range(1, len(distance)):
        diff.append(distance[x] - distance[x - 1])
    return diff


def preprocessing(
    activity_type: str,
    athlete_name: str,
    df_columns: list,
    path_to_load="fit_files",
    run=None,
    cyc=None,
) -> list:
    """
    Load zahradnik from Fit files and add some needed variables (slope steep, weather information)
    :param athlete_name: name of the athlete
    :param activity_type: type of the zahradnik used in model
    :param df_columns: dataframe columns
    :param path_to_load: pickle file path
    :param run: Running thresholds
    :param cyc: Cycling thresholds
    :return: loaded list with dataframes of valuable zahradnik
    """
    if run is None:
        run = [(50, 210), (3, 100), (30, 180)]
    if cyc is None:
        cyc = [(60, 210), (3, 25), (50, 210)]

    start = time.monotonic()
    warnings.filterwarnings("ignore")

    dfs = []
    form = []
    days = glob.glob(os.path.join(path_to_load, athlete_name, activity_type, "*.fit"))
    for x in tqdm(range(len(days))):
        try:
            df, record = parse_fit(days[x], df_columns)
            df.dropna(inplace=True)
            if len(df) != 0 and (set(df.columns) == set(df_columns)):
                df.set_index(["timestamp"], inplace=True)
                df["enhanced_speed"] = [x * 3.6 for x in df["enhanced_speed"]]

                df["temp"], df["wind_speed"], df["wind_direct"], df["rain"] = get_meteo(
                    record.get_value("position_lat"),
                    record.get_value("position_long"),
                    df["enhanced_altitude"][0],
                    df.index[0],
                )

                df["dist_diff"] = calc_dist_diff(df.distance)
                if "cadence" in df_columns:
                    df["cadence_delayed"] = calc_delayed(df.cadence)

                df["enhanced_altitude_delayed"] = calc_delayed(df.enhanced_altitude)

                (
                    df["slope_ascent"],
                    df["slope_descent"],
                ) = calc_ascent_descent(df)

                df["slope_steep"] = calc_slope_steep(df)

                if form == []:
                    form = gen_fce_form(df=df, endog="enhanced_speed")
                for k in form:
                    df[f"{k[0]}_{k[2]}"] = eval(k[1])

                df = basic_outlier_detect(
                    df,
                    activity_type,
                    threshold_running=run,
                    threshold_cycling=cyc,
                )
            else:
                continue

        except RuntimeError as e:
            log.error("RuntimeError in preprocessing")
            log.error(e)
        except ValueError as e:
            log.erorr("Value error in preprocessing")
            log.error(e)
        df.dropna(inplace=True)
        if len(df) != 0:
            dfs.append(df)

    log.info(
        f"{len(days)} files was preprocessed after {round(time.monotonic() - start,2)}"
    )
    return dfs


def save_pcls(
    dfs: list,
    activity_type: str,
    athlete_name: str,
    path_to_save: str,
):
    """
    Save loaded dataframes into pickles
    :param dfs: list of dataframes to save into pickles
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

    for x in tqdm(range(len(dfs))):
        dfs[x].to_pickle(
            os.path.join(path, dfs[x].index[0].strftime("%Y%m%d%H%M") + ".pcl")
        )

    log.info(f"{len(dfs)} files saved into pickles")


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


def gen_fce_form(df: pd.DataFrame, endog: str) -> list:
    """
    Function for generate all possible combinations between columns and basic math. functions
    :param train_df: Dataframe with columns
    :param endog: endogenous variable
    """

    possibilities = [x for x in list(df.columns) if x != endog]
    form_full = []
    for x in possibilities:
        if not (((df[x] <= 0).any().any())):
            form_full.append(("log", f'np.log(df["{x}"])', x))
        form_full.append(("sin", f'np.sin(df["{x}"])', x))
        form_full.append(("cos", f'np.cos(df["{x}"])', x))
        form_full.append(("tan", f'np.tan(df["{x}"])', x))
        form_full.append(("diff", f'df["{x}"].diff()', x))

    return form_full


def load_race(
    conf: dict, train_df: pd.Series, path: str, day_of_race: str, columns=None
):
    warnings.filterwarnings("ignore")
    if columns is None:
        columns = ["distance", "enhanced_altitude"]
    test_df, record = parse_fit(path, columns)
    day_of_race = datetime.strptime(day_of_race, "%Y-%m-%d-%H-%M")
    test_df = get_meteo(
        record.get_value("position_lat"),
        record.get_value("position_long"),
        test_df["enhanced_altitude"][0],
        day_of_race,
        test_df,
    )

    test_df["dist_diff"] = calc_dist_diff(test_df.distance)
    test_df["enhanced_altitude_delayed"] = calc_delayed(test_df.enhanced_altitude)
    (
        test_df["slope_ascent"],
        test_df["slope_descent"],
    ) = calc_ascent_descent(test_df)
    test_df["slope_steep"] = calc_slope_steep(test_df)

    df = test_df
    form = gen_fce_form(df)
    for k in form:
        if k[0] not in df.columns:
            df[f"{k[0]}_{k[2]}"] = eval(k[1])

    test_df = df
    test_df.dropna(inplace=True)
    test_df.drop_duplicates(inplace=True)
    from src.heuristics.random_shooting import get_form
    from src.modules.fit import clean_data
    from src.modules.pred import predict
    from src.modules.spec import ols_form

    train_df, test_df = clean_data(train_df), clean_data(test_df)

    # form = get_form(train_df.columns.intersection(test_df.columns), endog='heart_rate')
    form = get_form(
        test_df.columns.intersection(conf["best_loss_func_hr"]),
        endog="heart_rate",
    )
    heart_model = ols_form(train_df, form)
    test_df["heart_rate"] = predict(test_df, heart_model)

    test_df = clean_data(test_df)

    test_df["log_heart_rate"] = np.log(test_df["heart_rate"])
    test_df["diff_heart_rate"] = test_df["heart_rate"].diff()
    test_df["sin_heart_rate"] = np.sin(test_df["heart_rate"])
    test_df["cos_heart_rate"] = np.cos(test_df["heart_rate"])
    test_df["tan_heart_rate"] = np.tan(test_df["heart_rate"])

    # form = get_form(list(conf['best_loss_func_cad']), endog='cadence')
    # form = get_form(train_df.columns.intersection(test_df.columns), endog='cadence')
    form = get_form(
        test_df.columns.intersection(conf["best_loss_func_cad"]),
        endog="cadence",
    )

    cadence_model = ols_form(train_df, form)
    test_df["cadence"] = predict(test_df, cadence_model)
    test_df["cadence"] = [
        x if not math.isnan(x) else test_df["cadence"].mean()
        for x in test_df["cadence"]
    ]

    test_df = clean_data(test_df)

    test_df["cadence_delayed"] = calc_delayed(test_df["cadence"])
    test_df["cadence_delayed"] = [x if x > 1 else 1 for x in test_df["cadence_delayed"]]
    test_df["log_cadence"] = np.log(test_df["cadence"])
    test_df["log_cadence_delayed"] = np.log(test_df["cadence_delayed"])
    test_df["diff_cadence"] = test_df["cadence"].diff()
    test_df["diff_cadence_delayed"] = test_df["cadence_delayed"].diff()
    test_df["sin_cadence_delayed"] = np.sin(test_df.cadence_delayed)
    test_df["cos_cadence_delayed"] = np.cos(test_df.cadence_delayed)
    test_df["tan_cadence_delayed"] = np.tan(test_df.cadence_delayed)
    test_df["sin_cadence"] = np.sin(test_df["cadence"])
    test_df["cos_cadence"] = np.cos(test_df["cadence"])
    test_df["tan_cadence"] = np.tan(test_df["cadence"])

    return clean_data(test_df)
