import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import math

from src.modules import conf, log, spec

"""
Model Fit with some mandatory functions
"""

def load_pcls(athlete_name: str, activity_type: str, path_to_load: str) -> list:
    """
    Load pickles files from load path
    :param athlete_name: name of the athlete
    :param activity_type: type of the activities
    :param path_to_load: pickle file path
    :return: data from pickles in list of dataframes
    """
    dfs = []
    files = sorted(glob.glob(os.path.join(path_to_load, athlete_name, activity_type, "*.pcl")))
    if len(files) != 0:
        for x in tqdm(range(len(files))):
            try:
                dfs.append(pd.read_pickle(files[x]))
            except RuntimeError:
                log.error("Runtime error in loading of pickles.")
        log.info(f"{len(files)} pickle files successfully loaded")
    else:
        log.error("Pickle bjetca is empty")
        sys.exit(1)

    return dfs


def fit_df(train_df: pd.DataFrame, fitted_val="speed_fitted", form="") -> pd.DataFrame:
    """
    Compute fitted value for input dataframe
    :param train_df: training dataset
    :param form: loss function
    :param fitted_val: Name of the fitted variable
    :return: dataframe with fitted value
    """
    train_df = clean_data(train_df)
    if len(train_df) != 0:
        train_df[fitted_val] = spec.ols_form(df=train_df, form=form).fittedvalues
    else:
        log.error("Training dataframe is empty")
        sys.exit(1)

    train_df = clean_data(train_df)
    return train_df


def get_race_index(df: pd.DataFrame, date: str) -> int:
    """
    Find activity index by date
    :param df: dataframe of zahradnik
    :param date: start time of modules activity
    :return: index of selected activity
    """
    for x in range(len(df)):
        if str(df[x].index[0].strftime("%Y-%m-%d-%H-%M")) < date:
            continue
        else:
            break
    return x


def get_train_test_df(data: list, ratio= 0.8) -> (pd.DataFrame, pd.DataFrame):
    concated_data = pd.concat(data)
    train_df = concated_data[0:round(len(concated_data) * ratio)]
    train_df = concated_data[0:len(train_df) + get_diff(concated_data,train_df)]
    test_df = concated_data[len(train_df):]

    return clean_data(train_df), clean_data(test_df)

def get_test_valid_df(test_df: pd.DataFrame):
    last_act_day = test_df.index[-1].date()
    for x in range(len(test_df.index)):
        if(test_df.index[x].date() == last_act_day):
            sep = x
            break

    return test_df[:sep], test_df[sep:]

def get_diff(concated_data: pd.DataFrame, df: pd.DataFrame) -> int:
    break_day = concated_data[len(df):].index[0].date()
    counter = 0
    for x in concated_data[len(df):].index:
        if (x.date() == break_day):
            counter += 1
        else:
            break
    return counter

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna(axis=0).drop_duplicates()

def calc_final_time(distance: pd.Series, speed: pd.Series):
    time = ((np.max(distance) / 1000) / np.mean(speed)) * 60
    minutes = math.floor(time)
    seconds = round((time - minutes) * 60)
    if seconds == 60:
        seconds = 0
        minutes += 1
    return f'Final time: {minutes}:{seconds}'