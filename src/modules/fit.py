import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from feature_engine.creation import MathFeatures
from src.modules import conf, log, spec, preprocess
from datetime import datetime, timedelta
"""
Model Fit with some mandatory functions
"""

def load_pcls(athlete_name: str, activity_type: str, path_to_load: str) -> list:
    """
    Loads pickle files containing data for a specific athlete and activity type.

    This function reads pickle files from a specified directory, each file representing different activity data for an athlete.
    It appends the data from each pickle file to a list of dataframes and returns it.

    Args:
        athlete_name (str): The name of the athlete.
        activity_type (str): The type of the activities to be loaded.
        path_to_load (str): The path to the directory containing the pickle files.
        race_day (datetime): The date of the race or event.

    Returns:
        list: A list of dataframes, each containing data from a single pickle file.
    """
    dfs = []
    files = sorted(glob.glob(os.path.join(path_to_load, athlete_name, activity_type, "*.pcl")))
    if len(files) != 0:
        for x in tqdm(range(len(files))):
            try:
                df = pd.read_pickle(files[x])
                dfs.append(df)
            except RuntimeError:
                log.error("Runtime error in loading of pickles.")
        log.info(f"{len(files)} pickle files successfully loaded")
    else:
        log.error("Pickle bjetca is empty")
        sys.exit(1)

    return dfs


def fit_df(train_df: pd.DataFrame, fitted_val="speed_fitted", form="") -> pd.DataFrame:
    """
    Computes and adds a fitted value column to the training dataframe.

    This function takes a dataframe, cleans it, and then computes a fitted value based on a specified formula.
    The fitted values are added as a new column to the dataframe.

    Args:
        train_df (pd.DataFrame): The training dataframe.
        fitted_val (str, optional): The name of the column to be added with fitted values. Defaults to "speed_fitted".
        form (str, optional): The formula to be used for fitting. Defaults to an empty string.

    Returns:
        pd.DataFrame: The modified dataframe with an additional column of fitted values.
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
    Finds the index of a specific activity in a dataframe based on its date.

    Args:
        df (pd.DataFrame): The dataframe containing multiple activities.
        date (str): The start time of the activity to find, in string format.

    Returns:
        int: The index of the selected activity within the dataframe.
    """
    for x in range(len(df)):
        if str(df[x].index[0].strftime("%Y-%m-%d-%H-%M")) < date:
            continue
        else:
            break
    return x


def get_train_test_df(data: list, ratio= 0.8) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits a concatenated dataframe into training and testing sets based on a specified ratio.

    Args:
        data (list): A list containing dataframes to be concatenated and split.
        ratio (float, optional): The ratio of training data to the total data. Defaults to 0.8.

    Returns:
        tuple: A tuple containing two dataframes (training set, testing set).
    """
    concated_data = pd.concat(data)
    train_df = concated_data[0:round(len(concated_data) * ratio)]
    train_df = concated_data[0:len(train_df) + get_diff(concated_data,train_df)]
    test_df = concated_data[len(train_df):]

    return clean_data(train_df), clean_data(test_df)

def get_test_valid_df(test_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits a test dataframe into test and validation sets based on the date of the last activity.

    Args:
        test_df (pd.DataFrame): The dataframe to be split.

    Returns:
        tuple: A tuple containing two dataframes (test set, validation set).
    """
    last_act_day = test_df.index[-1].date()
    for x in range(len(test_df.index)):
        if(test_df.index[x].date() == last_act_day):
            sep = x
            break

    return test_df[:sep], test_df[sep:]

def get_diff(concated_data: pd.DataFrame, df: pd.DataFrame) -> int:
    """
    Computes the difference in data points between two consecutive days in a dataframe.

    Args:
        concated_data (pd.DataFrame): The complete concatenated dataframe.
        df (pd.DataFrame): A subset of the concatenated dataframe.

    Returns:
        int: The number of data points on the break day.
    """
    break_day = concated_data[len(df):].index[0].date()
    counter = 0
    for x in concated_data[len(df):].index:
        if (x.date() == break_day):
            counter += 1
        else:
            break
    return counter

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataframe by replacing infinite values with NaN, dropping NaN values and duplicates.

    Args:
        df (pd.DataFrame): The dataframe to be cleaned.

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    return df.replace([np.inf, -np.inf], np.nan).dropna(axis=0).drop_duplicates()

def calc_final_time(distance: pd.Series, speed: pd.Series) -> pd.Series:
    """
    Calculates the final time for a distance at an average speed.

    Args:
        distance (pd.Series): The series containing distance data.
        speed (pd.Series): The series containing speed data.

    Returns:
        str: The final time in the format 'minutes:seconds'.
    """
    time = ((np.max(distance) / 1000) / np.mean(speed)) * 60
    minutes = math.floor(time)
    seconds = round((time - minutes) * 60)
    if seconds == 60:
        seconds = 0
        minutes += 1
    return f'Final time: {minutes}:{seconds}'

def get_final_df(train_df: pd.DataFrame,test_df: pd.DataFrame, model, race_name: str, athlete_name: str) -> pd.DataFrame:
    """
    Fits a model to the training data and applies it to the test data to predict various performance metrics.

    The function fits the model to the training data for different variables like cadence and heart rate,
    and then predicts these variables for the test data. It then applies various transformations and calculations
    to the test data, including speed prediction and final time calculation.

    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The testing dataset.
        model: The machine learning model to be used for prediction.
        race_name (str): The name of the race or event.
        athlete_name (str): The name of the athlete.

    Returns:
        str: The final predicted time for the race in the format 'minutes:seconds'.
    """

    model.fit(train_df[test_df.columns], train_df.cadence)
    test_df['cadence'] = model.predict(test_df)
    test_df['cadence'].mean()

    test_df = preprocess.calc_windows(df=test_df,
                                      lagged=15,
                                      cols=["cadence"])
    test_df = preprocess.calc_moving(df=test_df, max_range=110, col="cadence")

    model.fit(train_df[test_df.columns], train_df.heart_rate)
    test_df["heart_rate"] = model.predict(test_df)
    test_df["heart_rate"].mean()

    for fce in ["sum", "mean", "min", "max"]:
        test_df = MathFeatures(variables=["heart_rate", "cadence"], func=fce).fit(test_df).transform(test_df)

    test_df = preprocess.calc_windows(df=test_df,
                                      lagged=12,
                                      cols=["heart_rate"])
    test_df = preprocess.calc_moving(df=test_df, max_range=110, col="heart_rate")

    model.fit(train_df[test_df.columns], train_df.enhanced_speed)
    test_df["enhanced_speed"] = model.predict(test_df)


    time = ((np.max(test_df.distance) / 1000) / np.mean(test_df["enhanced_speed"])) * 60
    minutes = math.floor(time)
    seconds = round((time - minutes) * 60)
    if seconds == 60:
        seconds = 0
        minutes += 1

    final_time = f'{minutes}:{seconds}'

    test_df.to_csv(f"src/output/{athlete_name}_{race_name}.csv")

    return final_time