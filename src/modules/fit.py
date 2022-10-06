import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import src.modules.spec as spec

"""
Model Fit with some mandatory functions
"""


def input_cli(conf: dict, log: logging.Logger):
    """
    Solution for CLI input without __init__
    :param conf: Configuration dictionary
    """
    data = load_pcls(
        conf["Athlete"]["name"],
        conf["Athlete"]["activity_type"],
        conf["Paths"]["pcl"],
    )
    log.info("Pickles loaded")
    if conf["Athlete"]["test_activity"] != "":
        index = get_race_index(data, conf["Athlete"]["test_activity"])
    else:
        index = len(data)
    train_df = pd.concat(data[0 : index - 1])
    train_df = clean_data(train_df)
    import src.heuristics.random_shooting as random_shooting

    df = fit_df(
        train_df=train_df,
        form=random_shooting.get_form(train_df, conf["endo_var"]),
    )
    log.info("Fitted")
    df.to_csv(
        os.path.join(conf["Paths"]["output"], f'{conf["Athlete"]["name"]}_fitted.csv')
    )
    log.info(f'Csv saved into {conf["Paths"]["output"]}')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True, axis=1)
    return df


def load_pcls(athlete: str, activity_type: str, path_to_load: str) -> list:
    """
    Load pickles files from load path
    :param athlete: name of the athlete
    :param activity_type: type of the activities used in model
    :param path_to_load: pickle file path
    :return: data from pickles in list of dataframes
    """
    df = []
    files = glob.glob(f"{path_to_load}{athlete}_{activity_type}/*")
    if len(files) != 0:
        for x in tqdm(range(len(files))):
            try:
                df.append(pd.read_pickle(files[x]))
                df[x].dropna(inplace=True)
            except RuntimeError:
                logging.getLogger("project").error(
                    "Runtime error in loading of pickles."
                )

        df = [x.dropna() for x in df]
        df = [x for x in df if len(x) != 0]
        logging.getLogger("project").info("Data successfully loaded from Pickles")
    else:
        logging.getLogger("project").error("Pickle folder is empty")
        sys.exit(1)

    return df


def fit_df(train_df: pd.DataFrame, fitted_val="speed_fitted", form="") -> pd.DataFrame:
    """
    Compute fitted value for input dataframe
    :param train_df: training dataset
    :param form: loss function
    :param fitted_val: Name of the fitted variable
    :return: dataframe with fitted value
    """
    train_df.drop_duplicates(inplace=True)
    if len(train_df) != 0:
        train_df[fitted_val] = spec.ols_form(df=train_df, form=form).fittedvalues
    else:
        logging.getLogger("project").error("Training dataframe is empty")
        sys.exit(1)
    train_df.drop_duplicates(inplace=True)
    train_df.dropna(inplace=True)
    return train_df


def get_race_index(df: pd.DataFrame, date: str) -> int:
    """
    Find activity index by date
    :param df: dataframe of activities
    :param date: start time of modules activity
    :return: index of selected activity
    """
    pos = -1
    for x in range(len(df)):
        if str(df[x].index[0].strftime("%Y-%m-%d-%H-%M")) == date:
            pos = x
            break
    return pos
