import random
import time
from typing import Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.modules.evl import calc_rmse
from src.modules.pred import predict
from src.modules.spec import ols_form
from src.modules import conf, log

"""
Module for implementation of heuristics algorithm
"""


def get_form(cols: list, endog: str):
    return f'{endog} ~ {" + ".join([x for x in cols if x != endog])} -1'


def generate_cols(cols: list) -> list:
    """
    Create list of all possible values in formula
    :param col: list of columns name
    :return: list of possible values in formula
    """
    n = random.randint(4, len(cols))
    new_cols = []
    for x in range(n):
        tmp = cols[random.randint(0, len(cols) - 1)]
        if tmp not in new_cols:
            new_cols.append(tmp)
    return new_cols


def random_shoot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hmax: int,
    endog: str,
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Implementation of random shooting heuristics algorithm
    :param train_df: training dataframe
    :param test_df: testing dataframe
    :param hmax: maximal number of shoots
    :param endog: endogenous variable
    :param barrier: RMSE number that we want
    :return: minimal RMSE, the best formula, RMSE list, columns list
    """
    rmse = []
    new_cols = []
    for _ in tqdm(range(hmax)):
        columns = generate_cols(train_df.columns)
        form = get_form(columns, endog)
        result = ols_form(train_df, form)                   # 2s
        pred_ols = predict(test_df, result)                 # + 0.5 s
        rmse.append(calc_rmse(test_df[endog], pred_ols))
        new_cols.append(columns)


    return np.min(rmse), new_cols[rmse.index(np.min(rmse))], rmse, new_cols


def shoot_and_go(
    train_df: pd.core.frame.DataFrame,
    test_df: pd.core.frame.DataFrame,
    columns: list,
    rmse: int,
    n_iter: int,
    endog='enhanced_speed',
) -> tuple:
    """
    Implementation of Shoot&Go heuristics algorithm, optimization after random shooting algorithm
    :param train_df: training dataframe
    :param test_df: testing dataframe
    :param columns: list of columns name for generate formula
    :param rmse: basic metric to minimize
    :param endog: endogenous variable
    :return: best RMSE and best columns
    """

    while rmse > n_iter:
        for x in tqdm(columns):
            cols = columns[:]
            cols.remove(x)
            new_form = get_form(cols, endog)
            result = ols_form(train_df, new_form)
            pred_ols = predict(test_df, result)
            new_rmse = calc_rmse(test_df[endog], pred_ols)
            if new_rmse < rmse:
                rmse = new_rmse
                break

    return rmse, n
