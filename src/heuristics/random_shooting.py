import random
import time
from typing import Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.modules.evl import calc_rmse
from src.modules.predict import predict
from src.modules.spec import ols_form
from src.modules import conf, log

"""
Module for implementation of heuristics algorithm
"""


def get_form(cols: list, endog: str):
    return f'{endog} ~ {" + ".join([x for x in cols if x != endog])} -1'


def generate_cols(col: list) -> list:
    """
    Create list of all possible values in formula
    :param col: list of columns name
    :return: list of possible values in formula
    """
    n = random.randint(1, len(col))
    cols = []
    for x in range(n):
        tmp = col[random.randint(0, len(col) - 1)]
        if tmp not in cols:
            cols.append(tmp)
    return cols


def random_shoot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list,
    hmax: int,
    endog: str,
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Implementation of random shooting heuristics algorithm
    :param train_df: training dataframe
    :param test_df: testing dataframe
    :param cols: list of maximal possible formula
    :param hmax: maximal number of shoots
    :param endog: endogenous variable
    :param barrier: RMSE number that we want
    :return: minimal RMSE, the best formula, best columns, RMSE list, all columns
    """
    rmse = []

    col = []
    for _ in tqdm(range(hmax)):
        columns = generate_cols(cols)
        form = get_form(columns, endog)
        result = ols_form(train_df, form)
        pred_ols = predict(test_df, result)
        if len(columns) > 3:
            rmse.append(calc_rmse(test_df[endog], pred_ols) * len(columns))
            col.append(columns)
        else:
            continue

    return np.min(rmse), col[rmse.index(np.min(rmse))], rmse, col


def shoot_and_go(
    train_df: pd.core.frame.DataFrame,
    test_df: pd.core.frame.DataFrame,
    columns: list,
    rmse: int,
    barrier: float,
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
    n = columns[:]
    while rmse > barrier:
        for x in tqdm(columns):
            n = columns[:]
            n.remove(x)
            new_form = f'{endog} ~ {" + ".join(n)} -1'
            result = ols_form(train_df, new_form)
            pred_ols = predict(test_df, result)
            new_rmse = calc_rmse(test_df[endog], pred_ols)
            if new_rmse < rmse:
                rmse = new_rmse
                print(rmse)
                break

    return rmse, n
