import random
import time
from typing import Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.modules.evl import rmse, nrmse
from src.modules.pred import predict
from src.modules.spec import ols_form
from src.modules import conf, log

"""
Module for implementation of heuristics_deprecated algorithm
"""


def get_form(cols: list, endog: str):
    return f'{endog} ~ {" + ".join([x for x in cols if x != endog])} -1'


def generate_cols(cols: list, ban_cols=[]) -> list:
    """
    Create list of all possible values in formula
    :param col: list of columns name
    :return: list of possible values in formula
    """
    n = random.randint(4, len(cols))
    new_cols = []
    can_append =[]
    for x in range(n):
        tmp = cols[random.randint(0, len(cols) - 1)]
        if ban_cols !=[]:
            can_append = [True if x not in tmp else False for x in ban_cols]
        if tmp not in new_cols and False not in can_append:
            new_cols.append(tmp)
    return new_cols


def random_shoot(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hmax: int,
    endog: str,
    ban_cols=[]
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Implementation of random shooting heuristics_deprecated algorithm
    :param train_df: training dataframe
    :param test_df: testing dataframe
    :param hmax: maximal number of shoots
    :param endog: endogenous variable
    :param barrier: RMSE number that we want
    :return: minimal RMSE, the best formula, RMSE list, columns list
    """
    rmse_lst = []
    new_cols = []
    nrmse_lst = []
    for _ in tqdm(range(hmax)):
        columns = generate_cols(train_df.columns, ban_cols)
        form = get_form(columns, endog)
        result = ols_form(train_df, form)                   # 2s
        pred_ols = predict(test_df, result)                 # + 0.5 s
        rmse_lst.append(rmse(test_df[endog], pred_ols))
        nrmse_lst.append(nrmse(test_df[endog],pred_ols))
        new_cols.append(columns)


    return np.min(rmse_lst), new_cols[rmse_lst.index(np.min(rmse_lst))], rmse_lst,new_cols,nrmse_lst


def shoot_and_go(
    train_df: pd.core.frame.DataFrame,
    test_df: pd.core.frame.DataFrame,
    columns: list,
    rmse: int,
    n_iter: int,
    endog='enhanced_speed',
) -> tuple:
    """
    Implementation of Shoot&Go heuristics_deprecated algorithm, optimization after random shooting algorithm
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
            new_rmse = elv.rmse(test_df[endog], pred_ols)
            if new_rmse < rmse:
                rmse = new_rmse
                break

    return rmse, n
