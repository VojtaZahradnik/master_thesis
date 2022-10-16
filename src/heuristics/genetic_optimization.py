import random
from typing import Any, List

import numpy as np
import pandas as pd
from numpy.random import randint, uniform
from tqdm import tqdm

from src.heuristics.random_shooting import get_form
from src.modules.evl import rmse
from src.modules.pred import predict
from src.modules.spec import ols_form

from src.modules import log, conf

"""
Module for GO
"""


def tournament_selection(pop: list, scores: list):
    selection_ix = randint(len(pop))

    for ix in randint(0, len(pop), randint(1,3)):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

def proportionate_selection(pop: list, scores: list):
    max_score = sum(scores)
    pick = random.uniform(0, max_score)
    current = 0
    for x in range(len(scores)):
        current += scores[x]
        if current > pick:
            return pop[x]

def generic_func(
    cols: list,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    endog: str,
):
    """
    Function that defines function for optimization
    :param cols: loss function
    :param train_df: training dataframe
    :param test_df: testing dataframe
    :param endog: endogenous variable
    """
    form = get_form(list(cols), endog=endog)
    result = ols_form(train_df, form)
    pred_ols = predict(test_df, result)
    return rmse(test_df[endog], pred_ols)


def mutation(cols: list, r_mut: int, full_form: list):
    """
    This procedure simply flips bits with a low probability controlled by the “r_mut”
    :param bitstring: list of columns for formula
    :param r_mut: hyperparameter determines the amount of mutation
    :param full_form: maximal possible formula in list
    """
    new_cols = cols.copy()
    for i in range(len(full_form)):
        if np.random.rand() < r_mut:
            if full_form[i] not in cols:
                new_cols.append(full_form[i])
            else:
                new_cols.pop(new_cols.index(full_form[i]))
    return new_cols

def crossover(p1: list, p2: list, r_cross: int) -> list:
    """
    Function that implementing crossover between two population
    :param p1: first parent
    :param p2: second parent
    :param r_cross: crossover rate that determines whether crossover is performed or not
    :return: list of childrens
    """
    c1, c2 = p1.copy(), p2.copy()
    if np.random.rand() < r_cross:
        pt = np.random.randint(1, len(p1))
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


def genetic_algorithm(
    n_iter: int,
    r_cross: int,
    r_mut: int,
    pop: list,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_col: list,
    endog: str,
) -> List[Any]:
    """
    Method for genetic algorithm, using functions defines above
    :param n_iter: number of iteration
    :param r_cross: crossover coefficient
    :param r_mut: mutation coefficient
    :param pop: population for GO
    :param train_df: train dataframe
    :param test_df: test dataframe
    :param full_col: List of exogenous variables
    :param endog: Endogenous variable
    """

    best, best_eval = pop[0], generic_func(
        pop[0], train_df, test_df, endog=endog
    )
    for _ in tqdm(range(n_iter)):
        scores = []
        for c in pop:
            scores.append(generic_func(c, train_df, test_df, endog=endog))

        for i in range(len(pop)):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(f'New best: {best_eval}')

        children = []
        for i in range(0, len(pop), 2):
            p1, p2 = proportionate_selection(pop, scores), proportionate_selection(pop, scores)
            for c in crossover(p1, p2, r_cross):
                children.append(mutation(c, r_mut, full_col))
        pop = children
    return [best, best_eval]
