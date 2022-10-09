import random
from typing import Any, List

import numpy as np
import pandas as pd
from numpy.random import randint, uniform
from tqdm import tqdm

from src.heuristics.random_shooting import get_form
from src.modules.evl import calc_rmse
from src.modules.predict import predict
from src.modules.spec import ols_form

"""
Module for GO
"""


def selection(pop, scores, k=3):
    selection_ix = randint(len(pop))

    for ix in randint(0, len(pop), k - 1):
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
        # break_?
    return pop[selection_ix]


def select_parent(pop, scores):
    """
    Selection procedure that returns selected parent
    :param pop: population
    :param scores: metrics that determines quality of generation
    :return: selected parents
    """
    parent = []
    total = sum(scores)
    norm_fitness_values = [x / total for x in scores]
    cumulative_fitness = []

    start = 0
    for norm_value in norm_fitness_values:
        start += norm_value
        cumulative_fitness.append(start)

    individual_number = 0
    for score in cumulative_fitness:
        if uniform(0, 1) <= score:
            parent = pop[individual_number]
            break
        individual_number += 1

    return parent


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
    return calc_rmse(test_df[endog], pred_ols)


def mutation(bitstring: list, r_mut: int, full_form: list):
    """
    This procedure simply flips bits with a low probability controlled by the “r_mut”
    :param bitstring: list of columns for formula
    :param r_mut: hyperparameter determines the amount of mutation
    :param full_form: maximal possible formula in list
    """
    # upravit
    for i in range(len(bitstring)):
        if np.random.rand() < r_mut:
            new_bitstring = full_form[randint(0, len(full_form))]
            if new_bitstring not in bitstring:
                bitstring.append(new_bitstring)
            else:
                bitstring.pop(i)
            break
    return bitstring


def crossover(p1: list, p2: list, r_cross: int) -> list:
    """
    Function that implementing crossover between two population
    :param p1: first parent
    :param p2: second parent
    :param r_cross: crossover rate that determines whether crossover is performed or not
    :return: list of childrens
    """
    c1, c2 = p1.copy(), p2.copy()
    r_cross = int(r_cross * 100)
    if random.randint(0, 100) < r_cross:
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
    col: list,
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
    :param col: List of endogenous variables
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
            p1, p2 = selection(pop, scores), selection(pop, scores)
            for c in crossover(p1, p2, r_cross):
                children.append(mutation(c, r_mut, col))
        pop = children
    return [best, best_eval]
