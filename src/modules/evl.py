import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import seaborn as sns

from src.modules import evl, fit, predict, spec

"""
Model evaluation with basic metrics of statistics
"""


def input_cli(conf: dict, log: logging.Logger):
    """
    Solution for CLI input without __init__
    :param conf: Configuration dictionary
    """

    data = fit.load_pcls(
        conf["Athlete"]["name"],
        conf["Athlete"]["activity_type"],
        conf["Paths"]["pcl"],
    )

    if conf["Athlete"]["test_activity"] != "":
        index = fit.get_race_index(data, conf["Athlete"]["test_activity"])
        train_df = pd.concat(data[0 : index - 1])
        test_df = data[index]
    else:
        train_df = pd.concat(data[0 : len(data) - 1])
        test_df = data[-1]

    train_df, test_df = fit.clean_data(train_df), fit.clean_data(test_df)
    from src.heuristics import random_shooting

    result = spec.ols_form(
        train_df, random_shooting.get_form(train_df, "enhanced_speed")
    )
    ypred = predict.predict(test_df, result)
    evl.save_figure(
        true=test_df[conf["endo_var"]],
        pred=ypred,
        name=f'{conf["Athlete"]["name"]}_prediction',
        path_to_save=conf["Paths"]["output"],
    )
    log.info("Img saved into output")
    log.info(f"RMSE: {calc_rmse(test_df.enhanced_speed, ypred)}")


def press_statistic(
    y_true: pd.DataFrame,
    y_pred: pd.Series,
    xs: pd.DataFrame,
) -> int:
    """
    Predicted residual error sum of squares
    :param y_true: true values of activity
    :param y_pred: predicted values of activity
    :param xs: whole testing dataframe
    :return: sum of the squares of all the resulting prediction errors
    """
    den = 1 - np.diagonal(xs.dot(np.linalg.pinv(xs)))
    sqr = np.square(y_pred - y_true / den)

    return sqr.sum()


def calc_predicted_r2(
    df: pd.DataFrame,
    pred: list,
    endog="enhanced_speed",
) -> int:
    """
    Calculation of predicted R squared
    :param df: tested dataframe
    :param pred: predicted dataframe
    :param endog: endogenous variable
    :return: R squared value
    """
    press = press_statistic(y_true=df[endog], y_pred=pred, xs=df)

    sst = np.square(df[endog] - df[endog].mean()).sum()

    return 1 - press / sst


def calc_r2(
    true_data: list,
    pred: list,
) -> int:
    """
    Calculation of normal adjusted R squared
    :param true_data: tested dataframe
    :param pred: predicted dataframe
    :return: adjusted R squared
    """
    sse = np.square(
        [(element1 - element2) for (element1, element2) in zip(true_data, pred)]
    ).sum()
    sst = np.square(true_data - np.mean(pred)).sum()

    return 1 - sse / sst


def calc_rmse(
    true_data: pd.Series,
    pred: pd.Series,
) -> int:
    """
    Calculation of root mean squared error
    :param true_data: tested list
    :param pred: predicted list
    :return: differences between values predicted by a model (RMSE)
    """
    n = len(true_data)
    return np.sqrt(
        1
        / n
        * np.sum(
            [
                (element1 - element2) ** 2
                for (element1, element2) in zip(pred, true_data)
            ]
        )
    )


def calc_std_dev(
    true_data: list,
    pred: list,
) -> int:
    """
    Calculation of standard deviation
    :param true_data: tested list
    :param pred: predicted list
    :return:  measure of the amount of variation (standard deviation)
    """
    return (np.std(true_data) - np.std(pred)) ** 2


def calc_mse(
    true_data: list,
    pred: list,
) -> int:
    """
    Calculation of mean squared error
    :param true_data: tested list
    :param pred: predicted list
    :return: average of the squares of the errors (MSE)
    """
    return (
        1
        / len(true_data)
        * np.sum(
            [
                (element1 - element2) ** 2
                for (element1, element2) in zip(pred, true_data)
            ]
        )
    )


def calc_mae(
    true_data: list,
    pred: list,
) -> int:
    """
    Calculation of mean absolute error
    :param true_data: tested list
    :param pred: predicted list
    :return: errors between paired observations
    """
    return (
        1
        / len(true_data)
        * np.sum(
            np.abs(
                [(element1 - element2) for (element1, element2) in zip(pred, true_data)]
            )
        )
    )


def calc_residual(
    true_data: list,
    pred: list,
) -> int:
    """
    Calculation of basic residual metric
    :param true_data: tested list
    :param pred: predicted list
    :return: sum of residual divided by number of observations
    """
    return np.sum(
        np.abs([(element1 - element2) for (element1, element2) in zip(true_data, pred)])
    ) / len(true_data)


def whole_eval(
    true_data: list,
    pred: list,
):
    """
    Print some basic evaluations of model
    :param true_data: tested dataframe
    :param pred: predicted dataframe
    """
    print(f"RMSE: {calc_rmse(true_data, pred)}")
    print(f"R2: {calc_r2(true_data, pred)}")
    # print(f'Predicted R2: {calc_predicted_r2(true_data, pred)}')
    print(f"Std. dev.: {calc_std_dev(true_data, pred)}")
    print(f"MSE: {calc_mse(true_data, pred)}")
    print(f"MAE: {calc_mae(true_data, pred)}")
    print(f"Residual metric: {calc_residual(true_data, pred)}")


def plot(df: pd.DataFrame, pred: list, endog="enhanced_speed", spline=False, lwidth=1):
    """
    Plot graph with difference between true values and predicted values.
    :param df: tested dataframe
    :param pred: predicted dataframe
    :param spline: boolean variable determine to use spline or not
    :param lwidth: line width
    """
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Speed", color=color)
    pred = pred[1:]
    x = np.linspace(0, 20, len(pred))

    spl = scipy.interpolate.UnivariateSpline(x, df.enhanced_altitude[1:])
    altitude = spl(x)

    ax1.plot(x, df[endog][1:], "b", label="True data")
    ax1.tick_params(axis="y", labelcolor=color)

    if spline:
        spl = scipy.interpolate.UnivariateSpline(x, pred)
        pred = spl(x)
    ax1.plot(x, pred, "r", label="Prediction", linewidth=lwidth)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Altitude", color=color)
    # ax2.plot(x, test_df.enhanced_altitude, 'y',label = 'alt')
    ax2.plot(x, altitude, "y", label="Altitude")
    ax2.tick_params(axis="y", labelcolor=color)

    sns.set(style="ticks")
    ax1.legend(loc="upper left")
    ax2.legend(loc="best")

    return fig


def save_figure(
    name: str, path_to_save: str, pred: pd.Series, dist: pd.Series, true=[]
):
    """
    Save figure of plot to img
    :param true list with true values
    :param pred: list with prediction
    :param name: name of the imgs file
    :param path_to_save: save path of img file
    """
    p = Path(os.getcwd())
    if "output" not in os.listdir(os.path.join(p.parent, p.parent)):
        os.mkdir(os.path.join(p.parent, "output"))

    plt.rcParams["agg.path.chunksize"] = 10000
    dist = dist / 1000
    x = np.linspace(0, np.max(dist), len(dist))
    fig = plt.figure()
    fig.set_size_inches(24, 8)

    if len(true) != 0:
        plt.plot(x, true, "r", label="True data")
        name += "_test"
    plt.plot(x, pred, "g", label="Prediction")
    plt.legend(loc="best")
    plt.xlabel("Kilometers")
    plt.ylabel("Km/h")
    plt.savefig(
        os.path.join(path_to_save + name + ".png"), bbox_inches="tight", dpi=600
    )
    plt.close(
        fig,
    )

    return fig
