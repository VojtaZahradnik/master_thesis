import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import seaborn as sns

from io import BytesIO
import base64

from src.modules import evl, fit, spec, conf, log

"""
Model evaluation with basic metrics of statistics
"""

EPSILON = 1e-10


def error(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    """
    Calculation of simple rror
    :param actual: True values
    :param predicted: Predicted values
    :return: Error/residuum
    """
    return actual - predicted

def percentage_error(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    """
    Method to return series of percentage errors
    :param actual: True values
    :param predicted: Predicted values
    :return: Percentage of errors
    """
    return error(actual, predicted) / (actual + EPSILON)

def mse(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Mean square error metric
    :param actual: True values
    :param predicted: Predicted values
    :return: MSE number
    """
    return np.mean(np.square(error(actual, predicted)))

def rmse(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Root mean square error metric
    :param actual: True values
    :param predicted: Predicted values
    :return: RMSE number
    """
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Normalized root mean square error metric
    :param actual: True values
    :param predicted: Predicted values
    :return: N number
    """
    return rmse(actual, predicted) / (actual.max() - actual.min())


def me(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Mean error
    :param actual: True values
    :param predicted: Predicted values
    :return: ME number
    """
    return np.mean(error(actual, predicted))


def mae(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Mean absolute error
    :param actual: True values
    :param predicted: Predicted values
    :return: MAE number
    """
    return np.mean(np.abs(error(actual, predicted)))

def mdae(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Median absolute error
    :param actual: True values
    :param predicted: Predicted values
    :return: MDAE number
    """
    return np.median(np.abs(error(actual, predicted)))


def mpe(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Mean percentage error
    :param actual: True values
    :param predicted: Predicted values
    :return: MPE number
    """
    return np.mean(percentage_error(actual, predicted))


def mape(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Mean absolute percentage error
    :param actual: True values
    :param predicted: Predicted values
    :return: MAPE number
    """
    return np.mean(np.abs(percentage_error(actual, predicted)))

def rrse(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Root relative squared error
    :param actual: True values
    :param predicted: Predicted values
    :return: RRSE
    """
    return np.sqrt(np.sum(np.square(actual - predicted)) /
                   np.sum(np.square(actual - np.mean(actual))))

def evaluate(actual: pd.Series, predicted: pd.Series,
             metrics=('rmse','nrmse','mape')):
    results = {}
    for name in metrics:
        try:
            results[name] = METRICS[name](actual, predicted)
        except Exception as err:
            results[name] = np.nan
            log.error(f'Unable to compute metric {name}: {err}')
    return results

METRICS = {
    'RMSE': rmse,
    'NRMSE': nrmse,
    'MAPE': mape,
}

def eval_all(actual: pd.Series, predicted: pd.Series):
    return evaluate(actual, predicted, metrics=set(METRICS.keys()))

def plot(df: pd.DataFrame, pred: list, ylabel:str,color:str,true_data= [], spline=False, lwidth=1):
    """
    Plot graph with difference between true values and predicted values.
    :param df: tested dataframe
    :param pred: predicted dataframe
    :param spline: boolean variable determine to use spline or not
    :param lwidth: line width
    """
    fig, ax1 = plt.subplots(figsize=(24, 8))
    color = f"tab:{color}"
    ax1.set_xlabel("Distance")
    ax1.set_ylabel(ylabel, color=color)
    pred = pred[1:]
    x = np.linspace(min(df.distance)/1000,max(df.distance)/1000, len(pred))

    spl = scipy.interpolate.UnivariateSpline(x, df.enhanced_altitude[1:])
    altitude = spl(x)

    if true_data != []:
        ax1.plot(x, true_data[1:], color, label="True data")
        ax1.tick_params(axis="y", labelcolor=color)

    if spline:
        spl = scipy.interpolate.UnivariateSpline(x, pred)
        pred = spl(x)
    ax1.plot(x, pred, color, label=ylabel, linewidth=lwidth)
    ax1.plot(x,np.linspace(np.mean(pred),np.mean(pred),len(pred)),label="Mean")

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Altitude", color=color)
    # ax2.plot(x, test_df.enhanced_altitude, 'y',label = 'alt')
    ax2.plot(x, altitude, "y", label="Altitude")
    ax2.tick_params(axis="y", labelcolor=color)

    sns.set(style="ticks")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return plot


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


def plot_compare(df: pd.DataFrame, pred1: list, pred2: list, ylabel="Speed", color1="blue", color2="red", spline=False,
                 lwidth=1):
    """
    Plot graph with difference between true values and predicted values.
    :param df: tested dataframe
    :param pred1: predicted values for the first dataframe
    :param pred2: predicted values for the second dataframe
    :param lwidth: line width
    """
    fig, ax1 = plt.subplots(figsize=(24, 8))
    ax1.set_xlabel("Distance")
    ax1.set_ylabel(ylabel, color=color1)

    x = np.linspace(min(df.distance) / 1000, max(df.distance) / 1000, len(pred1))

    # Interpolate using spline for the first dataframe
    spl1 = scipy.interpolate.UnivariateSpline(x, df.enhanced_altitude)
    altitude1 = spl1(x)

    ax1.plot(x, pred1, "r", label=conf["Athlete"]["name"], linewidth=lwidth)

    ax1.plot(x, pred2, "b", label="Reference Athlete", linewidth=lwidth)

    ax2 = ax1.twinx()
    color2 = "tab:" + color2
    ax2.set_ylabel("Altitude", color=color2)
    ax2.plot(x, altitude1, "y", label="Altitude 1")
    ax2.tick_params(axis="y", labelcolor=color2)

    sns.set(style="ticks")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return plot
