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
    """"
    Calculates the error between actual and predicted values.

    Args:
        actual (pd.Series): Series of actual values.
        predicted (pd.Series): Series of predicted values.

    Returns:
        pd.Series: The calculated error for each data point.
    """
    return actual - predicted

def percentage_error(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    """
    Calculates the percentage error between actual and predicted values.

    Args:
        actual (pd.Series): Series of actual values.
        predicted (pd.Series): Series of predicted values.

    Returns:
        pd.Series: The calculated percentage error for each data point.
    """
    return error(actual, predicted) / (actual + EPSILON)

def mse(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Calculates the mean squared error between actual and predicted values.

    Args:
        actual (pd.Series): Series of actual values.
        predicted (np.ndarray): Array of predicted values.

    Returns:
        float: The mean squared error.
    """
    return np.mean(np.square(error(actual, predicted)))

def rmse(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Calculates the root mean squared error between actual and predicted values.

    Args:
        actual (pd.Series): Series of actual values.
        predicted (np.ndarray): Array of predicted values.

    Returns:
        float: The root mean squared error.
    """
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Calculates the normalized root mean squared error between actual and predicted values.

    Args:
        actual (pd.Series): Series of actual values.
        predicted (np.ndarray): Array of predicted values.

    Returns:
        float: The normalized root mean squared error.
    """
    return rmse(actual, predicted) / (actual.max() - actual.min())


def me(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Calculates the mean error between actual and predicted values.

    Args:
        actual (pd.Series): Series of actual values.
        predicted (np.ndarray): Array of predicted values.

    Returns:
        float: The mean error.
    """
    return np.mean(error(actual, predicted))


def mae(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Calculates the mean absolute error between actual and predicted values.

    Args:
        actual (pd.Series): Series of actual values.
        predicted (np.ndarray): Array of predicted values.

    Returns:
        float: The mean absolute error.
    """
    return np.mean(np.abs(error(actual, predicted)))

def mdae(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Calculates the median absolute error between actual and predicted values.

    Args:
        actual (pd.Series): Series of actual values.
        predicted (np.ndarray): Array of predicted values.

    Returns:
        float: The median absolute error.
    """
    return np.median(np.abs(error(actual, predicted)))


def mpe(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Calculates the mean percentage error between actual and predicted values.

    Args:
        actual (pd.Series): Series of actual values.
        predicted (np.ndarray): Array of predicted values.

    Returns:
        float: The mean percentage error.
    """
    return np.mean(percentage_error(actual, predicted))


def mape(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Calculates the mean absolute percentage error between actual and predicted values.

    Args:
        actual (pd.Series): Series of actual values.
        predicted (np.ndarray): Array of predicted values.

    Returns:
        float: The mean absolute percentage error.
    """
    return np.mean(np.abs(percentage_error(actual, predicted)))

def rrse(actual: pd.Series, predicted: np.ndarray) -> int:
    """
    Calculates the root relative squared error between actual and predicted values.

    Args:
        actual (pd.Series): Series of actual values.
        predicted (np.ndarray): Array of predicted values.

    Returns:
        float: The root relative squared error.
    """
    return np.sqrt(np.sum(np.square(actual - predicted)) /
                   np.sum(np.square(actual - np.mean(actual))))

def evaluate(actual: pd.Series, predicted: pd.Series,
             metrics=('rmse','nrmse','mape')) -> dict:
    """
    Evaluates the prediction error using various metrics.

    Args:
        actual (pd.Series): Series of actual values.
        predicted (pd.Series): Series of predicted values.
        metrics (tuple, optional): Metrics to be used for evaluation. Defaults to ('rmse', 'nrmse', 'mape').

    Returns:
        dict: A dictionary with metric names as keys and computed metric values as values.
    """
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

def eval_all(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Evaluates the prediction error using all available metrics.

    Args:
        actual (pd.Series): Series of actual values.
        predicted (pd.Series): Series of predicted values.

    Returns:
        dict: A dictionary with all metric names as keys and computed metric values as values.
    """
    return evaluate(actual, predicted, metrics=set(METRICS.keys()))

def plot(df: pd.DataFrame, pred: list, ylabel:str,color:str,true_data= [], spline=False, lwidth=1) -> None:
    """
    Plots a comparison graph between two sets of predicted values.

    Args:
        df (pd.DataFrame): The dataframe containing the test data.
        pred1 (list): The first set of predicted values.
        pred2 (list): The second set of predicted values.
        ylabel (str, optional): The label for the y-axis. Defaults to "Speed".
        color1 (str, optional): The color for the first set of predictions. Defaults to "blue".
        color2 (str, optional): The color for the second set of predictions. Defaults to "red".
        spline (bool, optional): A flag to indicate whether to use spline interpolation or not. Defaults to False.
        lwidth (int, optional): The line width for the plot. Defaults to 1.

    Returns:
        str: A base64-encoded string of the comparison plot image.
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
    Saves a plotted figure as an image file.

    This function creates a plot using prediction data and, optionally, true data. The plot is then saved as an image file
    in the specified path. The distance data is used for the x-axis, and it is converted to kilometers. The plot includes
    lines for both prediction and true data (if provided).

    Args:
        name (str): The name of the image file to be saved.
        path_to_save (str): The path where the image file will be saved.
        pred (pd.Series): The series containing prediction values.
        dist (pd.Series): The series containing distance values, used for the x-axis of the plot.
        true (list, optional): A list containing true values for comparison. Defaults to an empty list.

    Returns:
        matplotlib.figure.Figure: The figure object that was saved as an image.
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
    Plots a comparison graph between two sets of predicted values against a common dataframe.

    This function creates a subplot to compare two different sets of predicted values, typically representing two different models
    or approaches. The subplot includes two y-axes: one for the predicted values and another for the altitude from the dataframe.
    Spline interpolation can be optionally applied for smoothing the lines.

    Args:
        df (pd.DataFrame): The dataframe containing the common data used for comparison, including distance and altitude.
        pred1 (list): The first set of predicted values.
        pred2 (list): The second set of predicted values.
        ylabel (str, optional): The label for the y-axis representing the predicted values. Defaults to "Speed".
        color1 (str, optional): The color for the first set of prediction line. Defaults to "blue".
        color2 (str, optional): The color for the second set of prediction line. Defaults to "red".
        spline (bool, optional): Indicates whether to use spline interpolation for smoothing lines. Defaults to False.
        lwidth (int, optional): The line width for the plot lines. Defaults to 1.

    Returns:
        str: A base64-encoded string of the comparison plot image.
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
