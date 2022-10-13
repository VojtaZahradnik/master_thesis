import pandas as pd
import statsmodels.regression.linear_model

from src.modules import conf, log
"""
Compute prediction for testing dataframe
"""


def input_cli():
    """
    Solution for CLI input without __init__
    :param conf: Configuration dictionary
    """

    data = fit.load_pcls(
        conf["Athlete"]["name"],
        conf["Athlete"]["activity_type"],
        path_to_load=conf["Paths"]["pcl"],
    )
    log.info("Pickles loaded")
    if conf["Athlete"]["test_activity"] != "":
        index = fit.get_race_index(data, conf["Athlete"]["test_activity"])
        train_df = pd.concat(data[0 : index - 1])
        test_df = data[index]
    else:
        train_df = pd.concat(data[0 : len(data) - 1])
        test_df = data[-1]
    train_df, test_df = fit.clean_data(train_df), fit.clean_data(test_df)
    log.info("Training and testing dataframe created")
    import src.heuristics.random_shooting as random_shooting

    result = spec.ols_form(
        train_df, random_shooting.get_form(train_df, conf["endo_var"])
    )
    log.info("Model trained")
    ypred = pred.predict(test_df, result)
    log.info("Prediction provided")
    evl.save_figure(
        true=test_df[conf["endo_var"]],
        pred=ypred,
        name=f'{conf["Athlete"]["name"]}_prediction',
        path_to_save=conf["Paths"]["output"],
    )
    log.info("Img saved into output")


def predict(
    test_df: pd.DataFrame,
    result: statsmodels.regression.linear_model.RegressionResultsWrapper,
) -> pd.Series:
    """
    Predict value based on training dataset and trained model
    :param test_df: testing dataframe
    :param result: learned model from training dataset
    :return: predicted values based on learned model
    """
    return result.predict(test_df)
