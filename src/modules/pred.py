import pandas as pd
import statsmodels.regression.linear_model

from src.modules import conf, log
"""
Compute prediction for testing dataframe
"""



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
