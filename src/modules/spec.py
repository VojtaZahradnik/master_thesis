import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import linear_model

"""
Model specification
"""


def ols(
    df: pd.DataFrame, col: list, endog: str
) -> statsmodels.regression.linear_model.RegressionResultsWrapper:
    """
    Performs Ordinary Least Squares (OLS) regression without a formula interface.

    Args:
        df (pd.DataFrame): The training DataFrame.
        col (list): List of column names to be used as exogenous variables.
        endog (str): The name of the column to be used as the endogenous variable.

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: The fitted OLS model.
    """
    df.reset_index()
    x = df[col].astype(float)
    y = df[endog].astype(float)

    return sm.OLS(y, x).fit()


def ols_form(
    df: pd.DataFrame, form: str
) -> statsmodels.regression.linear_model.RegressionResultsWrapper:
    """
    Performs Ordinary Least Squares (OLS) regression using a formula interface.

    Args:
        df (pd.DataFrame): The training DataFrame.
        form (str): The formula string to specify the model.

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: The fitted OLS model.
    """
    return smf.ols(form, data=df).fit()


def sklearn(
    df: pd.DataFrame, col: list,  endog: str
) -> linear_model.LinearRegression:
    """
    Performs Ordinary Least Squares (OLS) regression using the scikit-learn library.

    Args:
        df (pd.DataFrame): The training DataFrame.
        col (list): List of column names to be used as exogenous variables.
        endog (str): The name of the column to be used as the endogenous variable.

    Returns:
        linear_model.LinearRegression: The fitted Linear Regression model from scikit-learn.
    """
    df.reset_index()
    lr = linear_model.LinearRegression()

    x = df[col].astype(float)
    y = df[endog].astype(float)

    return lr.fit(x, y)


def lasso(
    df: pd.DataFrame, col: list, endog: str, alpha=0.1,
) -> linear_model.Lasso:
    """
    Performs Lasso regression using the scikit-learn library.

    Args:
        df (pd.DataFrame): The training DataFrame.
        col (list): List of column names to be used as exogenous variables.
        endog (str): The name of the column to be used as the endogenous variable.
        alpha (float, optional): The constant that multiplies the L1 term. Default is 0.1.

    Returns:
        linear_model.Lasso: The fitted Lasso Regression model from scikit-learn.
    """
    df.reset_index()
    lar = linear_model.Lasso(alpha)

    x = df[col].astype(float)
    y = df[endog].astype(float)

    return lar.fit(x, y)
