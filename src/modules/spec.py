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
    Create model based on ordinary least squares method without formula
    :param df: training dataframe
    :return: trained model
    """
    df.reset_index()
    x = df[col].astype(float)
    y = df[endog].astype(float)

    return sm.OLS(y, x).fit()


def ols_form(
    df: pd.DataFrame, form: str
) -> statsmodels.regression.linear_model.RegressionResultsWrapper:
    """
    Create model based on ordinary least squares method with loss function
    :param df: training dataframe
    :param form: loss function
    :return: trained model
    """
    return smf.ols(form, data=df).fit()


def sklearn(
    df: pd.DataFrame, col: list,  endog: str
) -> linear_model.LinearRegression:
    """
    Create model based on ordinary least squares method from library scikit-learn
    :param df: training dataframe
    :param col: names of columns
    :return: trained model
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
    Create model based on lasso regression from library scikit-learn
    :param df: training dataframe
    :param col: names of columns
    :param alpha: constant that multiplies the L1 term
    :param endog: endogenous variable
    :return: trained model
    """
    df.reset_index()
    lar = linear_model.Lasso(alpha)

    x = df[col].astype(float)
    y = df[endog].astype(float)

    return lar.fit(x, y)
