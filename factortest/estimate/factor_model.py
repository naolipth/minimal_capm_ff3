from typing import List
import statsmodels.api as sm

import pandas as pd


def estimate_ff3(monthly_returns: pd.DataFrame, left_hand_side_variable: str, right_hand_side_variables: List,
                 ticker_column: str) -> pd.DataFrame:
    params = {}
    ticker_based_groups = monthly_returns.groupby([ticker_column])

    for ticker, data in ticker_based_groups:
        Y = data.loc[:, left_hand_side_variable]
        X = data.loc[:, right_hand_side_variables]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        temp_dict = {ticker: {'intercept coef': model.params[0], 'intercept p-value': model.pvalues[0],
                              right_hand_side_variables[0] + ' coef': model.params[1],
                              right_hand_side_variables[0] + ' p-value': model.pvalues[1],
                              right_hand_side_variables[1] + ' coef': model.params[2],
                              right_hand_side_variables[1] + ' p-value': model.pvalues[2],
                              right_hand_side_variables[2] + ' coef': model.params[3],
                              right_hand_side_variables[2] + ' p-value': model.pvalues[3]}}

        params.update(temp_dict)
    params_df = pd.DataFrame.from_dict(params, orient='index')
    return params_df


def estimate_capm(monthly_returns: pd.DataFrame, left_hand_side_variable: str, right_hand_side_variable: str,
                 ticker_column: str) -> pd.DataFrame:
    params = {}
    ticker_based_groups = monthly_returns.groupby([ticker_column])

    for ticker, data in ticker_based_groups:
        Y = data.loc[:, left_hand_side_variable]
        X = data.loc[:, right_hand_side_variable]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        temp_dict = {ticker: {'intercept coef': model.params[0], 'intercept p-value': model.pvalues[0],
                              right_hand_side_variable + ' coef': model.params[1],
                              right_hand_side_variable + ' p-value': model.pvalues[1]}}

        params.update(temp_dict)

    params_df = pd.DataFrame.from_dict(params, orient='index')
    return params_df

