from typing import List, Tuple

import pandas as pd
import statsmodels.api as sm


def estimate_ff3(monthly_returns: pd.DataFrame, left_hand_side_variable: str, right_hand_side_variables: List,
                 ticker_column: str, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = monthly_returns.copy()
    pvalues = []
    betas = []
    ticker_based_groups = df.groupby([ticker_column])

    for ticker, data in ticker_based_groups:
        Y = data.loc[:, left_hand_side_variable]
        X = data.loc[:, right_hand_side_variables]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()

        betas.append(model.params)
        pvalues.append(model.params)

    rhs_vars = ['const', *right_hand_side_variables]

    beta_df = pd.DataFrame(betas, columns=rhs_vars, index=df[ticker_column].unique())
    beta_df.index.name = ticker_column

    pvalues_df = pd.DataFrame(pvalues, columns=rhs_vars, index=df[ticker_column].unique())
    pvalues_df.index.name = ticker_column

    return beta_df, pvalues_df


def estimate_capm(monthly_returns: pd.DataFrame, left_hand_side_variable: str, right_hand_side_variable: str,
                  ticker_column: str, ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = monthly_returns.copy()
    pvalues = []
    betas = []
    ticker_based_groups = df.groupby([ticker_column])

    for ticker, data in ticker_based_groups:
        Y = data.loc[:, left_hand_side_variable]
        X = data.loc[:, right_hand_side_variable]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()

        betas.append(model.params)
        pvalues.append(model.params)

    rhs_vars = ['const', right_hand_side_variable]

    beta_df = pd.DataFrame(betas, columns=rhs_vars, index=df[ticker_column].unique())
    beta_df.index.name = ticker_column

    pvalues_df = pd.DataFrame(pvalues, columns=rhs_vars, index=df[ticker_column].unique())
    pvalues_df.index.name = ticker_column

    return beta_df, pvalues_df


def estimate_factor_exposure(monthly_returns: pd.DataFrame, expected_return: str, factor_betas: List) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    df = monthly_returns.copy()
    lambdas = []
    pvalues = []

    for period in df.index.unique():
        step2 = sm.OLS(endog=df[expected_return].loc[period],
                       exog=df[factor_betas].loc[period]).fit()
        lambdas.append(step2.params)
        pvalues.append(step2.pvalues)

    lambda_pd = pd.DataFrame(lambdas, index=df.index.unique(), columns=factor_betas)
    lambda_pd.index.name = 'Date'

    pvalues_pd = pd.DataFrame(pvalues, index=df.index.unique(), columns=factor_betas)
    pvalues_pd.index.name = 'Date'

    return lambda_pd, pvalues_pd
