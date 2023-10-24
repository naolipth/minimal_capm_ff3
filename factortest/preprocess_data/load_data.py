import pandas as pd
from pathlib import Path
from datetime import datetime

import yfinance as yf
from getFamaFrenchFactors import famaFrench3Factor

from data_cache import pandas_cache


def load_sp500_tickers() -> pd.DataFrame:
    data_folder = Path.cwd().parent.joinpath('data')
    df: pd.DataFrame

    if data_folder.joinpath('S&P500-Symbols.csv').is_file():
        df = pd.read_csv(data_folder.joinpath('S&P500-Symbols.csv'))
        df = df['Symbol']
    else:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        all_data = table[0]
        all_data.to_csv(data_folder.joinpath('S&P500-Symbols.csv'), columns=['Symbol'], index=False)
        df = all_data[['Symbol']]

    return df


@pandas_cache
def fetch_data_from_yahoo(tickers: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    data = yf.download(tickers.values.tolist(), start=start_date, end=end_date)
    df = data.stack().reset_index().rename(index=str, columns={"level_1": "Symbol"}).sort_values(['Symbol', 'Date'])
    df.set_index('Date', inplace=True)
    df = df[['Symbol', 'Adj Close']]
    return df


@pandas_cache
def fetch_fama_french_monthly_factors() -> pd.DataFrame:
    df = famaFrench3Factor(frequency='m')
    df.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)
    df['RF'] = df['RF'] * 10
    return df


def generate_monthly_returns_from_daily_price_data(daily_stock_returns: pd.DataFrame, price_column: str, ticker_column: str, date_column: str) -> pd.DataFrame:

    df = daily_stock_returns.copy()
    data = df.groupby([ticker_column])
    data = data[price_column].resample('M').last().pct_change()
    data = data.reset_index(level=date_column)
    data = data.rename(columns={price_column: 'Monthly Returns'})

    data_extended_with_monthly_returns = df.merge(data, how= 'left', on= [date_column, ticker_column])
    data_extended_with_monthly_returns = data_extended_with_monthly_returns.set_index(date_column)
    data_extended_with_monthly_returns = data_extended_with_monthly_returns.drop(columns=price_column).dropna()

    return data_extended_with_monthly_returns
