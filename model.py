import os, sys
import pandas as pd
import matplotlib

# Qt backend for Mac
if sys.platform == "darwin":
    matplotlib.use('Qt5Agg')

import mplfinance as mpf

"""
load coin data into dataframe
"""
def loadData(coin:str) -> pd.DataFrame:
    root = "./data"

    csvFile = os.path.join(root, coin + "_price.csv")

    with open(csvFile) as csv:
        df = pd.read_csv(csv, index_col=0, parse_dates=True)

    return df

"""
"""
def render(df:pd.DataFrame):

    kwargs = dict(type='candle', style="charles")
    mpf.plot(df, **kwargs)


df_bitcoin = loadData("qtum")
render(df_bitcoin)

