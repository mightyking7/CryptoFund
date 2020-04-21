import os, sys
import pandas as pd
import matplotlib

# Qt backend for Mac
if sys.platform == "darwin":
    matplotlib.use('Qt5Agg')

import mplfinance as mpf


class CoinGraph:

    def __init__(self, coin:str):

        self.coin = coin.lower() # coin name
        self.root = "./data"
        self.df = None # data frame of prices

    """
    load coin data into dataframe
    """
    def loadCoinData(self) -> pd.DataFrame:

        # initial csv read
        if type(self.df) != pd.DataFrame:

            csvFile = os.path.join(self.root, self.coin + "_price.csv")

            with open(csvFile) as csv:
                self.df = pd.read_csv(csv, index_col=0, parse_dates=True)

        return self.df

    """
    """
    def render(self):

        if type(self.df) != pd.DataFrame:
            print(f"Error: Must load data for {self.coin} before usage")
            return

        mpf.plot(self.df, type="candle", style="charles",
                          title=self.coin.title(), ylabel_lower="Volume")


cg = CoinGraph("Bitcoin")
df_bitcoin = cg.loadCoinData()
cg.render()
