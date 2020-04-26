# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:41:34 2020

@author: techm
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 19:16:24 2020

@author: techm
"""
import pandas as pd
import numpy as np


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = (value - leftMin) / leftSpan

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


#MACD indicator to identify the trend and the Bollinger Bands as a trade trigger, volatility, Crossover Signal
def MACD_Bands(df):
    df.fillna(method="bfill", inplace=True)
    df['Volatility'] = (df['High'] - df['Low'])/df['Close']
    df['30 mavg'] = df['Close'].copy().rolling(30).mean()
    df['30 std'] = df['Close'].copy().rolling(30).std()
    df['30 upper band'] = df['30 mavg'] + (df['30 std']*2)
    df['30 lower band'] = df['30 mavg'] - (df['30 std']*2)
    df['26 ema'] = df['Close'].copy().ewm(span=26).mean()
    df['12 ema'] = df['Close'].copy().ewm(span=12).mean()
    df['MACD'] = (df['12 ema'] - df['26 ema'])
    df['Signal'] = df['MACD'].copy().ewm(span=9).mean()
    df['Crossover'] = df['MACD'] - df['Signal']
    df['UpperDiff'] = df['30 upper band'] - df['Close']
    df['UpperGrad'] = df['30 upper band'].copy()  # Just a temporary placeholder to get the shape for modifying values
    df['LowerDiff'] = df['Close'] - df['30 lower band']
    df['LowerGrad'] = df['30 lower band'].copy()  # Just a temporary placeholder to get the shape for modifying values
    df['Crossover'] = translate(df['Crossover'], df['Crossover'].min(), df['Crossover'].max(), df['Close'].min(),
                                df['Close'].max())
    df['CrossDiff'] = df['Crossover'] - df['Close']
    df['CrossGrad'] = df['Crossover'].copy()  # Just a temporary placeholder to get the shape for modifying values
    for i in range(len(df['30 upper band'])):
        if i == len(df) - 1:
            df.loc[i, 'UpperGrad'] = df.loc[i - 1, 'UpperGrad']
            df.loc[i, 'LowerGrad'] = df.loc[i - 1, 'LowerGrad']
            df.loc[i, 'CrossGrad'] = df.loc[i - 1, 'CrossGrad']
        else:
            df.loc[i, 'UpperGrad'] = df['30 upper band'][i + 1] - df['30 upper band'][i]
            df.loc[i, 'LowerGrad'] = df['30 lower band'][i + 1] - df['30 lower band'][i]
            df.loc[i, 'CrossGrad'] = df['Crossover'][i + 1] - df['Crossover'][i]
    return df

def load_extract(cryptocurrency):
    df = pd.read_csv(f'testApril/{cryptocurrency}.csv')
#    for i,row in df['Date'].iteritems():
#        if int(df['Date'][i][-1]) < 6:
#            df = df.drop(i)
    df = df.reindex(index=df.index[::-1]) 
    df = MACD_Bands(df)
    #df['Volume'] = df['Volume'].copy().replace("-", "0")
    #df['Volume'] = df['Volume'].str.replace(',', '').astype('int64')
    df = df.drop(['Date','High','Low'], axis=1)
    df.reset_index(drop=True, inplace=True)
    df.fillna(method="bfill", inplace=True)
    return df


df_btc = load_extract('bitcoin')
df_eth = load_extract('ethereum')
df_dash = load_extract('dash')
df_ltc = load_extract('litecoin')
df_xmr = load_extract('monero')
df_xrp = load_extract('ripple')

df_btc.to_csv(r'C:\Users\techm\Documents\GitHub\CryptoFund\bitcoin.csv', index=False)
df_eth.to_csv(r'C:\Users\techm\Documents\GitHub\CryptoFund\ethereum.csv', index=False)
df_dash.to_csv(r'C:\Users\techm\Documents\GitHub\CryptoFund\dash.csv', index=False)
df_ltc.to_csv(r'C:\Users\techm\Documents\GitHub\CryptoFund\litecoin.csv', index=False)
df_xmr.to_csv(r'C:\Users\techm\Documents\GitHub\CryptoFund\monero.csv', index=False)
df_xrp.to_csv(r'C:\Users\techm\Documents\GitHub\CryptoFund\ripple.csv', index=False)