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
    df['UpperDiff'] = abs(df['30 upper band'] - df['Close'])
    df['LowerDiff'] = abs(df['30 lower band'] - df['Close'])
    df['Crossover'] = translate(df['Crossover'], df['Crossover'].min(), df['Crossover'].max(), df['Close'].min(), df['Close'].max())
    df['CrossDiff'] = df['Crossover'] - df['Close']
    return df

def load_extract(cryptocurrency):
    df = pd.read_csv(f'data/{cryptocurrency}_price.csv')
    for i,row in df['Date'].iteritems():
        if int(df['Date'][i][-1]) < 6:
            df = df.drop(i)
    df = df.reindex(index=df.index[::-1]) 
    df = MACD_Bands(df)
    df['Volume'] = df['Volume'].copy().replace("-", "0")
    df['Volume'] = df['Volume'].str.replace(',', '').astype('int64')
    df = df.drop(['Date','Market Cap','Open','High','Low'], axis=1)
    df.reset_index(drop=True, inplace=True)
    return df

df_btc = load_extract('bitcoin') #loads bitcoin
df_eth = load_extract('ethereum') #loads ethereum
df_dash = load_extract('dash') #loads dash
df_ltc = load_extract('litecoin') #loads litecoin
df_xmr = load_extract('monero') #loads monero
df_xrp = load_extract('ripple') #loads ripple
