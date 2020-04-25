# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:53:48 2020

Wrangling crypto kaggle datasets to include additional features such as Volatility, MACD, Crossover
"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
# Qt backend for Mac
if sys.platform == "darwin":
    matplotlib.use('Qt5Agg')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Time span of data, 01/01/2016 to 02/20/2018
df_full_btc = pd.read_csv('data/bitcoin_dataset.csv')
df_full_btc = df_full_btc.drop(df_full_btc.index[0:2138])

df_btc = pd.read_csv('data/bitcoin_price.csv')
df_btc = df_btc.reindex(index=df_btc.index[::-1])
df_btc = df_btc.drop(df_btc.index[0:978])
#df_full_eth = pd.read_csv('ethereum_dataset.csv')
#df_full_eth = df_full_eth.drop(df_full_eth.index[0:155])
#df_eth = pd.read_csv('ethereum_price.csv')
#df_eth = df_eth.reindex(index=df_eth.index[::-1])
#df_eth = df_eth.drop(df_eth.index[0:147])

df_full_btc = df_full_btc.drop(columns=['Date','btc_market_cap','btc_blocks_size','btc_trade_volume','btc_avg_block_size','btc_n_orphaned_blocks','btc_n_transactions_per_block','btc_median_confirmation_time','btc_cost_per_transaction_percent','btc_n_transactions','btc_n_transactions_excluding_popular','btc_n_transactions_excluding_chains_longer_than_100','btc_output_volume','btc_estimated_transaction_volume'], axis=1)
corr_full_btc = df_full_btc.select_dtypes(include = ['float64', 'int64']).corr()
f0, ax0 = plt.subplots(figsize=(14, 5))
ax0 = sns.heatmap(corr_full_btc[corr_full_btc >= 0.5], cmap='Reds', annot=True, annot_kws={"size": 10})


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
    df['Crossover'] = translate(df['Crossover'], df['Crossover'].min(), df['Crossover'].max(), df['Close'].min(), df['Close'].max())
    df['CrossDiff'] = abs(df['Crossover'] - df['Close'])
    df['CrossGrad'] = df['Crossover'].copy()  # Just a temporary placeholder to get the shape for modifying values
    for i in range(len(df['30 upper band'])):
        if i == len(df)-1:
            df.loc[i, 'UpperGrad'] = df.loc[i-1, 'UpperGrad']
            df.loc[i, 'LowerGrad'] = df.loc[i-1, 'LowerGrad']
            df.loc[i, 'CrossGrad'] = df.loc[i-1, 'CrossGrad']
        else:
            df.loc[i, 'UpperGrad'] = df['30 upper band'][i + 1] - df['30 upper band'][i]
            df.loc[i, 'LowerGrad'] = df['30 lower band'][i + 1] - df['30 lower band'][i]
            df.loc[i, 'CrossGrad'] = df['Crossover'][i + 1] - df['Crossover'][i]
    return df

df_btc_final = MACD_Bands(df_btc)
df_btc_final.fillna(method="bfill", inplace=True)
df_btc_final['Volume'] = df_btc['Volume'].str.replace(',', '').astype('int64')
df_btc_final['Close'] = df_btc['Close'].copy()
df_btc_final['Market Cap'] = df_btc_final['Market Cap'].str.replace(',', '').astype('int64')
df_btc_final = df_btc_final.drop(['Market Cap'], axis=1)

f1, ax = plt.subplots(figsize=(14, 5))
ax = sns.lineplot(data=df_btc_final['Close'], legend='brief', label=str('Close Price'))
ax2 = plt.twinx()
ax.set(xlabel='Time [days]', ylabel='Close Price')
ax2.set(ylabel='Crossover Value')
sns.lineplot(data=df_btc_final['Crossover'], color="r", ax=ax2, legend='brief', label=str('Crossover from Signal'))

ax2.set_xlim([5,85])

df_btc_final = df_btc_final.drop(df_btc_final.iloc[:,0:4], axis=1)

corr_btc_macd = df_btc_final.select_dtypes(include = ['float64', 'int64']).corr()
f2, ax3 = plt.subplots(figsize=(14, 5))
ax3 = sns.heatmap(corr_btc_macd, cmap='Reds', annot=True, annot_kws={"size": 10})

df_full_btc = df_full_btc.drop(columns = ['btc_market_price'])

### Simple Prediction
df_btc_final.reset_index(drop=True, inplace=True)
df_full_btc.reset_index(drop=True, inplace=True)

df_btc_feature_set = pd.concat([df_btc_final, df_full_btc], axis=1)
print(df_btc_feature_set.shape)
X = np.array(df_btc_feature_set.iloc[:,1:])
y = np.array(df_btc_feature_set['Close']).ravel()
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = XGBRegressor()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
acc_mat = pd.DataFrame(data=np.column_stack((y_predict, y_test)))
acc_mat.columns = ['y_predict','y_test']
for j, row in acc_mat['y_predict'].iteritems():
    if acc_mat['y_predict'][j] == 0:
        acc_mat = acc_mat.drop(j)

