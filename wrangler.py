# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 02:18:09 2020

@author: techm
crypto kaggle dataset for data mining project
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv('./data/bitcoin_dataset.csv')

for i, row in df['btc_market_price'].iteritems():
    if df['btc_market_price'][i] == 0:
        df = df.drop(i)

corr = df.select_dtypes(include = ['float64', 'int64']).corr()
f, ax = plt.subplots(figsize=(14, 5))
ax = sns.heatmap(corr[(corr >= 0.9) | (corr <= -0.9)], cmap='viridis', annot=True, annot_kws={"size": 10})
df_3 = df[['btc_hash_rate','btc_difficulty','btc_estimated_transaction_volume_usd']].copy()

X = np.array(df_3)
y = np.array(df[['btc_market_price']].copy()).ravel()
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#lassoedCV = LassoCV(alphas=None, cv=10, max_iter=10000).fit(X_train,y_train)
#y_predict = lassoedCV.predict(X_test)

#model = XGBClassifier(learning_rate=0.02, n_estimators=750, max_depth= 3, min_child_weight= 1, colsample_bytree= 0.6, gamma= 0.0, reg_alpha= 0.001, subsample= 0.8)
model = XGBClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

accmat = pd.DataFrame(data=np.column_stack((y_predict, y_test)))
accmat.columns = ['y_predict','y_test']
for j, row in accmat['y_predict'].iteritems():
    if accmat['y_predict'][j] == 0:
        accmat = accmat.drop(j)

#accscore = mean_squared_error(y_predict, y_test)
accavgpercent = np.mean((y_predict-y_test)/y_test)