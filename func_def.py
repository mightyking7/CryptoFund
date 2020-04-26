#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 07:02:09 2020

@author: trentc
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

def get_params(Nmodels, Nneurons, dropOut, Nlstm_layers, Ndays, pred_size):
    params = {}
    params["Nneurons"] = np.random.choice(np.arange(Nneurons[0],Nneurons[1]+1), Nmodels)
    params["Nlstm_layers"] = np.random.choice(np.arange(Nlstm_layers[0],Nlstm_layers[1]+1), Nmodels)
    params["dropOut"] = np.random.choice(np.arange(dropOut[0]*10, dropOut[1]*10+1)/10, Nmodels)
    params["Ndays"] = np.round(np.linspace(Ndays[0], Ndays[1], Nmodels))
    params["pred_size"] = np.round(np.linspace(pred_size[0], pred_size[1], Nmodels))
    return params
# EOF #############################################################
###################################################################

def build_model(inShape, output_size, Nneurons, Nlstm_layers, activ_fx, dropOut, loss):
    model = Sequential()

    #print("Adding %s LSTM layers" % Nlstm_layers)
    for k in range(Nlstm_layers):
        returnSeq = k!=Nlstm_layers-1 # false for last LSTM layer
        if k==0: # need to define input size in first layer only
            model.add(LSTM(units = Nneurons, return_sequences=returnSeq,
                           activation=activ_fx, input_shape=inShape))
            model.add(Dropout(dropOut))
        else:
            model.add(LSTM(units = Nneurons, return_sequences=returnSeq,
                           activation=activ_fx))
            model.add(Dropout(dropOut))

    # Adding the output layer
    model.add(Dense(units=output_size, activation="linear"))
    # Compile model
    model.compile(loss=loss, optimizer="adam")
    return model
# EOF #############################################################
###################################################################
    
def get_coin(fpath, Ndays, pred_size, today):
    df = pd.read_csv(fpath)
    Nfeat = df.shape[1]
    #print(Nfeat)
    
    # Scale data for network efficiency
    sc = MinMaxScaler(feature_range = (-1, 1))
    y = df["Close"].copy()
    X = sc.fit_transform(df)
    
    # Create sequential data of size (Ninstances,Ndays,Nfeat)
    N = len(X) - pred_size + 1
    X_seq = np.zeros((N-Ndays, Ndays, Nfeat))
    y_seq = np.zeros((N-Ndays, pred_size))
    
    yNdx = np.arange(pred_size)
    for k in range(Ndays, N):
        X_seq[k-Ndays,:,:] = X[k-Ndays:k,:].reshape((1,Ndays,Nfeat)) # includes today's price
        y_seq[k-Ndays,:] = y[yNdx+k] # tomorrow's price
    #print(X_seq.shape, y_seq.shape)
    
    result = {}
    #print("Ntrain=%s , Ntest=%s" % (Ntrain,Ntest))
    result["X_train"] = X_seq[:today,:]
    result["y_train"] = y_seq[:today,:]
    result["X_test"]  = X_seq[today:,:]
    result["y_test"]  = y_seq[today:,:]
    #print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
    
    return result
    
# EOF #############################################################
###################################################################
    

def load_coins(maxNdays, max_pred_size, coin_names, today):
    # reference as data[coin][model_num]
    data = {}
    for coin in coin_names:
        if coin=="bit": data[coin] = get_coin("test/bitcoin.csv", maxNdays, max_pred_size, today)
        if coin=="dash": data[coin] = get_coin("test/dash.csv", maxNdays, max_pred_size, today)
        if coin=="eth": data[coin] = get_coin("test/ethereum.csv", maxNdays, max_pred_size, today)
        if coin=="lit": data[coin] = get_coin("test/litecoin.csv", maxNdays, max_pred_size, today)
        if coin=="mon": data[coin] = get_coin("test/monero.csv", maxNdays, max_pred_size, today)
        if coin=="rip": data[coin] = get_coin("test/ripple.csv", maxNdays, max_pred_size, today)
    return data
# EOF #############################################################
###################################################################
