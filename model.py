import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Control Params #######################################
#   Data Params
test_size = 0.33 # contiguous segments for train & test
Ndays = 10 # number of past days info to predict tomorrow
pred_size = 1 # num days to predict
#   RNN Params
Nneurons = 20 # num LSTM neurons per layer
dropOut = 0.2 # dropout rate
Nlstm_layers = 1 # num layers between input & output
Nepoch = 10 # for training
batchSize = 1 # num samples used at a time to train
activation_fx = "tanh" # eg. tanh, elu, relu, selu, linear
                         # don't work well with scaling: sigmoid, exponential
lossFx = "mean_squared_error" # mae, mean_squared_error
#########################################################


def build_model(inShape, output_size, Nneurons, Nlstm_layers, activ_fx, dropOut, loss):
    model = Sequential()

    for k in range(Nlstm_layers):
        returnSeq = k!=Nlstm_layers-1 # false for last LSTM layer
        if k==0: # need to define input size in first layer only
            print("Adding LSTM Layer")
            print(returnSeq)
            model.add(LSTM(units = Nneurons, return_sequences=returnSeq,
                           activation=activ_fx, input_shape=inShape))
            model.add(Dropout(dropOut))
        else:
            print("Adding LSTM Layer")
            model.add(LSTM(units = Nneurons, return_sequences=returnSeq,
                           activation=activ_fx))
            model.add(Dropout(dropOut))

    # Adding the output layer
    model.add(Dense(units=output_size, activation="linear"))
    # Compile model
    model.compile(loss=loss, optimizer="adam")
    return model
# EOF ########################################

df = pd.read_csv("data/df_btc_feature_set.csv", index_col=0)
df = df.drop(columns= ['30 mavg','30 std','26 ema','12 ema', 'MACD', 'Signal'], axis=1)
nfeat = df.shape[1]

#print(df.isna().sum())
df.fillna(method="bfill", inplace=True)
#print(df.isna().sum())
Nfeat = df.shape[1]

# This is a little cheat, but only slightly
# We should probably scale the data each day independently
sc = MinMaxScaler(feature_range = (-1, 1))
scaled = sc.fit_transform(df)

#scaled = np.array(df)
y = scaled[1:,0] # tomorrow's price
X = scaled[:-1,:] # use today's price to predict tomorrow's

Ntest = int(np.round(len(X) * test_size))
Ntrain = len(X) - Ntest
print("Ntrain=%s , Ntest=%s" % (Ntrain,Ntest))
X_train = X[:Ntrain,:]
y_train = y[:Ntrain]
X_test  = X[Ntrain:,:]
y_test  = y[Ntrain:]
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# Create sequential data of size (Ninstances,Ndays,Nfeat)
X_train2 = np.zeros((Ntrain-Ndays,Ndays,Nfeat))
X_test2 = np.zeros((Ntest-Ndays,Ndays,Nfeat))
y_train2 = y_train[Ndays-1:-1]
y_test2 = y_test[Ndays-1:-1]
for k in range(Ndays, Ntrain):
    X_train2[k-Ndays,:,:] = X_train[k-Ndays:k,:].reshape((1,Ndays,Nfeat))
    #for colNdx in range(X_train.shape[1]):
    #    X_train2[k-Ndays,:,colNdx] = X_train[k-Ndays:k,colNdx] / X_train[k-Ndays,colNdx] - 1
for k in range(Ndays, Ntest):
    X_test2[k-Ndays,:,:] = X_test[k-Ndays:k,:].reshape((1,Ndays,Nfeat))
    #for colNdx in range(X_test.shape[1]):
    #    X_test2[k-Ndays,:,colNdx] = X_test[k-Ndays:k,colNdx] / X_test[k-Ndays,colNdx] - 1

print(X_train2.shape, y_train2.shape)
print(X_test2.shape, y_test2.shape)

# CV
# kf = KFold(n_splits=5, random_state=None, shuffle=False)

inShape = X_train2.shape[1:]
regressor = build_model(inShape, pred_size, Nneurons, Nlstm_layers, activation_fx, 
                dropOut, loss=lossFx)

# # train LSTM on folds of data
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]


# Fitting the RNN to the Training set
regressor.fit(X_train2, y_train2, epochs=Nepoch, batch_size=batchSize)

y_pred = regressor.predict(X_test2)

# save model
regressor.save("./model/lstm.h5")

# plot predicted prices
plt.plot(y_test2, '.-')
plt.plot(y_pred, '.-')
plt.grid()

"""
plt.plot(y_test2,label='orig')
plt.plot(y_elu,label='elu')
plt.plot(y_relu,label='relu')
plt.plot(y_selu,label='selu')
plt.plot(y_tanh,label='tanh')
plt.plot(y_linear,label='linear')
plt.legend(); plt.grid()
plt.title("Ndays=10, Nneurons=256, dropout=0.2, Nhidden=2, Nepoch=5, batchSize=4")
"""

# TODO load dataset from scratch, extract features, feed into LSTM using CV