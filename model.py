import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Control Params #######################################
#   Data Params
Ndays = 14 # number of past days info to predict tomorrow
pred_size = 1 # num days to predict
#   RNN Params
Nneurons = 64 # num LSTM neurons per layer
dropOut = 0.0 # dropout rate
Nlstm_layers = 1 # num layers between input & output
Nepoch = (20,5) # (test data , daily updates)
batchSize = 1 # num samples used at a time to train
activation_fx = "tanh" # eg. tanh, elu, relu, selu, linear
                         # don't work well with scaling: sigmoid, exponential
lossFx = "mean_squared_error" # mae, mean_squared_error
#########################################################
fpath = "testLong/ethereum.csv"

def build_model(inShape, output_size, Nneurons, Nlstm_layers, activ_fx, dropOut, loss):
    model = Sequential()

    print("Adding %s LSTM layers" % Nlstm_layers)
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
# EOF ########################################

df = pd.read_csv(fpath)
nfeat = df.shape[1]

#print(df.isna().sum())
df.fillna(method="bfill", inplace=True)
#print(df.isna().sum())
Nfeat = df.shape[1]

# This is a little cheat, but only slightly
# We should probably scale the data each day independently
sc = MinMaxScaler(feature_range = (-1, 1))
X = sc.fit_transform(df)

##################################################
# Create sequential data of size (Ninstances,Ndays,Nfeat)
N = len(X) - pred_size + 1
X_seq = np.zeros((N-Ndays, Ndays, Nfeat))
y_seq = np.zeros((N-Ndays, pred_size))
yNdx = np.arange(pred_size)
for k in range(Ndays, N):
    X_seq[k-Ndays,:,:] = X[k-Ndays:k,:].reshape((1,Ndays,Nfeat)) # includes today's price
    y_seq[k-Ndays,:] = X[yNdx+k,0] # tomorrow's price
print(X_seq.shape, y_seq.shape)
#################################################################3

#Ntest = int(np.round(len(X) * test_size))
#Ntrain = len(X) - Ntest - Ndays
#print("Ntrain=%s , Ntest=%s" % (Ntrain,Ntest))
Ntest = 25
X_train = X_seq[:-Ntest,:,:]
y_train = y_seq[:-Ntest,:]
X_test  = X_seq[-Ntest:,:,:]
y_test  = y_seq[-Ntest:,:]
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


inShape = X_train.shape[1:]
regressor = build_model(inShape, pred_size, Nneurons, Nlstm_layers, activation_fx, 
                dropOut, loss=lossFx)

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=Nepoch[0], batch_size=batchSize)
y0 = regressor.predict(X_test)

# Process test data one day at a time, update model each day
#plt.figure()
#plt.plot(y_test[:,0], '.-'); plt.grid()
error = np.zeros((X_test.shape[0],2))
y_pred = np.zeros((Ntest,pred_size))
for k in range(X_test.shape[0]):
    todays_data = X_test[k,:,:].reshape((1,-1,Nfeat))
    y_pred[k,:] = regressor.predict(todays_data)
    #plt.plot(np.arange(pred_size)+k, y_pred.T, '.-')
    
    #error[k,0] = mean_squared_error(y_test[k,:], y0[k,:])
    #error[k,1] = mean_squared_error(y_test[k,:], y_pred.T)
    regressor.fit(todays_data, y_test[k,:].reshape((1,pred_size)), epochs=Nepoch[1], batch_size=batchSize)

#y1 = regressor.predict(X_test)
#y8 = regressor.predict(X_test)
#y50 = regressor.predict(X_test)

"""
sc = MinMaxScaler(feature_range = (-1, 1))
sc.fit(np.array(df["Close"]).reshape((-1,1)))
y = sc.inverse_transform(y_pred)

yOut = pd.DataFrame(data=y.T)
yOut.to_csv("../newOutputData/"+coin+".csv", index=False)

#plt.plot(y_test[:,0]); plt.plot(y_pred)
#plt.figure()
plt.style.use('dark_background')
plt.plot(y_test[:,0],'.-',label="True")
plt.plot(y64,label="64 units, 1 layer")
plt.plot(y128,label="128 units, 3 layers")
plt.legend(); plt.grid()
plt.title("Ethereum Price Predicion with LSTM Network")
#plt.figure()
#plt.plot(y_test[:,0]); plt.plot(y50)
"""


