import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Control Params #######################################
test_size = 0.33 # contiguous segments for train & test
Nneurons = 256 # num LSTM neurons per layer
dropOut = 0.2 # dropout rate
Nhidden_nodes = 2 # num layers between input & output
Nepoch = 20 # for training
batchSize = 32 # num samples used at a time
Ndays = 10 # number of past days info to predict tomorrow
#########################################################

df = pd.read_csv("data/df_btc_feature_set.csv", index_col=0)
df = df.drop(columns= ['30 mavg','30 std','26 ema','12 ema', 'MACD', 'Signal'], axis=1)
#print(df.isna().sum())
df.fillna(method="bfill", inplace=True)
#print(df.isna().sum())
Nfeat = df.shape[1]

# This is a little cheat, but only slightly
# We should probably scale the data each day independently
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
scaled = sc.fit_transform(df)
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
y_train2 = y_train[Ndays:]
for k in range(Ndays, Ntrain):
    X_train2[k-Ndays,:,:] = X_train[k-Ndays:k,:].reshape((1,Ndays,Nfeat))

X_test2 = np.zeros((Ntest-Ndays,Ndays,Nfeat))
y_test2 = y_test[Ndays:]
for k in range(Ndays, Ntest):
    X_test2[k-Ndays,:,:] = X_test[k-Ndays:k,:].reshape((1,Ndays,Nfeat))

print(X_train2.shape, y_train2.shape)
print(X_test2.shape, y_test2.shape)

# Building the RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = Nneurons, return_sequences = True, input_shape = X_train2.shape[1:]))
regressor.add(Dropout(dropOut))

for k in range(Nhidden_nodes-1):
    regressor.add(LSTM(units = Nneurons, return_sequences = True))
    regressor.add(Dropout(dropOut))

# Add last LSTM layer and some Dropout regularization
regressor.add(LSTM(units = Nneurons))
regressor.add(Dropout(dropOut))

# Adding the output layer
regressor.add(Dense(units = 1)) # single unit to output price

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train2, y_train2, epochs = Nepoch, batch_size = batchSize)

y_pred = regressor.predict(X_test2)

plt.plot(y_test2)
#plt.figure()
plt.plot(y_pred)


