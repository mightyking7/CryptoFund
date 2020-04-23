import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/df_btc_feature_set.csv", index_col=0)
#print(df.isna().sum())
df.fillna(method="bfill", inplace=True)
#print(df.isna().sum())

# This is a little cheat, but only slightly
# We should probably scale the data each day independently
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
scaled = sc.fit_transform(df)
y = scaled[1:,0] # tomorrow's price
X = scaled[:-1,:] # use today's price to predict tomorrow's

test_size = 0.33
Ntest = int(np.round(len(X) * test_size))
Ntrain = len(X) - Ntest
print("Ntrain=%s , Ntest=%s" % (Ntrain,Ntest))
X_train = X[:Ntrain,:]
y_train = y[:Ntrain]
X_test  = X[Ntrain:,:]
y_test  = y[Ntrain:]
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# Create sequential data of size (Ninstances,Ndays,Nfeat)
Ndays = 30
X_train2 = np.zeros((Ntrain-Ndays,Ndays,21))
y_train2 = y_train[Ndays:]
for k in range(Ndays, Ntrain):
    X_train2[k-Ndays,:,:] = X_train[k-Ndays:k,:].reshape((1,Ndays,21))

X_test2 = np.zeros((Ntest-Ndays,Ndays,21))
y_test2 = y_test[Ndays:]
for k in range(Ndays, Ntest):
    X_test2[k-Ndays,:,:] = X_test[k-Ndays:k,:].reshape((1,Ndays,21))

print(X_train2.shape, y_train2.shape)
print(X_test2.shape, y_test2.shape)

# Building the RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

Nneurons = 64 # num LSTM neurons
dropOut = 0.25

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = Nneurons, return_sequences = True, input_shape = X_train2.shape[1:]))
regressor.add(Dropout(dropOut))

# Adding a second LSTM layer and some Dropout regularization
regressor.add(LSTM(units = Nneurons, return_sequences = True))
regressor.add(Dropout(dropOut))

# Adding a third LSTM layer and some Dropout regularization
regressor.add(LSTM(units = Nneurons, return_sequences = True))
regressor.add(Dropout(dropOut))

# Adding a fourth LSTM layer and some Dropout regularization
regressor.add(LSTM(units = Nneurons))
regressor.add(Dropout(dropOut))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
Nepoch = 100
regressor.fit(X_train2, y_train2, epochs = Nepoch, batch_size = 32)

y_pred = regressor.predict(X_test2)

plt.plot(y_test)
#plt.figure()
plt.plot(y_pred)


