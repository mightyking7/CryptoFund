import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Control Params #######################################
#   Data Params
test_size = 0.33 # contiguous segments for train & test
Ndays = 8 # number of past days info to predict tomorrow
#   RNN Params
Nneurons = 256 # num LSTM neurons per layer
dropOut = 0.2 # dropout rate
Nhidden_layers = 2 # num layers between input & output
Nepoch = 100 # for training
batchSize = 8 # num samples used at a time to train
activation_fx = "relu" # eg. tanh, relu, sigmoid
#########################################################

df = pd.read_csv("data/df_btc_feature_set.csv", index_col=0)
df = df.drop(columns= ['30 mavg','30 std','26 ema','12 ema', 'MACD', 'Signal'], axis=1)
nfeat = df.shape[1]

#print(df.isna().sum())
df.fillna(method="bfill", inplace=True)
#print(df.isna().sum())
Nfeat = df.shape[1]

# This is a little cheat, but only slightly
# We should probably scale the data each day independently
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
Ndays = 30
X_train2 = np.zeros((Ntrain-Ndays,Ndays,nfeat))
y_train2 = y_train[Ndays:]
for k in range(Ndays, Ntrain):
    X_train2[k-Ndays,:,:] = X_train[k-Ndays:k,:].reshape((1,Ndays,nfeat))

X_test2 = np.zeros((Ntest-Ndays,Ndays,nfeat))
y_test2 = y_test[Ndays:]
for k in range(Ndays, Ntest):
    X_test2[k-Ndays,:,:] = X_test[k-Ndays:k,:].reshape((1,Ndays,nfeat))

X_train2 = np.zeros((Ntrain-Ndays,Ndays,Nfeat))
X_test2 = np.zeros((Ntest-Ndays,Ndays,Nfeat))
y_train2 = y_train[Ndays-1:-1]
y_test2 = y_test[Ndays-1:-1]
for k in range(Ndays, Ntrain):
    X_train2[k-Ndays,:,:] = X_train[k-Ndays:k,:].reshape((1,Ndays,Nfeat))
for k in range(Ndays, Ntest):
    X_test2[k-Ndays,:,:] = X_test[k-Ndays:k,:].reshape((1,Ndays,Nfeat))


print(X_train2.shape, y_train2.shape)
print(X_test2.shape, y_test2.shape)

# Building the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
#regressor.add(LSTM(units = Nneurons, return_sequences = True, input_shape = X_train2.shape[1:]))
regressor.add(LSTM(units = Nneurons, return_sequences=True, activation=activation_fx,
                   input_shape = X_train2.shape[1:]))
regressor.add(Dropout(dropOut))

for k in range(Nhidden_layers-1):
    regressor.add(LSTM(units = Nneurons, return_sequences=True, activation=activation_fx))
    regressor.add(Dropout(dropOut))

# Add last LSTM layer and some Dropout regularization
regressor.add(LSTM(units = Nneurons, activation=activation_fx))
regressor.add(Dropout(dropOut))

# Adding the output layer
regressor.add(Dense(units = 1)) # single unit to output price

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train2, y_train2, epochs=Nepoch, batch_size=batchSize)

# saving the model in tensorflow format
#regressor.save('../MyModel_tf',save_format='tf')
# loading the saved model
#loaded_model = keras.models.load_model('../MyModel_tf')
# retraining the model
#loaded_model.fit(x_train, y_train, epochs = 10, validation_data = (x_test,y_test),verbose=1)

y_pred = regressor.predict(X_test2)

plt.plot(y_test2)
#plt.figure()
plt.plot(y_pred)


