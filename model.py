import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Control Params #######################################
#   Data Params
Ndays = 21 # number of past days info to predict tomorrow
pred_size = 1 # num days to predict
#   RNN Params
Nneurons = 64 # num LSTM neurons per layer
dropOut = 0.2 # dropout rate
Nlstm_layers = 2 # num layers between input & output
Nepoch = (30,5) # (test data , daily updates)
batchSize = 1 # num samples used at a time to train
activation_fx = "tanh" # eg. tanh, elu, relu, selu, linear
                         # don't work well with scaling: sigmoid, exponential
lossFx = "mean_squared_error" # mae, mean_squared_error
#########################################################
fpath = "input_12mo/bitcoin.csv"
tomorrow = 183

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
Nfeat = df.shape[1]

# Sequence data
# Scale data for network efficiency
sc = MinMaxScaler(feature_range = (-1, 1))
X = sc.fit_transform(df.iloc[:tomorrow,:])
# Create sequential data of size (Ninstances,Ndays,Nfeat)
N = tomorrow - pred_size + 1
X_seq = np.zeros((N-Ndays, Ndays, Nfeat))
y_seq = np.zeros((N-Ndays, pred_size))
yNdx = np.arange(pred_size)
for k in range(Ndays, N):
    X_seq[k-Ndays,:,:] = X[k-Ndays:k,:].reshape((1,Ndays,Nfeat)) # includes today's price
    y_seq[k-Ndays,:] = X[yNdx+k,0] # tomorrow's price
X_train = X_seq
y_train = y_seq
# TEST ################################################
NN = len(df) - k
X_teq = np.zeros((NN-1, Ndays, Nfeat))
y_teq = np.zeros((NN-1, pred_size))
sc_inv = np.zeros((NN-1, 2))
n = 0
while (tomorrow<len(df)):
    Xt = sc.fit_transform(df.iloc[:tomorrow,:]) # doesn't scale into the future
    X_teq[n,:,:] = Xt[-Ndays:,:].reshape((1,Ndays,Nfeat)) # includes today's price
    y_teq[n,:] = df["Close"][tomorrow] # tomorrow's unscaled price
    sc_inv[n,:] = [sc.data_range_[0] , sc.data_min_[0]]
    tomorrow += 1
    n += 1
X_test = X_teq
y_test = y_teq
sc_inv = sc_inv
#################################################################3

inShape = X_train.shape[1:]
regressor = build_model(inShape, pred_size, Nneurons, Nlstm_layers, activation_fx, 
                dropOut, loss=lossFx)

# Fitting the RNN to the Training set
train_hist = regressor.fit(X_train, y_train, epochs=Nepoch[0], batch_size=batchSize)
y0 = regressor.predict(X_test)

# Plot training past
#plt.plot(y_train,'.-')

Ntest = len(y_test)
y_pred = np.zeros((Ntest, pred_size))
sc = MinMaxScaler(feature_range = (-1, 1))
for day_num in range(Ntest):
    if day_num>0:
        # Use yesterday's data to train model for use today
        y_test_sc = 2*(y_test[day_num-1]-sc_inv[day_num,1])/sc_inv[day_num,0] - 1
        regressor.fit(todays_data, y_test_sc.reshape((1,pred_size)),
                          epochs=Nepoch[1], batch_size=batchSize)
    
    # just before midnight
    todays_data = X_test[day_num,:,:].reshape((1,-1,Nfeat))
    
    #plt.plot(len(y_train)+day_num-1 , todays_data[0,-1,0],'b*')
    #plt.xlim((len(y_train)-5 , len(y_train)+1))
    
    y_pred_sc = regressor.predict(todays_data)
    y_pred[day_num,:] = (y_pred_sc+1)*sc_inv[day_num,0]/2 + sc_inv[day_num,1]
    
    #plt.plot(len(y_train)+day_num , y_pred_sc, 'r*')    


plt.plot(y_test,'.-'); plt.plot(y_pred,'.-'); plt.grid()
a = (np.diff(y_test,axis=0)>0) & (np.diff(y_pred,axis=0)>0)
print(a.sum()/len(a))

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


