import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from func_def import load_coins, build_model, get_params

# Control Params #######################################
# RNN Params that vary across models
Nmodels = 10
Nneurons =(64,256) # num LSTM neurons per layer
dropOut = (0,0.3) # dropout rate
Nlstm_layers = (2,8) # num layers between input & output
# RNN Params persistent across models
Nepoch = (100,5) # (test data , daily updates)
activation_fx = "tanh" # eg. tanh, elu, relu, selu, linear
                         # don't work well with scaling: sigmoid, exponential
lossFx = "mean_squared_error" # mae, mean_squared_error
# Data Params
#test_size = 0.33 # contiguous segments for train & test
Ndays = (7,21) # number of past days info to predict tomorrow
pred_size = (1,1) # num days to predict
#######################################################
coin_names = ["bitcoin","dash","ethereum","litecoin","monero","ripple"]
#coin_names = ["lit","mon","rip"]
#coin_names = ["bitcoin"]
tomorrow = 183
batchSize = Ndays[0] # num samples used at a time to train

# Get model parameters
params = get_params(Nmodels, Nneurons, dropOut, Nlstm_layers, Ndays, pred_size)

# Load Data
data = load_coins(Ndays[1], pred_size[1], coin_names, tomorrow) # data for each coin
Nfeat = data[coin_names[0]]["X_train"].shape[2]

# Build band of models for each coin
bank = {}
for coin in coin_names:
    bank[coin] = []
    for k in range(Nmodels):
        n_days = int(params["Ndays"][k])
        data_shape = (n_days,Nfeat)
        bank[coin].append(build_model(data_shape, int(params["pred_size"][k]), int(params["Nneurons"][k]), 
                      int(params["Nlstm_layers"][k]), activation_fx, params["dropOut"][k], lossFx))

# Train the bank on training data
for coin in coin_names:
    for k in range(Nmodels):
        print("Training model %s of %s" % (k+1,Nmodels))
        n_days = int(params["Ndays"][k])
        n_pred = int(params["pred_size"][k])
        X_train = data[coin]["X_train"][:,-n_days:,:]
        y_train = data[coin]["y_train"][:,:n_pred]
        bank[coin][k].fit(X_train, y_train, epochs=Nepoch[0], batch_size=batchSize)


# Process test data one day at a time, update models each day
Ntest = len(data[coin_names[0]]["y_test"])
Ncoins = len(coin_names)
NmaxPred = int(np.max(params["pred_size"]))
y_pred = np.zeros((Ntest, NmaxPred, Nmodels, Ncoins))
sc = MinMaxScaler(feature_range = (-1, 1))
coin_num = 0
for coin in coin_names:
    #plt.figure()
    X_test = data[coin]["X_test"][:,-n_days:,:]
    y_test = data[coin]["y_test"][:,:n_pred]
    sc_inv = data[coin]["sc_inv"]
    for model_num in range(Nmodels):
        n_days = int(params["Ndays"][model_num])
        n_pred = int(params["pred_size"][model_num])
        #if (model_num==0): plt.plot(y_test[:,0], '.-'); plt.grid()
        for day_num in range(Ntest):
            if day_num>0:
                # Use yesterday's data to train model for use today
                y_test_sc = 2*(y_test[day_num-1]-sc_inv[day_num,1])/sc_inv[day_num,0] - 1
                regressor.fit(todays_data, y_test_sc.reshape((1,pred_size)),
                          epochs=Nepoch[1], batch_size=1)
    
            # just before midnight
            todays_data = X_test[day_num,:,:].reshape((1,-1,Nfeat))
            y_pred_sc = bank[coin][model_num].predict(todays_data)
            y_pred[day_num,:n_pred,model_num,coin_num] = (y_pred_sc+1)*sc_inv[day_num,0]/2 + sc_inv[day_num,1]
            
    coin_num += 1


yNdx = np.arange(pred_size[1])
plt.plot(y_test[:,0],'.-'); plt.grid()
#mu = np.zeros((Ntest,4)); sigma = mu
for k in range(0,Ntest):
    #plt.plot(k+yNdx , y_pred[k,:,0,0],'.-')
    mu = np.mean(y_pred[k,:,:,0],axis=1)
    sigma = np.std(y_pred[k,:,:,0],axis=1)
    plt.errorbar(k+yNdx , mu , sigma)


# Save output
coin_num = 0
for coin in coin_names:
    yOut = pd.DataFrame(data=y_pred[:,0,:,coin_num].T)
    yOut.to_csv("output_12mo/"+coin+".csv", index=False)
    
    coin_num += 1
