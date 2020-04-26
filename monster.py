import numpy as np
import matplotlib.pyplot as plt
from func_def import load_coins, build_model, get_params

# Control Params #######################################
# RNN Params that vary across models
Nmodels = 10
Nneurons =(20,128) # num LSTM neurons per layer
dropOut = (0,0.5) # dropout rate
Nlstm_layers = (1,4) # num layers between input & output
# RNN Params persistent across models
Nepoch = (10,5) # (test data , daily updates)
batchSize = 1 # num samples used at a time to train
activation_fx = "tanh" # eg. tanh, elu, relu, selu, linear
                         # don't work well with scaling: sigmoid, exponential
lossFx = "mean_squared_error" # mae, mean_squared_error
coin_names = ("bit","dash","eth","lit","mon","rip")
# Data Params
#test_size = 0.33 # contiguous segments for train & test
Ndays = (7,30) # number of past days info to predict tomorrow
pred_size = (1,1) # num days to predict
#######################################################
#coin_names = ("bit","dash","eth","lit","mon","rip")
coin_names = ["bit"]
today = 183

# Get model parameters
params = get_params(Nmodels, Nneurons, dropOut, Nlstm_layers, Ndays, pred_size)

# Load Data
#data = load_coins(params, coin_names, today) # list of data, one for each model
data = load_coins(Ndays[1], pred_size[1], coin_names, today) # data for each coin
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
coin_num = 0
for coin in coin_names:
    #plt.figure()
    for model_num in range(Nmodels):
        n_days = int(params["Ndays"][model_num])
        n_pred = int(params["pred_size"][model_num])
        X_test = data[coin]["X_test"][:,-n_days:,:]
        y_test = data[coin]["y_test"][:,:n_pred]
        #if (model_num==0): plt.plot(y_test[:,0], '.-'); plt.grid()
        for day_num in range(Ntest):
            todays_data = X_test[day_num,:,:].reshape((1,-1,Nfeat))
            y_pred[day_num,:n_pred,model_num,coin_num] = bank[coin][model_num].predict(todays_data)
            #plt.plot(np.arange(Npred)+day_num, y_pred.T, '.-')
            
            # Use today's data to train model for use tomorrow
            bank[coin][model_num].fit(todays_data, y_test[day_num,:].reshape((1,n_pred)),
                              epochs=Nepoch[1], batch_size=batchSize)
            
    coin_num +=1
        
    
plt.plot(y_test,'.-'); plt.plot(y_pred[:,0,:,0]); plt.grid()
