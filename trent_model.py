import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Crypto_Fund/df_btc_feature_set.csv")
#print(df.isna().sum())
df.fillna(method="bfill", inplace=True)
#print(df.isna().sum())

X = np.array(df.iloc[:,1:])
y = np.array(df['Close']).ravel()
#tmp = np.array(df['Close']).ravel()
#y = np.diff(tmp) / tmp[:-1] # percent change

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=21)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
sc.fit(X_train)
X_train_scaled = sc.transform(X_train)
X_test_scaled = sc.transform(X_test)

print(X_train_scaled.shape)
#print(np.mean(X_train_scaled,axis=0)

# Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
Nepoch = 30
regressor.fit(X_train.reshape((523,21,1)), y_train, epochs = Nepoch, batch_size = 32)

y_pred = regressor.predict(X_test.reshape((-1,21,1)))

plt.plot(y_test)
plt.figure()
plt.plot(y_pred)


