#importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#feature scaling
# we use normailzation technique to featue scaling eventhoug we have standarization
# normalization means =  x- min(x)(minimum of total observation)
#						-----------------------------------------
#						 			max(x) - min(x)

from sklearn.preprocessing import MinMaxScaler
"""
	argument list of minmaxscaler which a function to find normal of observation
	feature_range =(0,1) is this because normal of observation will be between 1 and 0




"""
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#creating a datastruture with 60 steps and 1 output
""" 
   what this means is that at each time the rnn is going to look at 60 stock prices
   before time that is stock price is between 60 days before time t and time t
   and based on the trends it is capturing during this 60 previous times 
   it will predict next output

   60 time steps are the past information this rnn is going to learn and understand some correlation 
   or trends based on its understanding it is going to predict what is the next output

   which is stock price at time t+1

   number 60 days is obtained by experimenting 
   if 1 is used then it will cause overfitting nothing to learn
   similary 20, 30, are like that not enough to learn

   there is 20 financial days in one month then so 60 steps means 60 
   financial days that is 3 months

   """

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Part 2 - Building the RNN

#importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#Initializing the RNN
""" regression is predicting contious value
    classification is predicting a category or class
    """
regressor = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
#lstm has 3 parameters
#1. no of unit
#2 return sequence it is because we will be having several stack of lstm layer so we want to pass one layer  to other.
#3 input shape.
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

#2nd substep of datapreprocessing sublayer ie dropout regularization
#drop 20% in the layer
regressor.add(Dropout(0.2))
#adding second LSTM layer and some dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#adding third LSTM layer and some dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#adding fourth LSTM layer and some dropout regularization
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#adding the outputlayer
regressor.add(Dense(units = 1))

#compile RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#training rnn or fitting
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#part 3 -making prediction and visulaizing the result

#getting real stock price of google in 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#Getting the predicted stock price 2017
# for more details please see the notebook named scm and turn to page 12
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()