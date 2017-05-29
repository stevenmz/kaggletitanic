'''
Created on May 27, 2017

@author: stevenmz
'''

from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd

import numpy
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing import sequence
from keras.layers.core import Dropout
from __builtin__ import int

if __name__ == '__main__':
    trainpath = "../train_prepared.csv"
    testpath = "../test_prepared.csv"
    
    df_train = pd.read_csv(trainpath)
    df_val = pd.read_csv(testpath)
    
    Y = df_train.values[:, 1]
    X = df_train.values[:, 2:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=11610)
    print "Train shape: ", X_train.shape
    print "Test shape: ", X_test.shape
    
    print "Total: training instances -> ", len(X_train), ", test instances ->", len(X_test)
    
    # truncate and pad input sequences
    num_samples = X_train[0].shape[0]
    print "Padding and truncating to {} samples".format(num_samples)
    X_train = sequence.pad_sequences(X_train, maxlen=num_samples, dtype="int32")
    X_test = sequence.pad_sequences(X_test, maxlen=num_samples, dtype="int32")
        
    # Number of training instances      
    nb_samples_train = X_train.shape[0]   
    nb_samples_test = X_test.shape[0]
    
    timesteps = 1
    
    X_train = numpy.reshape(X_train, (nb_samples_train, timesteps, num_samples))
    X_test = numpy.reshape(X_test, (nb_samples_test, timesteps, num_samples))
    
    # create the model of input layer->lstm->lstm->output layer
    recurrent_node_count = 200
    number_of_epochs = 200
    verbose_level = 2
    model = Sequential()
    model.add(LSTM(recurrent_node_count, input_shape=X_train.shape[1:]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_split=0.25, nb_epoch=number_of_epochs, batch_size=num_samples, verbose=verbose_level, shuffle=False)
        
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # Kaggle provided values for us to predict
    val_passengerIDs = df_val.values[:, 0]
    val_X = df_val.values[:, 1:]
    
    num_samples_val = val_X[0].shape[0] 
    nb_samples_val = val_X.shape[0]
    val_X = numpy.reshape(val_X, (nb_samples_val, timesteps, num_samples_val))
    predictions = model.predict_classes(val_X, batch_size=32, verbose=1)
    
    # print(train_X.shape)
    
    dfFinal = pd.DataFrame(val_passengerIDs, columns=["PassengerId"])
    dfFinal["PassengerId"] = dfFinal["PassengerId"].astype(int)
    dfFinal["Survived"] = predictions
    
    print dfFinal.head(10)
    dfFinal.to_csv("../submissionLSTM.csv", index=False)
    
    
    
    
    
