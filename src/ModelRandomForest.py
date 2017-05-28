'''
Created on May 27, 2017

@author: stevenmz
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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
    
    # Kaggle provided values for us to predict
    val_passengerIDs = df_val.values[:, 1]
    val_X = df_val.values[:, 2:]
    # print(train_X.shape)
    
    model = RandomForestClassifier(n_estimators=15)
    model.fit(X_train, y_train)
    print "Accuracy on 33% test set: ", model.score(X_test, y_test)
    
    print "predicting Kaggle test set"
    predictions = model.predict(val_X)
    
    dfFinal = pd.DataFrame(val_passengerIDs, columns=["PassengerId"])
    dfFinal["Survived"] = predictions
    
    print dfFinal.head(10)
    dfFinal.to_csv("../submission.csv")
    
    
    
    
    
