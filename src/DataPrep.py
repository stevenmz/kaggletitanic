'''
Created on May 27, 2017

@author: stevenmz
'''
import pandas as pd

if __name__ == '__main__':
    path = "../train.csv"
    df = pd.read_csv(path)
    
    #Debug
    #print(df.head(10))
       
        
    #Fill missing age values
    meanAge = df.mean()["Age"]
    df["Age"] = df["Age"].fillna(meanAge)
    meanFare = df.mean()["Fare"]
    df["Fare"] = df["Fare"].fillna(meanFare)
        
    #Feature Engineering
    # i think a hasCabing is more useful than the actual cabin number
    df['HasCabin'] = df["Cabin"].notnull()
    df["Child"] = df["Age"] < 15
    
    #Get ready to code the nominal variables
    df["Embarked"] = df["Embarked"].astype('category')
    df["Sex"] = df["Sex"].astype('category')
    df["HasCabin"] = df["HasCabin"].astype('category')
    df["Child"] = df["Child"].astype('category')
    
    df["Embarked"] = df['Embarked'].cat.codes
    df['Sex'] = df['Sex'].cat.codes
    df['HasCabin'] = df['HasCabin'].cat.codes
    df['Child'] = df['Child'].cat.codes
    
    
    # remove the columns that i dont think will have predictive value
    #df = df.drop("PassengerId", axis=1)
    df = df.drop("Ticket", axis=1)
    df = df.drop("Name", axis=1)
    df = df.drop("Cabin", axis=1)
    
    #Debug
    print(df.head(10))
    
    df.to_csv("../train_prepared.csv", index=False)
    