import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score

def preprocess(data):
    data = data.drop(['model', 'transmission','fuelType'],axis=1)
    return data

def split(data):
    X = data.drop(['price'],axis=1)
    y = data['price']
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=25)
    return X_train, X_test, y_train, y_test

def lr(X,y):
    regressor = LinearRegression()
    regressor.fit(X, y)
    return regressor

