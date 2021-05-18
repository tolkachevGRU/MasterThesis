import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
import Classes
import Database
import numpy as np
import warnings
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,WhiteKernel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import mean_squared_error as MSE
from sklearn import metrics
from keras.initializers import Initializer
from keras.optimizers import SGD
from sklearn.cluster import KMeans
from keras.layers import Layer
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Dense, Flatten, Activation
from keras.models import Sequential
from keras.losses import binary_crossentropy
from sklearn.preprocessing import OneHotEncoder
from keras.initializers import RandomUniform, Initializer, Constant
from numpy import shape
from neupy import algorithms
from pyGRNN import GRNN
from sklearn.base import BaseEstimator, RegressorMixin
from pyGRNN import GRNN
from tensorflow.keras.models import  load_model
import time
from sklearn.metrics import accuracy_score
import re
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Input, Dropout
from keras import Sequential
from sklearn.metrics import classification_report, confusion_matrix

def NNwithJungle():
    time_start = time.time()

    print('NN with jungle')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `kills`, `deaths`, `assists`, `team kpm`, `dragons`,`elementaldrakes`, `infernals`, `mountains`, `clouds`, `oceans`, `elders`, `heralds`, `barons`, `towers`, `inhibitors`, `damagetochampions`, `dpm`,`damagetakenperminute`, `damagemitigatedperminute`, `wardsplaced`, `wpm`, `wardskilled`, `wcpm`, `controlwardsbought`, `visionscore`, `vspm`, `totalgold`, `earnedgold`, `earned gpm`, `goldspent`, `gspd`, `minionkills`, `monsterkills`, `monsterkillsownjungle`, `monsterkillsenemyjungle`, `cspm`, `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15`, `jng_damagetochampions`, `jng_dpm`, `jng_damageshare`, `jng_damagetakenperminute`, `jng_damagemitigatedperminute`, `jng_wardsplaced`, `jng_wpm`, `jng_wardskilled`, `jng_wcpm`, `jng_controlwardsbought`, `jng_visionscore`, `jng_vspm`, `jng_totalgold`, `jng_earnedgold`, `jng_earned gpm`, `jng_earnedgoldshare`, `jng_goldspent`, `jng_gspd`, `jng_total cs`, `jng_minionkills`, `jng_monsterkills`, `jng_monsterkillsownjungle`, `jng_monsterkillsenemyjungle`, `jng_cspm`, `jng_goldat10`, `jng_xpat10`, `jng_csat10`, `jng_golddiffat10`, `jng_xpdiffat10`, `jng_csdiffat10`, `jng_killsat10`, `jng_assistsat10`, `jng_deathsat10`,`jng_goldat15`, `jng_xpat15`, `jng_csat15`, `jng_golddiffat15`, `jng_xpdiffat15`, `jng_csdiffat15`, `jng_killsat15`, `jng_assistsat15`, `jng_deathsat15` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    #Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)
    y = y.ravel()

    #Scaling the data
    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    parameters = {'batch_size': [16, 32],  
              'epochs': [10, 50, 100], 'optimizer': ['adam', 'sgd', 'RMSprop']}
    def creat_model(optimizer):
        model = Sequential()
        model.add(Dense(96, input_dim=96, activation='relu'))
        model.add(Dense(74, activation='relu'))
        model.add(Dense(1, activation ='sigmoid' ))
        model.compile(optimizer = optimizer,loss ='binary_crossentropy', metrics = ['accuracy'])
        return model
    
    estimator = KerasClassifier(build_fn=creat_model)

    grid = GridSearchCV(estimator,param_grid = parameters, scoring = 'accuracy', verbose = 0)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    print(grid.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)

def NNwithoutJungle():
    time_start = time.time()

    print("NN without jungle")
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `kills`, `deaths`, `assists`, `team kpm`, `dragons`,`elementaldrakes`, `infernals`, `mountains`, `clouds`, `oceans`, `elders`, `heralds`, `barons`, `towers`, `inhibitors`, `damagetochampions`, `dpm`,`damagetakenperminute`, `damagemitigatedperminute`, `wardsplaced`, `wpm`, `wardskilled`, `wcpm`, `controlwardsbought`, `visionscore`, `vspm`, `totalgold`, `earnedgold`, `earned gpm`, `goldspent`, `gspd`, `minionkills`, `monsterkills`, `monsterkillsownjungle`, `monsterkillsenemyjungle`, `cspm`, `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    #Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)
    y = y.ravel()

    #Scaling the data
    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    parameters = {'batch_size': [16, 32],  
              'epochs': [10, 50, 100], 'optimizer': ['adam', 'sgd', 'RMSprop']}
    def creat_model(optimizer):
        model = Sequential()
        model.add(Dense(54, input_dim=54, activation='relu'))
        model.add(Dense(27, activation='relu'))
        model.add(Dense(1, activation ='sigmoid' ))
        model.compile(optimizer = optimizer,loss ='binary_crossentropy', metrics = ['accuracy'])
        return model
    
    estimator = KerasClassifier(build_fn=creat_model)

    grid = GridSearchCV(estimator,param_grid = parameters, scoring = 'accuracy', verbose = 0)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    print(grid.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)

def NN10jungle():
    time_start = time.time()
    print('NN with jungle 10')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `wpm`, `wcpm`, `vspm`, `jng_goldat10`, `jng_xpat10`, `jng_csat10`, `jng_golddiffat10`, `jng_xpdiffat10`, `jng_csdiffat10`, `jng_killsat10`, `jng_assistsat10`, `jng_deathsat10`, `jng_wpm`, `jng_wcpm`, `jng_vspm` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    #Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)
    y = y.ravel()

    #Scaling the data
    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    parameters = {'batch_size': [32],  
              'epochs': [100], 'optimizer': ['sgd']}
    def creat_model(optimizer):
        model = Sequential()
        model.add(Dense(24, input_dim=24, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(1, activation ='sigmoid' ))
        model.compile(optimizer = optimizer,loss ='binary_crossentropy', metrics = ['accuracy'])
        return model
    
    estimator = KerasClassifier(build_fn=creat_model)

    grid = GridSearchCV(estimator,param_grid = parameters, scoring = 'accuracy', verbose = 0)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    print(grid.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)

def NN10():
    time_start = time.time()
    print("NN 10")
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `wpm`, `wcpm`, `vspm` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    #Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)
    y = y.ravel()

    #Scaling the data
    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    parameters = {'batch_size': [32],  
              'epochs': [100], 'optimizer': ['sgd']}
    def creat_model(optimizer):
        model = Sequential()
        model.add(Dense(12, input_dim=12, activation='relu'))
        model.add(Dense(6, activation='relu'))
        model.add(Dense(1, activation ='sigmoid' ))
        model.compile(optimizer = optimizer,loss ='binary_crossentropy', metrics = ['accuracy'])
        return model
    
    estimator = KerasClassifier(build_fn=creat_model)

    grid = GridSearchCV(estimator,param_grid = parameters, scoring = 'accuracy', verbose = 0)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    print(grid.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)

def NN15jungle():
    

    print('NN with jungle 15')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `wpm`, `wcpm`, `vspm`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15`, `jng_goldat15`, `jng_xpat15`, `jng_csat15`, `jng_golddiffat15`, `jng_xpdiffat15`, `jng_csdiffat15`, `jng_killsat15`, `jng_assistsat15`, `jng_deathsat15`, `jng_wpm`, `jng_wcpm`, `jng_vspm` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    #Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)
    y = y.ravel()

    #Scaling the data
    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    parameters = {'batch_size': [32],  
              'epochs': [100], 'optimizer': ['sgd']}
    def creat_model(optimizer):
        model = Sequential()
        model.add(Dense(24, input_dim=24, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(1, activation ='sigmoid' ))
        model.compile(optimizer = optimizer,loss ='binary_crossentropy', metrics = ['accuracy'])
        return model
    
    estimator = KerasClassifier(build_fn=creat_model)

    grid = GridSearchCV(estimator,param_grid = parameters, scoring = 'accuracy', verbose = 0)
    grid.fit(X_train, y_train)
    time_start = time.time()
    y_pred = grid.predict(X_test)
    end = time.time()
    print(end-time_start)
    print(grid.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    

def NN15():
    time_start = time.time()

    print('NN 15')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `wpm`, `wcpm`, `vspm`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    #Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)
    y = y.ravel()

    #Scaling the data
    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    parameters = {'batch_size': [32],  
              'epochs': [100], 'optimizer': ['sgd']}
    def creat_model(optimizer):
        model = Sequential()
        model.add(Dense(12, input_dim=12, activation='relu'))
        model.add(Dense(6, activation='relu'))
        model.add(Dense(1, activation ='sigmoid' ))
        model.compile(optimizer = optimizer,loss ='binary_crossentropy', metrics = ['accuracy'])
        return model
    
    estimator = KerasClassifier(build_fn=creat_model)

    grid = GridSearchCV(estimator,param_grid = parameters, scoring = 'accuracy', verbose = 0)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    print(grid.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)