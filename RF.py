from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
import Database
import numpy as np
from numpy import arange
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import ElasticNetCV
import time
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt


def RFjungle():
    time_start = time.time()

    print('RF jungle all')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `kills`, `deaths`, `assists`, `team kpm`, `dragons`,`elementaldrakes`, `infernals`, `mountains`, `clouds`, `oceans`, `elders`, `heralds`, `barons`, `towers`, `inhibitors`, `damagetochampions`, `dpm`,`damagetakenperminute`, `damagemitigatedperminute`, `wardsplaced`, `wpm`, `wardskilled`, `wcpm`, `controlwardsbought`, `visionscore`, `vspm`, `totalgold`, `earnedgold`, `earned gpm`, `goldspent`, `gspd`, `minionkills`, `monsterkills`, `monsterkillsownjungle`, `monsterkillsenemyjungle`, `cspm`, `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15`, `jng_damagetochampions`, `jng_dpm`, `jng_damageshare`, `jng_damagetakenperminute`, `jng_damagemitigatedperminute`, `jng_wardsplaced`, `jng_wpm`, `jng_wardskilled`, `jng_wcpm`, `jng_controlwardsbought`, `jng_visionscore`, `jng_vspm`, `jng_totalgold`, `jng_earnedgold`, `jng_earned gpm`, `jng_earnedgoldshare`, `jng_goldspent`, `jng_gspd`, `jng_total cs`, `jng_minionkills`, `jng_monsterkills`, `jng_monsterkillsownjungle`, `jng_monsterkillsenemyjungle`, `jng_cspm`, `jng_goldat10`, `jng_xpat10`, `jng_csat10`, `jng_golddiffat10`, `jng_xpdiffat10`, `jng_csdiffat10`, `jng_killsat10`, `jng_assistsat10`, `jng_deathsat10`,`jng_goldat15`, `jng_xpat15`, `jng_csat15`, `jng_golddiffat15`, `jng_xpdiffat15`, `jng_csdiffat15`, `jng_killsat15`, `jng_assistsat15`, `jng_deathsat15` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = { 
    'n_estimators': [10,20,30,40,50,60,70,80,90],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,9,10,11,12],
    'criterion' :['gini', 'entropy']}
    rfc=RandomForestClassifier(random_state=42)

    model = GridSearchCV(estimator=rfc, param_grid=param_grid)
    # Fit on training data
    model.fit(X_train, y_train)
    print(model.best_params_)
    model.best_estimator_.coef_
    importance = model.coef_
    for i,v in enumerate(importance):
	    print('Feature: %0d, Score: %.5f' % (i,v))
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)

def RFwithoutjungle():
    time_start = time.time()

    print('RF jungle all')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `kills`, `deaths`, `assists`, `team kpm`, `dragons`,`elementaldrakes`, `infernals`, `mountains`, `clouds`, `oceans`, `elders`, `heralds`, `barons`, `towers`, `inhibitors`, `damagetochampions`, `dpm`,`damagetakenperminute`, `damagemitigatedperminute`, `wardsplaced`, `wpm`, `wardskilled`, `wcpm`, `controlwardsbought`, `visionscore`, `vspm`, `totalgold`, `earnedgold`, `earned gpm`, `goldspent`, `gspd`, `minionkills`, `monsterkills`, `monsterkillsownjungle`, `monsterkillsenemyjungle`, `cspm`, `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = { 
    'n_estimators': [10,20,30,40,50,60,70,80,90],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,9,10,11,12],
    'criterion' :['gini', 'entropy']}
    rfc=RandomForestClassifier(random_state=42)
    

    model = GridSearchCV(estimator=rfc, param_grid=param_grid)
    # Fit on training data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(model.best_params_)
    end = time.time()
    print(end-time_start)

def RF10jungle():
    time_start = time.time()

    print('RF 10 jungle')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `wpm`, `wcpm`, `vspm`, `jng_goldat10`, `jng_xpat10`, `jng_csat10`, `jng_golddiffat10`, `jng_xpdiffat10`, `jng_csdiffat10`, `jng_killsat10`, `jng_assistsat10`, `jng_deathsat10`, `jng_wpm`, `jng_wcpm`, `jng_vspm` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = { 
    'n_estimators': [3,5,7,9,11,13,15,18],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,9,10,11,12],
    'criterion' :['gini', 'entropy']}
    rfc=RandomForestClassifier(random_state=42)

    model = GridSearchCV(estimator=rfc, param_grid=param_grid)
    # Fit on training data
    model.fit(X_train, y_train)

    df = pd.DataFrame({})


    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)

def RF10():
    time_start = time.time()

    print('RF 10')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `wpm`, `wcpm`, `vspm` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = { 
    'n_estimators': [3,5,7,9,11,13,15,18],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,9,10,11,12],
    'criterion' :['gini', 'entropy']}
    rfc=RandomForestClassifier(random_state=42)

    model = GridSearchCV(estimator=rfc, param_grid=param_grid)
    # Fit on training data
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(model.best_params_)
    end = time.time()
    print(end-time_start)

def RF15jungle():
    time_start = time.time()

    print('RF 15 jungle')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `wpm`, `wcpm`, `vspm`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15`, `jng_goldat15`, `jng_xpat15`, `jng_csat15`, `jng_golddiffat15`, `jng_xpdiffat15`, `jng_csdiffat15`, `jng_killsat15`, `jng_assistsat15`, `jng_deathsat15`, `jng_wpm`, `jng_wcpm`, `jng_vspm` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    headersx = [ 'wpm', 'wcpm', 'vspm', 'goldat15', 'xpat15', 'csat15', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15', 'jng_goldat15', 'jng_xpat15', 'jng_csat15', 'jng_golddiffat15', 'jng_xpdiffat15', 'jng_csdiffat15', 'jng_killsat15', 'jng_assistsat15', 'jng_deathsat15', 'jng_wpm', 'jng_wcpm', 'jng_vspm']
    print(headersx)
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    param_grid = { 
    'n_estimators': [3,5,7,9,11,13,15,18],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,9,10,11,12],
    'criterion' :['gini', 'entropy']}
    rfc=RandomForestClassifier(random_state=42)

    model = GridSearchCV(estimator=rfc, param_grid=param_grid)
    # Fit on training data
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    print(importances) 

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(model.best_params_)
    end = time.time()
    print(end-time_start)

def RF15():
    time_start = time.time()

    print('RF 15')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `wpm`, `wcpm`, `vspm`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = { 
    'n_estimators': [3,5,7,9,11,13,15,18],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,9,10,11,12],
    'criterion' :['gini', 'entropy']}
    rfc=RandomForestClassifier(random_state=42)

    model = GridSearchCV(estimator=rfc, param_grid=param_grid)
    # Fit on training data
    model.fit(X_train, y_train)
    #importances = model.feature_importances_
    #print(importances) 

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(model.best_params_)
    end = time.time()
    print(end-time_start)


