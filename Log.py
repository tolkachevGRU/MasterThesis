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

def LogWithJungle():
    print('Log predict score with jungle data')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `kills`, `deaths`, `assists`, `team kpm`, `dragons`,`elementaldrakes`, `infernals`, `mountains`, `clouds`, `oceans`, `elders`, `heralds`, `barons`, `towers`, `inhibitors`, `damagetochampions`, `dpm`,`damagetakenperminute`, `damagemitigatedperminute`, `wardsplaced`, `wpm`, `wardskilled`, `wcpm`, `controlwardsbought`, `visionscore`, `vspm`, `totalgold`, `earnedgold`, `earned gpm`, `goldspent`, `gspd`, `minionkills`, `monsterkills`, `monsterkillsownjungle`, `monsterkillsenemyjungle`, `cspm`, `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15`, `jng_damagetochampions`, `jng_dpm`, `jng_damageshare`, `jng_damagetakenperminute`, `jng_damagemitigatedperminute`, `jng_wardsplaced`, `jng_wpm`, `jng_wardskilled`, `jng_wcpm`, `jng_controlwardsbought`, `jng_visionscore`, `jng_vspm`, `jng_totalgold`, `jng_earnedgold`, `jng_earned gpm`, `jng_earnedgoldshare`, `jng_goldspent`, `jng_gspd`, `jng_total cs`, `jng_minionkills`, `jng_monsterkills`, `jng_monsterkillsownjungle`, `jng_monsterkillsenemyjungle`, `jng_cspm`, `jng_goldat10`, `jng_xpat10`, `jng_csat10`, `jng_golddiffat10`, `jng_xpdiffat10`, `jng_csdiffat10`, `jng_killsat10`, `jng_assistsat10`, `jng_deathsat10`,`jng_goldat15`, `jng_xpat15`, `jng_csat15`, `jng_golddiffat15`, `jng_xpdiffat15`, `jng_csdiffat15`, `jng_killsat15`, `jng_assistsat15`, `jng_deathsat15` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'l1_ratio':[.1, .5, .7, .9, .95, .99]}

    clf = GridSearchCV(LogisticRegression(penalty = 'elasticnet', random_state = 2, solver = 'saga', max_iter = 10000), param_grid)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def LogWithoutJungle():
    print('Log predict score without jungle data')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `kills`, `deaths`, `assists`, `team kpm`, `dragons`,`elementaldrakes`, `infernals`, `mountains`, `clouds`, `oceans`, `elders`, `heralds`, `barons`, `towers`, `inhibitors`, `damagetochampions`, `dpm`,`damagetakenperminute`, `damagemitigatedperminute`, `wardsplaced`, `wpm`, `wardskilled`, `wcpm`, `controlwardsbought`, `visionscore`, `vspm`, `totalgold`, `earnedgold`, `earned gpm`, `goldspent`, `gspd`, `minionkills`, `monsterkills`, `monsterkillsownjungle`, `monsterkillsenemyjungle`, `cspm`, `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'l1_ratio':[.99]}

    clf = GridSearchCV(LogisticRegression(penalty = 'elasticnet', random_state = 2, solver = 'saga', max_iter = 10000), param_grid)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def Log10jungle():
    time_start = time.time()
    print('Log predict score with jungle data')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `wpm`, `wcpm`, `vspm`, `jng_goldat10`, `jng_xpat10`, `jng_csat10`, `jng_golddiffat10`, `jng_xpdiffat10`, `jng_csdiffat10`, `jng_killsat10`, `jng_assistsat10`, `jng_deathsat10`, `jng_wpm`, `jng_wcpm`, `jng_vspm` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'l1_ratio':[.5]}

    clf = GridSearchCV(LogisticRegression(penalty = 'elasticnet', random_state = 2, solver = 'saga', max_iter = 10000), param_grid)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)

def Log10():
    time_start = time.time()
    print('Log 10')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `wpm`, `wcpm`, `vspm` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'l1_ratio':[.1, .5, .7, .9, .95, .99]}

    clf = GridSearchCV(LogisticRegression(penalty = 'elasticnet', random_state = 2, solver = 'saga', max_iter = 10000), param_grid)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)

def Log15jungle():
    time_start = time.time()
    print('Log predict score with jungle data')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT wpm, wcpm, vspm, goldat15, xpat15, csat15, golddiffat15, xpdiffat15, csdiffat15, killsat15, assistsat15, deathsat15, jng_goldat15`, jng_xpat15, jng_csat15, jng_golddiffat15, jng_xpdiffat15, jng_csdiffat15, jng_killsat15, jng_assistsat15, jng_deathsat15, jng_wpm, jng_wcpm, jng_vspm FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'l1_ratio':[.5]}

    clf = GridSearchCV(LogisticRegression(penalty = 'elasticnet', random_state = 2, solver = 'saga', max_iter = 10000), param_grid)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)

def Log15():
    time_start = time.time()
    print('Log 15')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `wpm`, `wcpm`, `vspm`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'l1_ratio':[.1, .5, .7, .9, .95, .99]}

    clf = GridSearchCV(LogisticRegression(penalty = 'elasticnet', random_state = 2, solver = 'saga', max_iter = 10000), param_grid)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.best_params_)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)