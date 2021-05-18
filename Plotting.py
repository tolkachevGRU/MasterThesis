from seaborn import palettes
import Database
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy
import numpy as np
import dash
import html
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import seaborn as sb
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly as py
import matplotlib.pyplot as plt
import keras
import seaborn as sns
import time
import time
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report

def plotHistograms():
    x = Database.query_mysql("SELECT `result`,`gamelength`, `kills`, `deaths`, `assists`, `teamkills`, `teamdeaths`, `firstbloodkill`, `firstbloodassist`, `firstbloodvictim`, `team kpm`, `ckpm`, `firstdragon`, `dragons`, `opp_dragons`, `elementaldrakes`, `opp_elementaldrakes`, `infernals`, `mountains`, `clouds`, `oceans`, `dragons (type unknown)`, `elders`, `opp_elders`, `firstherald`, `heralds`, `opp_heralds`, `firstbaron`, `barons`, `opp_barons`, `firsttower`, `towers`, `opp_towers`, `firstmidtower`, `firsttothreetowers`, `inhibitors`, `opp_inhibitors`, `damagetochampions`, `dpm`, `damageshare`, `damagetakenperminute`, `damagemitigatedperminute`, `wardsplaced`, `wpm`, `wardskilled`, `wcpm`, `controlwardsbought`, `visionscore`, `vspm`, `totalgold`, `earnedgold`, `earned gpm`, `earnedgoldshare`, `goldspent`, `gspd`, `total cs`, `minionkills`, `monsterkills`, `monsterkillsownjungle`, `monsterkillsenemyjungle`, `cspm`, `goldat10`, `xpat10`, `csat10`, `opp_goldat10`, `opp_xpat10`, `opp_csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `opp_killsat10`, `opp_assistsat10`, `opp_deathsat10`, `goldat15`, `xpat15`, `csat15`, `opp_goldat15`, `opp_xpat15`, `opp_csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15`, `opp_killsat15`, `opp_assistsat15`, `opp_deathsat15`, `jng_damagetochampions`, `jng_dpm`, `jng_damageshare`, `jng_damagetakenperminute`, `jng_damagemitigatedperminute`, `jng_wardsplaced`, `jng_wpm`, `jng_wardskilled`, `jng_wcpm`, `jng_controlwardsbought`, `jng_visionscore`, `jng_vspm`, `jng_totalgold`, `jng_earnedgold`, `jng_earned gpm`, `jng_earnedgoldshare`, `jng_goldspent`, `jng_gspd`, `jng_total cs`, `jng_minionkills`, `jng_monsterkills`, `jng_monsterkillsownjungle`, `jng_monsterkillsenemyjungle`, `jng_cspm`, `jng_goldat10`, `jng_xpat10`, `jng_csat10`, `jng_opp_goldat10`, `jng_opp_xpat10`, `jng_opp_csat10`, `jng_golddiffat10`, `jng_xpdiffat10`, `jng_csdiffat10`, `jng_killsat10`, `jng_assistsat10`, `jng_deathsat10`, `jng_opp_killsat10`, `jng_opp_assistsat10`, `jng_opp_deathsat10`, `jng_goldat15`, `jng_xpat15`, `jng_csat15`, `jng_opp_goldat15`, `jng_opp_xpat15`, `jng_opp_csat15`, `jng_golddiffat15`, `jng_xpdiffat15`, `jng_csdiffat15`, `jng_killsat15`, `jng_assistsat15`, `jng_deathsat15`, `jng_opp_killsat15`, `jng_opp_assistsat15`, `jng_opp_deathsat15` FROM `dataset`.`matches_2020_team_jng_merged`  ORDER BY RAND() ;", False)
    headers = Database.query_headers("SELECT `result`,`gamelength`, `kills`, `deaths`, `assists`, `teamkills`, `teamdeaths`, `firstbloodkill`, `firstbloodassist`, `firstbloodvictim`, `team kpm`, `ckpm`, `firstdragon`, `dragons`, `opp_dragons`, `elementaldrakes`, `opp_elementaldrakes`, `infernals`, `mountains`, `clouds`, `oceans`, `dragons (type unknown)`, `elders`, `opp_elders`, `firstherald`, `heralds`, `opp_heralds`, `firstbaron`, `barons`, `opp_barons`, `firsttower`, `towers`, `opp_towers`, `firstmidtower`, `firsttothreetowers`, `inhibitors`, `opp_inhibitors`, `damagetochampions`, `dpm`, `damageshare`, `damagetakenperminute`, `damagemitigatedperminute`, `wardsplaced`, `wpm`, `wardskilled`, `wcpm`, `controlwardsbought`, `visionscore`, `vspm`, `totalgold`, `earnedgold`, `earned gpm`, `earnedgoldshare`, `goldspent`, `gspd`, `total cs`, `minionkills`, `monsterkills`, `monsterkillsownjungle`, `monsterkillsenemyjungle`, `cspm`, `goldat10`, `xpat10`, `csat10`, `opp_goldat10`, `opp_xpat10`, `opp_csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `opp_killsat10`, `opp_assistsat10`, `opp_deathsat10`, `goldat15`, `xpat15`, `csat15`, `opp_goldat15`, `opp_xpat15`, `opp_csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15`, `opp_killsat15`, `opp_assistsat15`, `opp_deathsat15`, `jng_damagetochampions`, `jng_dpm`, `jng_damageshare`, `jng_damagetakenperminute`, `jng_damagemitigatedperminute`, `jng_wardsplaced`, `jng_wpm`, `jng_wardskilled`, `jng_wcpm`, `jng_controlwardsbought`, `jng_visionscore`, `jng_vspm`, `jng_totalgold`, `jng_earnedgold`, `jng_earned gpm`, `jng_earnedgoldshare`, `jng_goldspent`, `jng_gspd`, `jng_total cs`, `jng_minionkills`, `jng_monsterkills`, `jng_monsterkillsownjungle`, `jng_monsterkillsenemyjungle`, `jng_cspm`, `jng_goldat10`, `jng_xpat10`, `jng_csat10`, `jng_opp_goldat10`, `jng_opp_xpat10`, `jng_opp_csat10`, `jng_golddiffat10`, `jng_xpdiffat10`, `jng_csdiffat10`, `jng_killsat10`, `jng_assistsat10`, `jng_deathsat10`, `jng_opp_killsat10`, `jng_opp_assistsat10`, `jng_opp_deathsat10`, `jng_goldat15`, `jng_xpat15`, `jng_csat15`, `jng_opp_goldat15`, `jng_opp_xpat15`, `jng_opp_csat15`, `jng_golddiffat15`, `jng_xpdiffat15`, `jng_csdiffat15`, `jng_killsat15`, `jng_assistsat15`, `jng_deathsat15`, `jng_opp_killsat15`, `jng_opp_assistsat15`, `jng_opp_deathsat15` FROM `dataset`.`matches_2020_team_jng_merged`  ORDER BY RAND() ;")

    x = numpy.asarray(x).astype(numpy.float32)

    fig = go.Figure()

    for header in range(0, len(headers)):
        # Predicting the training set
        # result through scatter plot 
        df = x[:,header]
        fig.add_trace(go.Histogram(x=df, name=headers[header], text=headers[header]))

    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig.show()

def plotSeriesCross():
    y = Database.query_mysql("SELECT `side` from matches_2020_team_jng_merged", False)
    x = Database.query_mysql("SELECT `kills`, `deaths`, `assists`, `teamkills`, `teamdeaths`, `firstbloodkill`, `firstbloodassist`, `firstbloodvictim`, `team kpm`, `ckpm` FROM `dataset`.`matches_2020_team_jng_merged`;", False)
    headersx = Database.query_headers("SELECT `kills`, `deaths`, `assists`, `teamkills`, `teamdeaths`, `firstbloodkill`, `firstbloodassist`, `firstbloodvictim`, `team kpm`, `ckpm` FROM `dataset`.`matches_2020_team_jng_merged`;")
    headersy = Database.query_headers("SELECT `side` from matches_2020_team_jng_merged;")

    
    x = numpy.asarray(x).astype(numpy.float32)

    fig = go.Figure()
    model = LinearRegression()
    model.fit(x, y)


    for headerx in range(0, len(headersx)):
        for headery in range(0, len(headersy)):
            # Predicting the training set
            # result through scatter plot 
            dx = x[:,headerx]
            dy = y[:,headery]

            fig.add_trace(go.Scatter(x=dx, y=dy, name=headersy[headery]+'_'+headersx[headerx], text=headersx[headerx], mode='markers'))


    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig.show()

def plotGamesPerLeague():
    x = Database.query_mysql("SELECT `result` FROM matches_2020_team_jng_merged WHERE `side` = 'Blue';", False)

    headers = Database.query_headers("SELECT `side` FROM matches_2020_team_jng_merged WHERE `side` = 'Blue';")

    x = numpy.asarray(x)

    fig = go.Figure()

    for header in range(0, len(headers)):
        # Predicting the training set
        # result through scatter plot 
        df = header
        fig.add_trace(go.Histogram(x=df, name='Game count', text='Game count'))

    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig.show()    

def plotWinrate():
    red_wins = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE side = 'Red' AND result = 1;", False)
    blue_wins = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE side = 'Blue' AND result = 1;", False)

    data = np.empty([2])
    total_wins = len(red_wins) + len(blue_wins)
    data[0] = len(red_wins) / total_wins * 100.0
    data[1] = len(blue_wins)  / total_wins * 100.0
    
    labels = ["Red", "Blue"]
    barlist = plt.bar(labels, data, align='center', alpha=0.5)
    
    # set car colors and add text
    barlist[0].set_color('r')
    barlist[1].set_color('b')
    
    plt.title("Percentage of wins per side")
    plt.ylabel('Wins (Percentage)')
    plt.show()  

#def plotTest():
    x = Database.query_mysql("SELECT result  FROM `dataset`.`matches_2020_team_jng_merged`  ORDER BY RAND() ;", False)
    headers = Database.query_headers("SELECT side FROM `dataset`.`matches_2020_team_jng_merged`  ORDER BY RAND() ;")

    object  = x
    y_pos = np.arange(len(x))
    #plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, object)
    plt.ylabel('Wins')
    plt.title('Distribution of wins')

    plt.show()


def RF15jungleplot():
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


    model = RandomForestClassifier(n_estimators=18, criterion='entropy', max_depth=7, max_features='auto', random_state=42)
    

    # Fit on training data
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)
    plt.title('Feature Importance "team-jungle dataset"')
    plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
    plt.yticks(range(len(sorted_indices)), [headersx[i] for i in sorted_indices])
    plt.xlabel("Relative importance")
    plt.tight_layout()
    plt.show()

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)

def RF15plot():
    time_start = time.time()

    print('RF 15 jungle')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `wpm`, `wcpm`, `vspm`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    headersx = [ 'wpm', 'wcpm', 'vspm', 'goldat15', 'xpat15', 'csat15', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15']
    print(headersx)
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = RandomForestClassifier(n_estimators=18, criterion='gini', max_depth=9, max_features='auto', random_state=42)
    

    # Fit on training data
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)
    plt.title('Feature Importance "team dataset"')
    plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
    plt.yticks(range(len(sorted_indices)), [headersx[i] for i in sorted_indices])
    plt.xlabel("Relative importance")
    plt.tight_layout()
    plt.show()

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)


def RFalljungleplot():
    time_start = time.time()

    print('RF all jungle')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `kills`, `deaths`, `assists`, `team kpm`, `dragons`,`elementaldrakes`, `infernals`, `mountains`, `clouds`, `oceans`, `elders`, `heralds`, `barons`, `towers`, `inhibitors`, `damagetochampions`, `dpm`,`damagetakenperminute`, `damagemitigatedperminute`, `wardsplaced`, `wpm`, `wardskilled`, `wcpm`, `controlwardsbought`, `visionscore`, `vspm`, `totalgold`, `earnedgold`, `earned gpm`, `goldspent`, `gspd`, `minionkills`, `monsterkills`, `monsterkillsownjungle`, `monsterkillsenemyjungle`, `cspm`, `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15`, `jng_damagetochampions`, `jng_dpm`, `jng_damageshare`, `jng_damagetakenperminute`, `jng_damagemitigatedperminute`, `jng_wardsplaced`, `jng_wpm`, `jng_wardskilled`, `jng_wcpm`, `jng_controlwardsbought`, `jng_visionscore`, `jng_vspm`, `jng_totalgold`, `jng_earnedgold`, `jng_earned gpm`, `jng_earnedgoldshare`, `jng_goldspent`, `jng_gspd`, `jng_total cs`, `jng_minionkills`, `jng_monsterkills`, `jng_monsterkillsownjungle`, `jng_monsterkillsenemyjungle`, `jng_cspm`, `jng_goldat10`, `jng_xpat10`, `jng_csat10`, `jng_golddiffat10`, `jng_xpdiffat10`, `jng_csdiffat10`, `jng_killsat10`, `jng_assistsat10`, `jng_deathsat10`,`jng_goldat15`, `jng_xpat15`, `jng_csat15`, `jng_golddiffat15`, `jng_xpdiffat15`, `jng_csdiffat15`, `jng_killsat15`, `jng_assistsat15`, `jng_deathsat15` FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    headersx = ['kills', 'deaths', 'assists', 'team kpm', 'dragons', 'elementaldrakes', 'infernals', 'mountains', 'clouds', 'oceans', 'elders', 'heralds', 'barons', 'towers', 'inhibitors', 'damagetochampions', 'dpm','damagetakenperminute', 'damagemitigatedperminute', 'wardsplaced', 'wpm', 'wardskilled', 'wcpm', 'controlwardsbought', 'visionscore', 'vspm', 'totalgold', 'earnedgold', 'earned gpm', 'goldspent', 'gspd', 'minionkills', 'monsterkills', 'monsterkillsownjungle', 'monsterkillsenemyjungle', 'cspm', 'goldat10', 'xpat10', 'csat10', 'golddiffat10', 'xpdiffat10', 'csdiffat10', 'killsat10', 'assistsat10', 'deathsat10', 'goldat15', 'xpat15', 'csat15', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15', 'jng_damagetochampions', 'jng_dpm', 'jng_damageshare', 'jng_damagetakenperminute', 'jng_damagemitigatedperminute', 'jng_wardsplaced', 'jng_wpm', 'jng_wardskilled', 'jng_wcpm', 'jng_controlwardsbought', 'jng_visionscore', 'jng_vspm', 'jng_totalgold', 'jng_earnedgold', 'jng_earned gpm', 'jng_earnedgoldshare', 'jng_goldspent', 'jng_gspd', 'jng_total cs', 'jng_minionkills', 'jng_monsterkills', 'jng_monsterkillsownjungle', 'jng_monsterkillsenemyjungle', 'jng_cspm', 'jng_goldat10', 'jng_xpat10', 'jng_csat10', 'jng_golddiffat10', 'jng_xpdiffat10', 'jng_csdiffat10', 'jng_killsat10', 'jng_assistsat10', 'jng_deathsat10','jng_goldat15', 'jng_xpat15', 'jng_csat15', 'jng_golddiffat15', 'jng_xpdiffat15', 'jng_csdiffat15', 'jng_killsat15', 'jng_assistsat15', 'jng_deathsat15']
    print(headersx)
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = RandomForestClassifier(n_estimators=80, criterion='entropy', max_depth=12, max_features='auto', random_state=42)
    

    # Fit on training data
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    sort_indices = np.argsort(importances)
    sorted_indices = sort_indices[:10]
    plt.title('Feature Importance "team-jungle dataset"')
    plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
    plt.yticks(range(len(sorted_indices)), [headersx[i] for i in sorted_indices])
    plt.xlabel("Relative importance")
    plt.tight_layout()
    plt.show()

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)

def RFallplot():
    time_start = time.time()

    print('RF all jungle')
    y = Database.query_mysql("SELECT result FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    x = Database.query_mysql("SELECT `kills`, `deaths`, `assists`, `team kpm`, `dragons`,`elementaldrakes`, `infernals`, `mountains`, `clouds`, `oceans`, `elders`, `heralds`, `barons`, `towers`, `inhibitors`, `damagetochampions`, `dpm`,`damagetakenperminute`, `damagemitigatedperminute`, `wardsplaced`, `wpm`, `wardskilled`, `wcpm`, `controlwardsbought`, `visionscore`, `vspm`, `totalgold`, `earnedgold`, `earned gpm`, `goldspent`, `gspd`, `minionkills`, `monsterkills`, `monsterkillsownjungle`, `monsterkillsenemyjungle`, `cspm`, `goldat10`, `xpat10`, `csat10`, `golddiffat10`, `xpdiffat10`, `csdiffat10`, `killsat10`, `assistsat10`, `deathsat10`, `goldat15`, `xpat15`, `csat15`, `golddiffat15`, `xpdiffat15`, `csdiffat15`, `killsat15`, `assistsat15`, `deathsat15`  FROM `dataset`.`matches_2020_team_jng_merged` WHERE `side` = 'Blue';", False)
    headersx = ['kills', 'deaths', 'assists', 'team kpm', 'dragons', 'elementaldrakes', 'infernals', 'mountains', 'clouds', 'oceans', 'elders', 'heralds', 'barons', 'towers', 'inhibitors', 'damagetochampions', 'dpm','damagetakenperminute', 'damagemitigatedperminute', 'wardsplaced', 'wpm', 'wardskilled', 'wcpm', 'controlwardsbought', 'visionscore', 'vspm', 'totalgold', 'earnedgold', 'earned gpm', 'goldspent', 'gspd', 'minionkills', 'monsterkills', 'monsterkillsownjungle', 'monsterkillsenemyjungle', 'cspm', 'goldat10', 'xpat10', 'csat10', 'golddiffat10', 'xpdiffat10', 'csdiffat10', 'killsat10', 'assistsat10', 'deathsat10', 'goldat15', 'xpat15', 'csat15', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'assistsat15', 'deathsat15']
    print(headersx)
    # Convert to numpy array
    X = np.array(x).astype('float32')
    y = np.array(y)

    y = y.ravel()

    scaler = StandardScaler() 
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = RandomForestClassifier(n_estimators=70, criterion='entropy', max_depth=12, max_features='auto', min_features = 10, random_state=42)
    

    # Fit on training data
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    sort_indices = np.argsort(importances)
    sorted_indices = sort_indices[:10]
    plt.title('Feature Importance "team dataset"')
    plt.barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
    plt.yticks(range(len(sorted_indices)), [headersx[i] for i in sorted_indices])
    plt.xlabel("Relative importance")
    plt.tight_layout()
    plt.show()

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    end = time.time()
    print(end-time_start)