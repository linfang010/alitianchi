# filename: feature_importance.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



def random_forest():
    # train data
    train_data = pd.read_csv('train/final_train_4_5.csv')
    X = train_data.iloc[:,1:35]
    y = train_data['label']
        
    # random forest
    rf = RandomForestClassifier(max_depth=10, n_estimators=200, max_features='sqrt', criterion='entropy',
                                class_weight='balanced', random_state=1, n_jobs=-1)
    
    rf.fit(X, y)
    
        
    # feature importances
    importances = rf.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    columns = X.columns.values
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))
    
    '''
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    '''



def GBDT():
    
    train_data = pd.read_csv('traindata/train_data_4_5.csv')
    X = train_data.iloc[:,1:95]
    y = train_data['label']
    
    # GBDT
    gb = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, max_features='sqrt', n_estimators=100,
                                    subsample=0.7)
    
    gb.fit(X, y)
    
    # feature importances
    importances = gb.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    columns = X.columns.values
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))
    



if __name__ == '__main__':
    
    random_forest()
    #GBDT()
    
