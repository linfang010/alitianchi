# filename: feature_importance.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


from sklearn.feature_selection import chi2, f_classif, mutual_info_classif



def chi2test():
    # train data
    train_data = pd.read_csv('train/final_train_5_6.csv')
    X = train_data.iloc[:,1:35]
    y = train_data['label']
        
    # chi2
    result = chi2(X, y)
    chi2_value = result[0]
    indices = np.argsort(chi2_value)[::-1]
    columns = X.columns.values
    
    # Print the feature ranking
    print("Feature chi2 ranking:")
    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, columns[indices[f]], chi2_value[indices[f]]))


def ANOVA():
    # train data
    train_data = pd.read_csv('train/final_train_5_6.csv')
    X = train_data.iloc[:,1:35]
    y = train_data['label']
    
    # ANOVA
    result = f_classif(X, y)
    F_value = result[0]
    indices = np.argsort(F_value)[::-1]
    columns = X.columns.values
    
    # Print the feature ranking
    print("Feature F ranking:")
    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, columns[indices[f]], F_value[indices[f]]))
        

def mutual_info():
     # train data
    train_data = pd.read_csv('train/final_train_5_6.csv')
    X = train_data.iloc[:,1:35]
    y = train_data['label']
    
    # mutual into
    mi = mutual_info_classif(X, y)
    indices = np.argsort(mi)[::-1]
    columns = X.columns.values
    
    # Print the feature ranking
    print("Feature mi ranking:")
    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, columns[indices[f]], mi[indices[f]]))
    


if __name__ == '__main__':
    
    #chi2test()
    #ANOVA()
    mutual_info()
