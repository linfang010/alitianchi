# filename: export_tree.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


from sklearn.tree import DecisionTreeClassifier, export_graphviz



def export():
    # train data
    train_data = pd.read_csv('traindata/clean_train_data_5.csv')
    X = train_data.iloc[:,1:66]
    y = train_data['label']
    
    feature_names = X.columns.values
    # decition tree
    dc = DecisionTreeClassifier(max_depth=5, criterion='entropy', max_features=7, random_state=1)
    dc.fit(X, y)
        
    export_graphviz(dc, out_file='tree.dot', feature_names=feature_names)

if __name__ == '__main__':
    
    export()
    
