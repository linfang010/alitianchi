# filename: model_selection.py
# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score



def random_forest():
    
    best_auc = 0.0
    best_parameter = {'max_features':0}
    # parameter grid
    max_features_list = range(5, 21, 1)
    #max_depth_list = range(11, 16, 1)
    #min_samples_split_list = range(100, 1001, 100)
    #min_samples_leaf_list = range(10, 101, 10)
    #n_estimators_list = range(100, 101, 1)
    
    
    # train and test data
    train1 = pd.read_csv('train/clean_train_4.csv')
    train2 = pd.read_csv('train/clean_train_4_5.csv')
    test1 = pd.read_csv('train/clean_train_5.csv')
    test2 = pd.read_csv('train/clean_train_5_6.csv')
    X_train1 = train1.iloc[:, 4:92]
    X_train2 = train2.iloc[:, 4:92]
    y_train1 = train1['label']
    y_train2 = train2['label']
    X_test1 = test1.iloc[:, 4:92]
    X_test2 = test2.iloc[:, 4:92]
    y_test1 = test1['label']
    y_test2 = test2['label']

    # find best parameters
    for max_features in max_features_list:
        #for max_features in max_features_list:
            # random forest
            rf = RandomForestClassifier(n_estimators=100, criterion='entropy', class_weight='balanced', 
                                        max_features=max_features, max_depth=9,
                                        random_state=1, n_jobs=-1)
            
            rf.fit(X_train1, y_train1)
            y_predprob1 = rf.predict_proba(X_test1)[:, 1]
            auc_score1 = roc_auc_score(y_test1, y_predprob1)
            
            rf.fit(X_train2, y_train2)
            y_predprob2 = rf.predict_proba(X_test2)[:, 1]
            auc_score2 = roc_auc_score(y_test2, y_predprob2)
            
            auc_score = (auc_score1 + auc_score2) / 2
            print ('auc_score:%f' % auc_score)
            print ('max_features:%d' % max_features)
            
            if auc_score > best_auc:  
                best_auc = auc_score
                best_parameter['max_features'] = max_features
    
    print ('best auc score:%f' % best_auc)
    print ('best parameter:')
    print (best_parameter)
    


def GBDT():
    
    best_auc = 0.0
    best_parameter = {'max_depth':0}
    # parameter grid
    #min_samples_split_list = range(30, 101, 10)
    #min_samples_leaf_list = range(100, 1001, 100)
    #max_features_list = range(5, 21, 1)
    max_depth_list = range(3, 11, 1)

    # train and test data
    #data1 = pd.read_csv('train/clean_train_4.csv')
    #data2 = pd.read_csv('train/clean_train_4_5.csv')
    #data3 = pd.read_csv('train/clean_train_5.csv')
    #data4 = pd.read_csv('train/clean_train_5_6.csv')
    train_data = pd.read_csv('train/final_train_cv.csv')
    X_train = train_data.iloc[:, 1:89]
    y_train = train_data['label']
    test_data = pd.read_csv('train/clean_train_6.csv')
    X_test = test_data.iloc[:, 4:92]
    y_test = test_data['label']

    # find best parameters
    for max_depth in max_depth_list:
                # GBDT
                gb = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=8,
                                                max_features=12, min_samples_leaf=500,                                        
                                                random_state=1)
                
                gb.fit(X_train, y_train)
                y_predprob = gb.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_predprob)
                
                print ('auc_score:%f' % auc_score)
                print ('max_depth:%d' % max_depth)
                
                if auc_score > best_auc:  
                    best_auc = auc_score
                    best_parameter['max_depth'] = max_depth

    print ('best auc score:%f' % best_auc)
    print ('best parameter:')
    print (best_parameter)


    
    
    
if __name__ == '__main__':
    
    #random_forest()
    GBDT()
    
    
