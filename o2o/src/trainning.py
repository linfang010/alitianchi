# filename: trainning.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import xgboost as xgb
import random as rd


def xgboost():

    #data1 = pd.read_csv('train/clean_train_4.csv')
    #data2 = pd.read_csv('train/clean_train_4_5.csv')
    #data3 = pd.read_csv('train/clean_train_5.csv')
    #data4 = pd.read_csv('train/clean_train_5_6.csv')
    #data5 = pd.read_csv('train/clean_train_6.csv')
    train_data = pd.read_csv('train/final_train.csv')
    #validation_data = pd.read_csv('train/clean_train_6.csv')
    #test_data = pd.read_csv('traindata/test.csv')
    X_train = train_data.iloc[:, 1:89]
    #X_val = validation_data.iloc[:, 4:92]
    #X_test = test_data.iloc[:, 1:88]
    y_train = train_data['label']
    #y_val = validation_data['label']
    #y_test = test_data['label']

    # xgboost
    seed = rd.randint(0,1000)
    print (seed)
    params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'gamma': 0.2,
    'max_depth': 3,
    'scale_pos_weight': 9,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 1, 
    'eta': 0.01,
    'seed': seed,
    'nthread': 7,
    'eval_metric': 'auc'
    }

    num_rounds = 1000

    xgb_train = xgb.DMatrix(X_train, label=y_train)
    #xgb_val = xgb.DMatrix(X_val, label=y_val)
    #xgb_test = xgb.DMatrix(X_test, label=y_test)
    
    watchlist = [(xgb_train, 'train')]
    
    model = xgb.train(params, xgb_train, num_rounds, watchlist,
                      early_stopping_rounds=100)

    #preds = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
    #print ('AUC Score (test):%f' % roc_auc_score(y_test, preds))
    
    # test
    data = pd.read_csv('train/test.csv')
    X_test = data.iloc[:, 3:91]
    xgb_test = xgb.DMatrix(X_test)    
    preds = model.predict(xgb_test)
    data['prob'] = preds
    result_data = data[['User_id','Coupon_id','Date_received','prob']]
    result_data.to_csv('result/xgb_result.csv', index=False)
    


def random_forest():
    
    data1 = pd.read_csv('train/clean_train_4.csv')
    data2 = pd.read_csv('train/clean_train_4_5.csv')
    data3 = pd.read_csv('train/clean_train_5.csv')
    data4 = pd.read_csv('train/clean_train_5_6.csv')
    data5 = pd.read_csv('train/clean_train_6.csv')
    train_data = pd.concat([data1, data2, data3, data4, data5])
    #test_data = pd.read_csv('train/clean_train_5_6.csv')
    X_train = train_data.iloc[:, 4:92]
    #X_test = test_data.iloc[:, 4:102]
    y_train = train_data['label']
    #y_test = test_data['label']
        
    # Standardlize
    #X = StandardScaler().fit_transform(X)
    
    # train test split
    #X_train, X_test, y_train, y_test = \
    #        train_test_split(X, y, test_size=.3, random_state=1)

    # random forest
    rs = rd.randint(0,1000)
    print (rs)
    rf = RandomForestClassifier(n_estimators=100, max_features=13, max_depth=9, criterion='entropy',
                                min_samples_split=5, min_samples_leaf=4, class_weight='balanced',
                                random_state=rs, n_jobs=-1)
    
    rf.fit(X_train, y_train)
    
    data = pd.read_csv('train/test.csv')
    X_test = data.iloc[:, 3:91]
    y_predprob = rf.predict_proba(X_test)[:, 1]
    
    data['prob'] = y_predprob
    #y_predprob_train = rf.predict_proba(X_train)[:, 1]
    #y_pred = rf.predict(X_test)
    result_data = data[['User_id','Coupon_id','Date_received','prob']]
    result_data.to_csv('result/rf_result.csv', index=False)
    
    #fpr, tpr, threshholds = roc_curve(y_test, y_predprob, pos_label=1)
    '''
    # plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    '''
    #print (auc(fpr, tpr))
    #print ('AUC Score (train):%f' % roc_auc_score(y_train, y_predprob_train))
    #print ('AUC Score (test):%f' % roc_auc_score(y_test, y_predprob))
    #cnf_matrix = confusion_matrix(y_test, y_pred, labels=[1,0])
    #print ('Confusion matrix:')
    #print (cnf_matrix)
    

    
def GBDT():
    
    #data1 = pd.read_csv('train/clean_train_4.csv')
    #data2 = pd.read_csv('train/clean_train_4_5.csv')
    #data3 = pd.read_csv('train/clean_train_5.csv')
    #data4 = pd.read_csv('train/clean_train_5_6.csv')
    #data5 = pd.read_csv('train/clean_train_6.csv')
    train_data = pd.read_csv('train/final_train.csv')
    X_train = train_data.iloc[:, 1:89]
    y_train = train_data['label']
    # GBDT
    rs = rd.randint(0,1000)
    print (rs)
    
    gb = GradientBoostingClassifier(learning_rate=0.02, n_estimators=500, max_depth=5, 
                                    max_features=11, min_samples_leaf=400, 
                                    random_state=rs)
    
    '''
    gb = GradientBoostingClassifier(learning_rate=0.01, n_estimators=1000, max_depth=8, 
                                    max_features=12, min_samples_leaf=500, 
                                    random_state=rs)
    '''
    gb.fit(X_train, y_train)
    #y_predprob1 = gb.predict_proba(X_test1)[:, 1]
    #auc_score1 = roc_auc_score(y_test1, y_predprob1)
    '''
    gb.fit(X_train2, y_train2)
    y_predprob2 = gb.predict_proba(X_test2)[:, 1]
    auc_score2 = roc_auc_score(y_test2, y_predprob2)
    '''
    #auc_score = (auc_score1 + auc_score2) / 2
    
    data = pd.read_csv('train/test.csv')
    X_test = data.iloc[:, 3:91]
    y_predprob = gb.predict_proba(X_test)[:, 1]
    
    data['prob'] = y_predprob
    result_data = data[['User_id','Coupon_id','Date_received','prob']]
    result_data.to_csv('result/gbdt_result.csv', index=False)
    
    #print ('AUC Score (test):%f' % auc_score)
    






if __name__ == '__main__':
    
    #random_forest()
    #GBDT()
    xgboost()
    
    
    
    
