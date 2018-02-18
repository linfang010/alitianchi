# filename: average.py
# -*- coding: utf-8 -*-

import pandas as pd


def average():
    r1 = pd.read_csv('temp/rf_result_875.csv')
    r2 = pd.read_csv('temp/rf_result_875(1).csv')
    r3 = pd.read_csv('temp/rf_result_617.csv')
    r4 = pd.read_csv('temp/rf_result_282.csv')
    r5 = pd.read_csv('temp/rf_result_352.csv')
    r6 = pd.read_csv('temp/gbdt_result_351.csv')
    r7 = pd.read_csv('temp/xgb_result_722.csv')
    
    
    prob_matrix = pd.DataFrame()
    prob_matrix['r1'] = r1['rank']
    prob_matrix['r2'] = r2['rank']
    prob_matrix['r3'] = r3['rank']
    prob_matrix['r4'] = r4['rank']
    prob_matrix['r5'] = r5['rank']
    prob_matrix['r6'] = r6['rank']
    prob_matrix['r7'] = r7['rank']
    
    #corr = prob_matrix.corr(method='pearson')
    avg_prob = prob_matrix.mean(axis=1)
    data = pd.read_csv('rawdata/ccf_offline_stage1_test_sorted.csv')
    data['prob'] = avg_prob
    result_data = data[['User_id','Coupon_id','Date_received','prob']]
    result_data.to_csv('result/average_rank.csv', index=False)


def rank():
    
    data = pd.read_csv('temp/rf_result_875_0.80055397.csv')
    rank = data['prob'].rank(method='first')
    max_rank = rank.max()
    normalised_rank = rank.apply(lambda x: x/max_rank)
    data['rank'] = normalised_rank
    result_data = data[['User_id','Coupon_id','Date_received','rank']]
    result_data.to_csv('temp/rf_result_875(1).csv', index=False)
    

if __name__ == '__main__':
    
    average()
    #rank()

    
    
