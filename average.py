# filename: average.py
# -*- coding: utf-8 -*-

import pandas as pd


 
rf1 = pd.read_csv('result/rf_result1.csv')
rf2 = pd.read_csv('result/rf_result2.csv')
rf3 = pd.read_csv('result/rf_result3.csv')
et1 = pd.read_csv('result/et_result1.csv')
et2 = pd.read_csv('result/et_result2.csv')
et3 = pd.read_csv('result/et_result3.csv')
gb1 = pd.read_csv('result/gbdt_result1.csv')
gb2 = pd.read_csv('result/gbdt_result2.csv')
gb3 = pd.read_csv('result/gbdt_result3.csv')
xgb1 = pd.read_csv('result/xgb_result1.csv')
xgb2 = pd.read_csv('result/xgb_result2.csv')
xgb3 = pd.read_csv('result/xgb_result3.csv')

prob_matrix = pd.DataFrame()
prob_matrix['rf1'] = rf1['prob']
prob_matrix['rf2'] = rf2['prob']
prob_matrix['rf3'] = rf3['prob']
prob_matrix['et1'] = et1['prob']
prob_matrix['et2'] = et2['prob']
prob_matrix['et3'] = et3['prob']
prob_matrix['gb1'] = gb1['prob']
prob_matrix['gb2'] = gb2['prob']
prob_matrix['gb3'] = gb3['prob']
prob_matrix['xgb1'] = xgb1['prob']
prob_matrix['xgb2'] = xgb2['prob']
prob_matrix['xgb3'] = xgb3['prob']

#corr = prob_matrix.corr(method='pearson')
avg_prob = prob_matrix.mean(axis=1)
data = pd.read_csv('rawdata/ccf_offline_stage1_test_sorted.csv')
data['prob'] = avg_prob
result_data = data[['User_id','Coupon_id','Date_received','prob']]
result_data.to_csv('result/result.csv', index=False)




    
    
