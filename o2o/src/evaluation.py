# filename: evaluation.py
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import roc_auc_score

'''
# eval_df is a pandas dataframe with columns
[Coupon_id, label, prob]
'''
 
def mean_auc(eval_df):
    
    auc_list = []
    coupon_list = eval_df['Coupon_id'].drop_duplicates()
    
    for coupon_id in coupon_list:
        try:
            eval_piece = eval_df[eval_df['Coupon_id'] == coupon_id]
            auc = roc_auc_score(eval_piece['label'], eval_piece['prob'])
            auc_list.append({'auc':auc})
        
        except Exception as e:   
            continue
    
    # mean auc among coupon
    auc_df = pd.DataFrame(auc_list)
    return auc_df['auc'].mean()

        
    




    
    
