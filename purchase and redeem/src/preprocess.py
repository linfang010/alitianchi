# filename: preprocess.py
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import timedelta
    

def merge_data():
    
    merge_list = []
    data = pd.read_csv('rawdata/user_balance_table.csv')
    report_date = data['report_date'].drop_duplicates()
    data_group = data.groupby('report_date')
    
    for date in report_date:
        
        data_piece = data_group.get_group(date)
        total_purchase = data_piece['total_purchase_amt'].sum()
        total_redeem = data_piece['total_redeem_amt'].sum()
        total_today = data_piece['tBalance'].sum()
        total_yesterday = data_piece['yBalance'].sum()
        total_direct_purchase = data_piece['direct_purchase_amt'].sum()
        total_purchase_bal = data_piece['purchase_bal_amt'].sum()
        total_purchase_bank = data_piece['purchase_bank_amt'].sum()
        total_consume = data_piece['consume_amt'].sum()
        total_transfer = data_piece['transfer_amt'].sum()
        total_tftobal = data_piece['tftobal_amt'].sum()
        total_tftocard = data_piece['tftocard_amt'].sum()
        total_share = data_piece['share_amt'].sum()
        total_category1 = data_piece['category1'].sum()
        total_category2 = data_piece['category2'].sum()
        total_category3 = data_piece['category3'].sum()
        total_category4 = data_piece['category4'].sum()
        
        
        merge_list.append({'date':date, 'total_purchase':total_purchase, 'total_redeem':total_redeem, 'total_today':total_today,
                           'total_yesterday':total_yesterday, 'total_direct_purchase':total_direct_purchase, 'total_purchase_bal':total_purchase_bal,
                           'total_purchase_bank':total_purchase_bank, 'total_consume':total_consume, 'total_transfer':total_transfer,
                           'total_tftobal':total_tftobal, 'total_tftocard':total_tftocard, 'total_share':total_share, 
                           'total_category1':total_category1, 'total_category2':total_category2, 'total_category3':total_category3, 'total_category4':total_category4})
    
    merge_df = pd.DataFrame(merge_list)
    merge_df.sort_values(by=['date'], inplace=True)
    columns = ['date', 'total_purchase', 'total_redeem', 'total_today', 'total_yesterday', 'total_direct_purchase', 'total_purchase_bal', 'total_purchase_bank',
               'total_consume', 'total_transfer', 'total_tftobal', 'total_tftocard', 'total_share', 'total_category1', 'total_category2', 'total_category3', 'total_category4']
    merge_df.to_csv('train/train_data.csv', index=False, columns=columns)


def extend_shibor():
    
    data = pd.read_csv('rawdata/mfd_bank_shibor.csv')
    date = pd.to_datetime('20130701')
    result = []
    data_group = data.groupby(['mfd_date'])
    
    for i in range(427):
        
        int_date = int(date.strftime('%Y%m%d'))
        
        try:
            piece = data_group.get_group(int_date)
            result.append({'mfd_date':int_date,'Interest_O_N':piece['Interest_O_N'].item(),'Interest_1_W':piece['Interest_1_W'].item(),'Interest_2_W':piece['Interest_2_W'].item(),
                           'Interest_1_M':piece['Interest_1_M'].item(),'Interest_3_M':piece['Interest_3_M'].item(),'Interest_6_M':piece['Interest_6_M'].item(),
                           'Interest_9_M':piece['Interest_9_M'].item(),'Interest_1_Y':piece['Interest_1_Y'].item()})
            
        except Exception as e:
            result.append({'mfd_date':int_date,'Interest_O_N':0,'Interest_1_W':0,'Interest_2_W':0,
                           'Interest_1_M':0,'Interest_3_M':0,'Interest_6_M':0,'Interest_9_M':0,'Interest_1_Y':0})
            
        date = date + timedelta(days=1)
    
    cols = ['mfd_date','Interest_O_N','Interest_1_W','Interest_2_W','Interest_1_M','Interest_3_M',
            'Interest_6_M','Interest_9_M','Interest_1_Y']
    result_df = pd.DataFrame(result)
    result_df.to_csv('train/shibor_extend.csv', index=False, columns=cols)
    
    
def time_features():
    
    time_feature = []
    data = pd.read_csv('train/train_data.csv')
    date_series = data['date'].apply(lambda x: pd.to_datetime(str(x)))
    
    for date in date_series:
        week_day = date.isoweekday()
        month_day = date.day
        month = date.month
        week = date.week
        time_feature.append({'date':int(date.strftime('%Y%m%d')),'week_day':week_day,'month_day':month_day,
                             'month':month,'week':week})
    
    time_feature_df = pd.DataFrame(time_feature)
    cols = ['date','week_day','month_day','month','week']
    time_feature_df.to_csv('train/time_feature.csv', index=False, columns=cols)
        
    

if __name__=='__main__':
    
    #merge_data()
    #extend_shibor()
    time_features()
    