# filename: preprocess.py
# -*- coding: utf-8 -*-

import pandas as pd
    

def offline_data_explore():
    
    rawdata = pd.read_csv('rawdata/ccf_offline_stage1_train.csv')
    unique_date = rawdata['Date'].drop_duplicates()
    data_group = rawdata.groupby('Date')
    offline_list = []
    
    for date in unique_date:
        if date == 'null':
            continue
        
        data_piece = data_group.get_group(date)
        normal_piece = data_piece[data_piece['Date_received'] == 'null']
        normal_count = normal_piece['User_id'].count()
        
        offline_list.append({'Date':date, 'normal_count':normal_count})
    
    offline_list_df = pd.DataFrame(offline_list)
    offline_list_df.sort_values(by=['Date'], inplace=True)
    columns = ['Date', 'normal_count']
    offline_list_df.to_csv('month_explore.csv', index=False, columns=columns)
        

def clean_duplicates():
    
    data = pd.read_csv('train/train_6.csv')
    
    col1 = ['User_id','Coupon_id','Date_received','label']
    col2 = ['User_id','Coupon_id','Date_received']
    raw = data[~data.duplicated(subset=col1)]
    data1 = raw[raw.duplicated(subset=col2, keep=False)]
    data2 = raw[~raw.duplicated(subset=col2, keep=False)]
    positive = data1[data1['label'] == 1]
    clean = pd.concat([data2, positive])
    clean = clean.sample(frac=1.0)
    clean.to_csv('train/clean_train_6.csv', index=False)
    
    

def final_data():
    
    data1 = pd.read_csv('train/clean_train_4.csv')
    data2 = pd.read_csv('train/clean_train_4_5.csv')
    data3 = pd.read_csv('train/clean_train_5.csv')
    data4= pd.read_csv('train/clean_train_5_6.csv')
    #data5= pd.read_csv('train/clean_train_6.csv')
    data = pd.concat([data1, data2, data3, data4])
    
    X = data.iloc[:, 3:92]
    X = X.drop_duplicates()
    
    cols = ['c1','c2','c3','c4','c5','c6','d1','d2','d3','d4','d5',
            'd6','d7','d8','d9','d10','d11','d12','d13','d14','d15','d16','d17',
            'u1','u2','u3','u4','u5','u6','u7','u8','u9','u10','u11','u12','u13',
            'u14','u15','u16','u17','u18','u19','u20','u21',
            'm1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','m13',
            'm14','m15','m16','m17','m18','m19','m20','m21',
            'um1','um2','um3','um4','um5','um6','um7','um8','um9','um10',
            'uo1','uo2','uo3','uo4','uo5','uo6','uo7',
            'hc1','hc2','hc3','uc1','uc2','uc3']
    
    final = X[~X.duplicated(subset=cols, keep=False)]
    final.to_csv('train/final_train_cv.csv', index=False)
    
    
    

def concat_data():
    
    current = pd.read_csv('train/current/v1/current_6.csv')
    history = pd.read_csv('train/history/v1/history_6.csv')
    data = pd.concat([current, history], axis=1)
    data1 = pd.read_csv('train/history/v2/history_6.csv')
    data['u21'] = data1['u21']
    cols = ['User_id','Coupon_id','Date_received','label',
            'c1','c2','c3','c4','c5','c6','d1','d2','d3','d4','d5',
            'd6','d7','d8','d9','d10','d11','d12','d13','d14','d15','d16','d17',
            'u1','u2','u3','u4','u5','u6','u7','u8','u9','u10','u11','u12','u13',
            'u14','u15','u16','u17','u18','u19','u20','u21',
            'm1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','m13',
            'm14','m15','m16','m17','m18','m19','m20','m21',
            'um1','um2','um3','um4','um5','um6','um7','um8','um9','um10',
            'uo1','uo2','uo3','uo4','uo5','uo6','uo7',
            'hc1','hc2','hc3','uc1','uc2','uc3']
    result = data[cols]
    #data.to_csv('train/raw/test_raw.csv', index=False)
    clean = result.fillna(-1)
    clean.to_csv('train/train_6.csv', index=False)

    



if __name__=='__main__':
    
    #clean_duplicates()
    #concat_data()
    final_data()
    