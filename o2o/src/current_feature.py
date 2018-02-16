# filename : current_feature.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np



def current_feature(data):
    
    current_month_feature = []
    # group data
    user_group = data.groupby('User_id')
    merchant_group = data.groupby('Merchant_id')
    
    for index,row in data.iterrows():
        try:
            
            # c1: day of week
            # c2: day of month
            Date_received = pd.to_datetime(str(row['Date_received']))
            c1 = Date_received.isoweekday()
            c2 = Date_received.day
            # c3: distance
            c3 = np.nan
            if row['Distance'] != 'null':
                c3 = int(row['Distance'])
            # c4: man
            # c5: rate
            # c6: type
            temp = row['Discount_rate'].split(':')
            if len(temp) == 2:
                c4 = int(temp[0])
                c5 = (float(temp[0]) - float(temp[1]))/float(temp[0])
                c6 = 1
            else:
                c4 = 0
                c5 = float(temp[0])
                c6 = 0
            

            # data leakage
            user_piece = user_group.get_group(row['User_id'])
            user_coupon_piece = user_piece[user_piece['Coupon_id'] == row['Coupon_id']]
            user_merchant_piece = user_piece[user_piece['Merchant_id'] == row['Merchant_id']]
            before_piece = user_piece[user_piece['Date_received'] < row['Date_received']]
            before_coupon_piece = before_piece[before_piece['Coupon_id'] == row['Coupon_id']]
            next_piece = user_piece[user_piece['Date_received'] > row['Date_received']]
            next_coupon_piece = next_piece[next_piece['Coupon_id'] == row['Coupon_id']]
            today_piece = user_piece[user_piece['Date_received'] == row['Date_received']]
            today_coupon_piece = today_piece[today_piece['Coupon_id'] == row['Coupon_id']]
            
            merchant_piece = merchant_group.get_group(row['Merchant_id'])
            merchant_coupon_piece = merchant_piece[merchant_piece['Coupon_id'] == row['Coupon_id']]
            
            # d1: coupon count in that month
            d1= user_piece['User_id'].count()
            # d2: specific coupon count in that month
            d2 = user_coupon_piece['User_id'].count()
            # d3: before coupon count
            # d5: time interval since last time
            d3 = before_piece['User_id'].count()
            d5 = np.nan
            if not before_piece.empty:
                last_time_row = before_piece.iloc[-1, :]
                last_time = pd.to_datetime(str(last_time_row['Date_received']))
                d5 = (Date_received - last_time).days
            # d4: next coupon count
            # d6: time interval until next time
            d4 = next_piece['User_id'].count()
            d6 = np.nan
            if not next_piece.empty:
                next_time_row = next_piece.iloc[0, :]
                next_time = pd.to_datetime(str(next_time_row['Date_received']))
                d6 = (next_time - Date_received).days
            # d7: user merchant coupon count
            d7 = user_merchant_piece['User_id'].count()
            # d8: unique merchant count by user
            d8 = user_piece['Merchant_id'].drop_duplicates().count()
            # d9: coupon count today
            d9 = today_piece['User_id'].count()
            # d10: specific coupon count today
            d10 = today_coupon_piece['User_id'].count()
            # d11: unique coupon count by user
            d11 = user_piece['Coupon_id'].drop_duplicates().count()
            # d12: coupon count by merchant in that month
            d12 = merchant_piece['User_id'].count()
            # d13: specific coupon count by merchant in that month
            d13 = merchant_coupon_piece['User_id'].count()
            # d14: unique user count by merchant
            d14 = merchant_piece['User_id'].drop_duplicates().count()
            # d15: unique coupon count by merchant
            d15 = merchant_piece['Coupon_id'].drop_duplicates().count()
            # d16: before specific coupon count
            d16 = before_coupon_piece['User_id'].count()
            # d17: next specific coupon count
            d17 = next_coupon_piece['User_id'].count()
            
            
            
            # label: 0:negtive sample 1:positive sample
            label = 0
            if row['Date'] != 'null':
                Date = pd.to_datetime(row['Date'])
                if (Date - Date_received).days <= 15:
                    label = 1
            
            
            # Assemble current month features
            current_month_feature.append({'User_id':row['User_id'],'Coupon_id':row['Coupon_id'],'Date_received':row['Date_received'],
                                          'label':label,'c1':c1,'c2':c2,'c3':c3,'c4':c4,'c5':c5,'c6':c6,
                                          'd1':d1,'d2':d2,'d3':d3,'d4':d4,'d5':d5,'d6':d6,'d7':d7,'d8':d8,'d9':d9,
                                          'd10':d10,'d11':d11,'d12':d12,'d13':d13,'d14':d14,'d15':d15,'d16':d16,
                                          'd17':d17})
        
        except Exception as e:
            print (e)
            continue
    
    # save to csv
    current_month_feature_df = pd.DataFrame(current_month_feature)
    columns = ['User_id','Coupon_id','Date_received','label','c1','c2','c3','c4','c5','c6',
               'd1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15',
               'd16','d17']
    current_month_feature_df.to_csv('feature_engineering/current_month_feature.csv', index=False, columns=columns)




if __name__ == '__main__':

    data = pd.read_csv('splitdata/train_split_20160601_20160615.csv')
    current_feature(data)
