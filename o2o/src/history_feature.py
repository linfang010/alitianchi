# filename : history_feature.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def offline_user_feature(unique_id,offline_data):
    # user feature list
    user_feature = []
    # offline data groupby user id
    data_group = offline_data.groupby('User_id')
    # extract user features by user_id in trainning data
    for user_id in unique_id:
        try:
            data_piece = data_group.get_group(user_id)
            # u1: coupon count by user
            coupon_piece = data_piece[data_piece['Coupon_id'] != 'null']
            u1 = coupon_piece['User_id'].count()
            # u2: use coupon count by user
            use_coupon_piece = coupon_piece[(coupon_piece['Date'] != 'null') & (coupon_piece['status'] != 1)]
            u2 = use_coupon_piece['User_id'].count()
            # u3: buy count
            buy_piece = data_piece[data_piece['Date'] != 'null']
            u3 = buy_piece['User_id'].count()
            # u4/u5/u6: min/max/mean distance
            u4 = u5 = u6 = np.nan
            distance_piece = use_coupon_piece[use_coupon_piece['Distance'] != 'null']
            if not distance_piece.empty:
                distance_piece['Distance'] = distance_piece['Distance'].apply(lambda x: int(x))
                u4 = distance_piece['Distance'].min()
                u5 = distance_piece['Distance'].max()
                u6 = distance_piece['Distance'].mean()
            # u7/u8/u9: min/max/mean gap between receive and use
            u7 = u8 = u9 = np.nan
            if not use_coupon_piece.empty:
                gap = pd.to_datetime(use_coupon_piece['Date']) - pd.to_datetime(use_coupon_piece['Date_received'])
                u7 = gap.min().days
                u8 = gap.max().days
                gap_mean = gap.mean()
                if gap_mean.seconds > 43200:
                    u9 = gap_mean.days + 1
                else:
                    u9 = gap_mean.days
            # u10: unique merchant count
            u10 = use_coupon_piece['Merchant_id'].drop_duplicates().count()
            # u11: unique coupon count
            u11 = use_coupon_piece['Coupon_id'].drop_duplicates().count()
            # u12: u2/u3
            u12 = np.nan
            if u3 > 0:
                u12 = float(u2)/u3
            # u13: u2/u1
            u13 = np.nan
            if u1 > 0:
                u13 = float(u2)/u1
            # u14: not use coupon count
            not_use_piece = coupon_piece[coupon_piece['Date'] == 'null']
            u14 = not_use_piece['User_id'].count()
            # u15/u16/u17: min/max/mean discount rate
            u15 = u16 = u17 = np.nan
            if not use_coupon_piece.empty:
                u15 = use_coupon_piece['rate'].min()
                u16 = use_coupon_piece['rate'].max()
                u17 = use_coupon_piece['rate'].mean()
            # u18: u10/unique_merchant
            u18 = np.nan
            unique_merchant = coupon_piece['Merchant_id'].drop_duplicates().count()
            if unique_merchant > 0:
                u18 = float(u10)/unique_merchant
            # u19: u11/unique_coupon
            u19 = np.nan
            unique_coupon = coupon_piece['Coupon_id'].drop_duplicates().count()
            if unique_coupon > 0:
                u19 = float(u11)/unique_coupon
            # u20: u2/u10
            u20 = 0.0
            if u10 > 0:
                u20 = float(u2)/u10
            # u21: u2/u11
            u21 = 0.0
            if u11 > 0:
                u21 = float(u2)/u11
            
            

            # Assemble user feature
            user_feature.append({'User_id':user_id,'u1':u1,'u2':u2,'u3':u3,'u4':u4,'u5':u5,'u6':u6,
                                 'u7':u7,'u8':u8,'u9':u9,'u10':u10,'u11':u11,'u12':u12,'u13':u13,'u14':u14,
                                 'u15':u15,'u16':u16,'u17':u17,'u18':u18,'u19':u19,'u20':u20,'u21':u21})

        except Exception as e:
            continue

    # save to csv
    user_feature_df = pd.DataFrame(user_feature)
    columns = ['User_id','u1','u2','u3','u4','u5','u6','u7','u8','u9','u10','u11','u12','u13',
               'u14','u15','u16','u17','u18','u19','u20','u21']
    user_feature_df.to_csv('feature_engineering/user_feature.csv',index=False,columns=columns)



def offline_merchant_feature(unique_id, offline_data):
    # merchant feature list
    merchant_feature = []
    # offline data groupby merchant id
    data_group = offline_data.groupby('Merchant_id')
    # extract merchant features by merchant_id in trainning data
    for merchant_id in unique_id:
        try:
            data_piece = data_group.get_group(merchant_id)
            # m1: coupon count
            coupon_piece = data_piece[data_piece['Coupon_id'] != 'null']
            m1 = coupon_piece['User_id'].count()
            # m2: use coupon count
            use_coupon_piece = coupon_piece[(coupon_piece['Date'] != 'null') & (coupon_piece['status'] != 1)]
            m2 = use_coupon_piece['User_id'].count()
            # m3: buy count
            buy_piece = data_piece[data_piece['Date'] != 'null']
            m3 = buy_piece['User_id'].count()
            # m4/m5/m6: min/max/mean distance
            m4 = m5 = m6 = np.nan
            distance_piece = use_coupon_piece[use_coupon_piece['Distance'] != 'null']
            if not distance_piece.empty:
                distance_piece['Distance'] = distance_piece['Distance'].apply(lambda x: int(x))
                m4 = distance_piece['Distance'].min()
                m5 = distance_piece['Distance'].max()
                m6 = distance_piece['Distance'].mean()
            # m7/m8/m9: min/max/mean gap between receive and use
            m7 = m8 = m9 = np.nan
            if not use_coupon_piece.empty:
                gap = pd.to_datetime(use_coupon_piece['Date']) - pd.to_datetime(use_coupon_piece['Date_received'])
                m7 = gap.min().days
                m8 = gap.max().days
                gap_mean = gap.mean()
                if gap_mean.seconds > 43200:
                    m9 = gap_mean.days + 1
                else:
                    m9 = gap_mean.days
            # m10: unique user count
            m10 = use_coupon_piece['User_id'].drop_duplicates().count()
            # m11: unique coupon count
            m11 = use_coupon_piece['Coupon_id'].drop_duplicates().count()
            # m12: m2/m3
            m12 = np.nan
            if m3 > 0:
                m12 = float(m2)/m3
            # m13: m2/m1
            m13 = np.nan
            if m1 > 0:
                m13 = float(m2)/m1
            # m14: not use coupon count
            not_use_piece = coupon_piece[coupon_piece['Date'] == 'null']
            m14 = not_use_piece['User_id'].count()
            # m15/m16/m17: min/max/mean discount rate
            m15 = m16 = m17 = np.nan
            if not use_coupon_piece.empty:
                m15 = use_coupon_piece['rate'].min()
                m16 = use_coupon_piece['rate'].max()
                m17 = use_coupon_piece['rate'].mean()
            # m18: m10/unique user receive
            m18 = np.nan
            unique_user_receive = coupon_piece['User_id'].drop_duplicates().count()
            if unique_user_receive > 0:
                m18 = float(m10)/unique_user_receive
            # m19: m2/m10
            m19 = 0.0
            if m10 > 0:
                m19 = float(m2)/m10
            # m20: m11/unique coupon
            m20 = np.nan
            unique_coupon = coupon_piece['Coupon_id'].drop_duplicates().count()
            if unique_coupon > 0:
                m20 = float(m11)/unique_coupon
            # m21: m2/m11
            m21 = 0.0
            if m11 > 0:
                m21 = float(m2)/m11
            
           
            # Assemble merchant feature
            merchant_feature.append({'Merchant_id':merchant_id,'m1':m1,'m2':m2,'m3':m3,'m4':m4,'m5':m5,'m6':m6,
                                     'm7':m7,'m8':m8,'m9':m9,'m10':m10,'m11':m11,'m12':m12,'m13':m13,'m14':m14,
                                     'm15':m15,'m16':m16,'m17':m17,'m18':m18,'m19':m19,'m20':m20,'m21':m21})

        except Exception as e:
            continue

    # save to csv
    merchant_feature_df = pd.DataFrame(merchant_feature)
    columns = ['Merchant_id','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11',
               'm12','m13','m14','m15','m16','m17','m18','m19','m20','m21']
    merchant_feature_df.to_csv('feature_engineering/merchant_feature.csv',index=False,columns=columns)
   


def offline_user_merchant_feature(unique_combine, offline_data, user_feature, merchant_feature):
    # user-merchant feature list
    user_merchant_feature = []
    # offline data groupby user_id and merchant_id
    data_group = offline_data.groupby(['User_id','Merchant_id'])
    # extract user-merchant features by user-merchant combination in trainning data
    for index,row in unique_combine.iterrows():
        try:
            data_piece = data_group.get_group((row['User_id'],row['Merchant_id']))
            # um1: coupon count by user in merchant
            coupon_piece = data_piece[data_piece['Coupon_id'] != 'null']
            um1 = coupon_piece['User_id'].count()
            # um2: use coupon count by user in merchant
            use_coupon_piece = coupon_piece[(coupon_piece['Date'] != 'null') & (coupon_piece['status'] != 1)]
            um2 = use_coupon_piece['User_id'].count()
            # um3: buy count
            buy_piece = data_piece[data_piece['Date'] != 'null']
            um3 = buy_piece['User_id'].count()
            # um4: um2/um3
            um4 = np.nan
            if um3 > 0:
                um4 = float(um2)/um3
            # um5: um2/um1
            um5 = np.nan
            if um1 > 0:
                um5 = float(um2)/um1
            # um6: not use coupon count
            not_use_piece = coupon_piece[coupon_piece['Date'] == 'null']
            um6 = not_use_piece['User_id'].count()
            # um7: um6/u14
            # um8: um2/u2
            um7 = np.nan
            um8 = np.nan
            user_feature_row = user_feature[user_feature['User_id'] == row['User_id']]
            if not user_feature_row.empty:
                u14 = user_feature_row['u14'].item()
                u2 = user_feature_row['u2'].item()
                if u14 > 0:
                    um7 = float(um6)/u14
                if u2 > 0:
                    um8 = float(um2)/u2
            # um9: um6/m14
            # um10: um2/m2
            um9 = np.nan
            um10 = np.nan
            merchant_feature_row = merchant_feature[merchant_feature['Merchant_id'] == row['Merchant_id']]
            if not merchant_feature_row.empty:
                m14 = merchant_feature_row['m14'].item()
                m2 = merchant_feature_row['m2'].item()
                if m14 > 0:
                    um9 = float(um6)/m14
                if m2 > 0:
                    um10 = float(um2)/m2
            
            
            # Assemble user-merchant feature
            user_merchant_feature.append({'User_id':row['User_id'],'Merchant_id':row['Merchant_id'],
                                          'um1':um1,'um2':um2,'um3':um3,'um4':um4,'um5':um5,
                                          'um6':um6,'um7':um7,'um8':um8,'um9':um9,'um10':um10})

        except Exception as e:
            continue

    # save to csv
    user_merchant_feature_df = pd.DataFrame(user_merchant_feature)
    columns = ['User_id','Merchant_id','um1','um2','um3','um4','um5','um6','um7','um8','um9','um10']
    user_merchant_feature_df.to_csv('feature_engineering/user_merchant_feature.csv',index=False,columns=columns)
    


def online_user_feature(unique_id,online_data):
    # online user feature list
    online_user_feature = []
    # online data groupby user_id
    data_group = online_data.groupby('User_id')
    # extract online user features by user_id in trainning data
    for user_id in unique_id:
        try:
            data_piece = data_group.get_group(user_id)
            buy_piece = data_piece[data_piece['Action'] == 1]
            received_piece = data_piece[data_piece['Action'] == 2]
            # uo1: online fixed buy count
            fixed_piece = buy_piece[buy_piece['Coupon_id'] == 'fixed']
            uo1 = fixed_piece['User_id'].count()
            # uo2: online normal buy count
            normal_piece = buy_piece[buy_piece['Coupon_id'] == 'null']
            uo2 = normal_piece['User_id'].count()
            # uo3: online coupon buy count
            uo3 = buy_piece['User_id'].count() - uo1 - uo2
            # uo4: online coupon receied count
            uo4 = received_piece['User_id'].count() + uo3
            # uo5: uo3/uo4
            uo5 = np.nan
            if uo4 > 0:
                uo5 = float(uo3)/uo4
            # uo6: uo3/uo2
            # uo7: uo1/uo2
            uo6 = np.nan
            uo7 = np.nan
            if uo2 > 0:
                uo6 = float(uo3)/uo2
                uo7 = float(uo1)/uo2
            

            # Assemble online user feature
            online_user_feature.append({'User_id':user_id,'uo1':uo1,'uo2':uo2,'uo3':uo3,'uo4':uo4,'uo5':uo5,
                                        'uo6':uo6,'uo7':uo7})

            
        except Exception as e:
            continue

    # save to csv
    online_user_feature_df = pd.DataFrame(online_user_feature)
    columns = ['User_id','uo1','uo2','uo3','uo4','uo5','uo6','uo7']
    online_user_feature_df.to_csv('feature_engineering/online_user_feature.csv',index=False,columns=columns)

    
def offline_coupon_feature(unique_id, offline_data):
    
    # coupon feature list
    coupon_feature = []
    # offline data group by Coupon_id
    data_group = offline_data.groupby('Coupon_id')
    # extract offline coupon feature by coupon_id in trainning data
    for coupon_id in unique_id:
        try:
            data_piece = data_group.get_group(str(coupon_id))
            # hc1: coupon count
            hc1 = data_piece['User_id'].count()
            # hc2: use coupon count
            use_piece = data_piece[(data_piece['Date'] != 'null') & (data_piece['status'] != 1)]
            hc2 = use_piece['User_id'].count()
            # hc3: hc2/hc1
            hc3 = np.nan
            if hc1 > 0:
                hc3 = float(hc2)/hc1
            
            # Assemble coupon feature
            coupon_feature.append({'Coupon_id':coupon_id,'hc1':hc1,'hc2':hc2,'hc3':hc3})
        
        except Exception as e:
            continue
    
    # save to csv
    coupon_feature_df = pd.DataFrame(coupon_feature)
    columns = ['Coupon_id','hc1','hc2','hc3']
    coupon_feature_df.to_csv('feature_engineering/coupon_feature.csv',index=False,columns=columns)



def offline_user_coupon_feature(unique_combine, offline_data):
    
    # user coupon feature list
    user_coupon_feature = []
    # offline data group by User_id and Coupon_id
    data_group = offline_data.groupby(['User_id','Coupon_id'])
    # extract offline user-coupon feature by user-coupon combination in trainning data
    for index,row in unique_combine.iterrows():
        try:
            data_piece = data_group.get_group((row['User_id'], str(row['Coupon_id'])))
            # uc1: coupon count
            uc1 = data_piece['User_id'].count()
            # uc2: use coupon count
            use_piece = data_piece[(data_piece['Date'] != 'null') & (data_piece['status'] != 1)]
            uc2 = use_piece['User_id'].count()
            # uc3: uc2/uc1
            uc3 = np.nan
            if uc1 > 0:
                uc3 = float(uc2)/uc1
            
            
            # Assemble coupon feature
            user_coupon_feature.append({'User_id':row['User_id'],'Coupon_id':row['Coupon_id'],
                                        'uc1':uc1,'uc2':uc2,'uc3':uc3})
        
        except Exception as e:
            continue
    
    # save to csv
    user_coupon_feature_df = pd.DataFrame(user_coupon_feature)
    columns = ['User_id','Coupon_id','uc1','uc2','uc3']
    user_coupon_feature_df.to_csv('feature_engineering/user_coupon_feature.csv',index=False,columns=columns)



if __name__ == '__main__':

    data = pd.read_csv('splitdata/train_split_20160601_20160615.csv')
    offline_data = pd.read_csv('splitdata/history/offline_raw_split_20160301_20160531.csv')
    online_data = pd.read_csv('splitdata/history/online_raw_split_20160301_20160531.csv')
    
    # extract offline/online user features
    unique_user_id = data['User_id'].drop_duplicates()
    offline_user_feature(unique_user_id, offline_data)
    online_user_feature(unique_user_id, online_data)
    
    # extract offline merchant features
    unique_merchant_id = data['Merchant_id'].drop_duplicates()
    offline_merchant_feature(unique_merchant_id, offline_data)
    
    # extract offline user-merchant features
    user_feature = pd.read_csv('feature_engineering/user_feature.csv')
    merchant_feature = pd.read_csv('feature_engineering/merchant_feature.csv')
    unique_user_merchant = data[['User_id','Merchant_id']].drop_duplicates()
    offline_user_merchant_feature(unique_user_merchant, offline_data, 
                                  user_feature, merchant_feature)
    
    # extract offline coupon features
    unique_coupon_id = data['Coupon_id'].drop_duplicates()
    offline_coupon_feature(unique_coupon_id, offline_data)
    
    # extract offline user-coupon features
    unique_user_coupon = data[['User_id','Coupon_id']].drop_duplicates()
    offline_user_coupon_feature(unique_user_coupon, offline_data)
    