# filename : train_data.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
    

def train_data(data, user_feature, merchant_feature, user_merchant_feature,
               online_user_feature, coupon_feature, user_coupon_feature):
    
    # feature list
    all_feature = []
    # extract features by iterating the monthly data
    for index,row in data.iterrows():
        try:
            # init features
            u1=u2=u3=u4=u5=u6=u7=u8=u9=u10=u11=u12=u13=u14=u15=u16=u17=u18=u19=u20=u21=\
            m1=m2=m3=m4=m5=m6=m7=m8=m9=m10=m11=m12=m13=m14=m15=m16=m17=m18=m19=m20=m21=\
            um1=um2=um3=um4=um5=um6=um7=um8=um9=um10=\
            uo1=uo2=uo3=uo4=uo5=uo6=uo7=\
            hc1=hc2=hc3=uc1=uc2=uc3= np.nan
            
            
            user_feature_row = user_feature[user_feature['User_id'] == row['User_id']]
            merchant_feature_row = merchant_feature[merchant_feature['Merchant_id'] == row['Merchant_id']]
            user_merchant_feature_row = user_merchant_feature[(user_merchant_feature['User_id'] == row['User_id']) & (user_merchant_feature['Merchant_id'] == row['Merchant_id'])]
            online_user_feature_row = online_user_feature[online_user_feature['User_id'] == row['User_id']]
            coupon_feature_row = coupon_feature[coupon_feature['Coupon_id'] == row['Coupon_id']]
            user_coupon_feature_row = user_coupon_feature[(user_coupon_feature['User_id'] == row['User_id']) & (user_coupon_feature['Coupon_id'] == row['Coupon_id'])]

            
            if not user_feature_row.empty:
                u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21 = \
                user_feature_row['u1'].item(),user_feature_row['u2'].item(),user_feature_row['u3'].item(),\
                user_feature_row['u4'].item(),user_feature_row['u5'].item(),user_feature_row['u6'].item(),\
                user_feature_row['u7'].item(),user_feature_row['u8'].item(),user_feature_row['u9'].item(),\
                user_feature_row['u10'].item(),user_feature_row['u11'].item(),user_feature_row['u12'].item(),\
                user_feature_row['u13'].item(),user_feature_row['u14'].item(),user_feature_row['u15'].item(),\
                user_feature_row['u16'].item(),user_feature_row['u17'].item(),user_feature_row['u18'].item(),\
                user_feature_row['u19'].item(),user_feature_row['u20'].item(),user_feature_row['u21'].item()
            
            if not merchant_feature_row.empty:
                m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,m21 = \
                merchant_feature_row['m1'].item(),merchant_feature_row['m2'].item(),merchant_feature_row['m3'].item(),\
                merchant_feature_row['m4'].item(),merchant_feature_row['m5'].item(),merchant_feature_row['m6'].item(),\
                merchant_feature_row['m7'].item(),merchant_feature_row['m8'].item(),merchant_feature_row['m9'].item(),\
                merchant_feature_row['m10'].item(),merchant_feature_row['m11'].item(),merchant_feature_row['m12'].item(),\
                merchant_feature_row['m13'].item(),merchant_feature_row['m14'].item(),merchant_feature_row['m15'].item(),\
                merchant_feature_row['m16'].item(),merchant_feature_row['m17'].item(),merchant_feature_row['m18'].item(),\
                merchant_feature_row['m19'].item(),merchant_feature_row['m20'].item(),merchant_feature_row['m21'].item()
            
            if not user_merchant_feature_row.empty:
                um1,um2,um3,um4,um5,um6,um7,um8,um9,um10 = \
                user_merchant_feature_row['um1'].item(),user_merchant_feature_row['um2'].item(),user_merchant_feature_row['um3'].item(),\
                user_merchant_feature_row['um4'].item(),user_merchant_feature_row['um5'].item(),user_merchant_feature_row['um6'].item(),\
                user_merchant_feature_row['um7'].item(),user_merchant_feature_row['um8'].item(),user_merchant_feature_row['um9'].item(),\
                user_merchant_feature_row['um10'].item()
            
            if not online_user_feature_row.empty:
                uo1,uo2,uo3,uo4,uo5,uo6,uo7 = \
                online_user_feature_row['uo1'].item(),online_user_feature_row['uo2'].item(),online_user_feature_row['uo3'].item(),\
                online_user_feature_row['uo4'].item(),online_user_feature_row['uo5'].item(),online_user_feature_row['uo6'].item(),\
                online_user_feature_row['uo7'].item()
            
            if not coupon_feature_row.empty:
                hc1,hc2,hc3 = \
                coupon_feature_row['hc1'].item(),coupon_feature_row['hc2'].item(),coupon_feature_row['hc3'].item()
            
            if not user_coupon_feature_row.empty:
                uc1,uc2,uc3 = \
                user_coupon_feature_row['uc1'].item(),user_coupon_feature_row['uc2'].item(),user_coupon_feature_row['uc3'].item()


            # Assemble all features
            all_feature.append({'u1':u1,'u2':u2,'u3':u3,'u4':u4,'u5':u5,'u6':u6,'u7':u7,'u8':u8,'u9':u9,'u10':u10,'u11':u11,
                                'u12':u12,'u13':u13,'u14':u14,'u15':u15,'u16':u16,'u17':u17,'u18':u18,'u19':u19,'u20':u20,'u21':u21,
                                'm1':m1,'m2':m2,'m3':m3,'m4':m4,'m5':m5,'m6':m6,'m7':m7,'m8':m8,'m9':m9,'m10':m10,'m11':m11,
                                'm12':m12,'m13':m13,'m14':m14,'m15':m15,'m16':m16,'m17':m17,'m18':m18,'m19':m19,'m20':m20,'m21':m21,
                                'um1':um1,'um2':um2,'um3':um3,'um4':um4,'um5':um5,'um6':um6,'um7':um7,'um8':um8,'um9':um9,'um10':um10,
                                'uo1':uo1,'uo2':uo2,'uo3':uo3,'uo4':uo4,'uo5':uo5,'uo6':uo6,'uo7':uo7,
                                'hc1':hc1,'hc2':hc2,'hc3':hc3,'uc1':uc1,'uc2':uc2,'uc3':uc3})
            
            
            
        except Exception as e:
            print (e)
            continue

    # save to csv
    all_feature_pd = pd.DataFrame(all_feature)
    
    columns = ['u1','u2','u3','u4','u5','u6','u7','u8','u9','u10','u11','u12','u13',
               'u14','u15','u16','u17','u18','u19','u20','u21',
               'm1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12','m13',
               'm14','m15','m16','m17','m18','m19','m20','m21',
               'um1','um2','um3','um4','um5','um6','um7','um8','um9','um10',
               'uo1','uo2','uo3','uo4','uo5','uo6','uo7',
               'hc1','hc2','hc3','uc1','uc2','uc3']
    
    all_feature_pd.to_csv('train/history/history_data.csv', index=False, columns=columns)




if __name__ == '__main__':

    data = pd.read_csv('splitdata/train_split_20160601_20160615.csv')
    user_feature = pd.read_csv('feature_engineering/train_6/user_feature.csv')
    merchant_feature = pd.read_csv('feature_engineering/train_6/merchant_feature.csv')
    user_merchant_feature = pd.read_csv('feature_engineering/train_6/user_merchant_feature.csv')
    online_user_feature = pd.read_csv('feature_engineering/train_6/online_user_feature.csv')
    coupon_feature = pd.read_csv('feature_engineering/train_6/coupon_feature.csv')
    user_coupon_feature = pd.read_csv('feature_engineering/train_6/user_coupon_feature.csv')
    
    
    # assemble trainning data
    train_data(data, user_feature, merchant_feature, user_merchant_feature,
               online_user_feature, coupon_feature, user_coupon_feature) 
    