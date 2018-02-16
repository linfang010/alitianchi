# filename : insert_mysql.py
# -*- coding: utf-8 -*-

import mysql,csv
from mysql import connector


CONFIG ={'host':'192.168.6.227',
        'user':'root',
        'password':'123456',
        'database':'o2o'
         }



def insert_mysql(filename,tablename):

    # connect to mysql
    cnx = mysql.connector.connect(**CONFIG)
    cursor = cnx.cursor()
    sql = "insert into " + tablename + " values(%s,%s,%s,%s,%s,%s,%s)"
    # batch insert
    values_to_insert = []
    
    # read from csv
    with open(filename,'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'User_id':
                continue
            values_to_insert.append(row)
            # insert
            if len(values_to_insert) >= 100000:
                cursor.executemany(sql,values_to_insert)
                # commit        
                cnx.commit()
                values_to_insert = []

    # insert last lines
    if values_to_insert != []:
        cursor.executemany(sql,values_to_insert)
        cnx.commit()

    # close
    cursor.close()
    cnx.close()
    








if __name__ == '__main__':

    insert_mysql('rawdata/ccf_online_stage1_train.csv','ccf_online_stage1_train')

    
