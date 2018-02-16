# filename: preprocess.py

import csv,datetime
import pandas as pd
            
    
def split_raw(start_date_str,end_date_str,b_online=False):

    openfilename = 'rawdata/ccf_offline_stage1_train.csv'
    closefilename = 'splitdata/history/offline_raw_split_'+start_date_str+'_'+end_date_str+'.csv'
    if b_online:
        openfilename = 'rawdata/online_train_clean.csv'
        closefilename = 'splitdata/history/online_raw_split_'+start_date_str+'_'+end_date_str+'.csv'
        
    date_reader = []
    start_date = datetime.datetime.strptime(start_date_str,'%Y%m%d')
    end_date = datetime.datetime.strptime(end_date_str,'%Y%m%d')
    
    with open(openfilename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'User_id':
                date_reader.append(row)
                continue

            # split offline data by date_received and date
            if row[5] != 'null':
                Date_received = datetime.datetime.strptime(row[5],'%Y%m%d')
                if Date_received >= start_date and Date_received <= end_date:
                    if row[6] != 'null':
                        Date = datetime.datetime.strptime(row[6],'%Y%m%d')
                        if Date >= start_date and Date <= end_date:
                            date_reader.append(row)
                    else:
                        date_reader.append(row)
                   

            elif row[6] != 'null':
                Date = datetime.datetime.strptime(row[6],'%Y%m%d')
                if Date >= start_date and Date <= end_date:
                    date_reader.append(row)
    
    with open(closefilename, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(date_reader)


    

        
def split_train(start_date_str,end_date_str):
    
    date_reader = []
    start_date = datetime.datetime.strptime(start_date_str,'%Y%m%d')
    end_date = datetime.datetime.strptime(end_date_str,'%Y%m%d')

    with open('rawdata/ccf_offline_stage1_train.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'User_id':
                date_reader.append(row)
                continue

            # split test data by date_received
            if row[5] != 'null':
                Date_received = datetime.datetime.strptime(row[5],'%Y%m%d')
                if Date_received >= start_date and Date_received <= end_date:
                    date_reader.append(row)

    filename = 'splitdata/train_split_'+start_date_str+'_'+end_date_str+'.csv'
    with open(filename, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(date_reader)



def convert_null():
    
    date_reader = []
    
    with open('splitdata/history/offline_raw_split_20160401_20160630.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'User_id':
                row.append('status')
                row.append('rate')
                date_reader.append(row)
                continue
            
            # convert Date - Date_recevied > 15 to null
            row.append(0)
            if row[5] != 'null' and row[6] != 'null':
                Date_received = datetime.datetime.strptime(row[5],'%Y%m%d')
                Date = datetime.datetime.strptime(row[6],'%Y%m%d')
                if (Date - Date_received).days > 15:
                    row[7] = 1
            
            # convert rate
            row.append(0)
            if row[3] != 'null':
                temp = row[3].split(':')
                if len(temp) == 2:
                    row[8] = (float(temp[0]) - float(temp[1]))/float(temp[0])
                else:
                    row[8] = float(temp[0])
            
            date_reader.append(row)
    
    with open('splitdata/history/train.csv', 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerows(date_reader)
            
                    
            
                


if __name__=='__main__':

    split_raw('20160401', '20160630', True)
    #split_train('20160401','20160430')
    #convert_null()

    #data = pd.read_csv('splitdata/train_split_20160401_20160430.csv')
    #data.sort_values(by=['Date_received'], inplace=True)
    #data.to_csv('splitdata/train_split_20160401_20160430.csv', index=False)
                

    
    
