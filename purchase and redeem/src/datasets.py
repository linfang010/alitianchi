# filename: datasets.py
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def generate_data(scale_train, scale_test, batch_size, input_dim, output_dim):

    sample_x = scale_train.reshape((batch_size, 30, input_dim))
    sample_y = scale_test.reshape((batch_size, 30, output_dim))
    # shape: (batch_size, seq_length, dim)
    batch_x = np.array(sample_x[0:batch_size-1]).transpose((1, 0, 2))
    batch_y = np.array(sample_y[1:batch_size]).transpose((1, 0, 2))
    # shape: (seq_length, batch_size, dim)
    return (batch_x, batch_y)
        
    
if __name__ == '__main__':
    
    scaler = MinMaxScaler()
    raw_data = pd.read_csv('train/train_data.csv')
    train_data = raw_data.iloc[:, 1:27]
    test_data = train_data.iloc[:, 0:2]
    scale_train = scaler.fit_transform(train_data)
    scale_test = scaler.fit_transform(test_data)
    scale_train = scale_train[-420:]
    scale_test = scale_test[-420:]
    batch_x,batch_y = generate_data(scale_train, scale_test, 14, 26, 2)
