# filename: train_predict.py
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datasets import generate_data
from datetime import timedelta
from build_seq2seq import build_graph


seq_length = 30
batch_size = 14

input_dim = 26
output_dim = 2

nb_iters = 5000
log_dir = 'D:/Tensorflow/logs/train'



def train_predict(is_predict=False):
    
    rnn_model = build_graph(input_dim, output_dim, seq_length, feed_previous=is_predict)
    temp_saver = rnn_model['saver']()
    
    # ## Training of the neural net
    def train_batch(step):
        """
        Training step that optimizes the weights
        provided some batch_size X and Y examples from the dataset.
        """
        feed_dict = {rnn_model['enc_inp'][t]: batch_x[t] for t in range(seq_length)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_y[t] for t in range(seq_length)})
        feed_dict.update({rnn_model['keep_prob']:0.5})
        
        if step % 100 == 0:
            summary, _, loss_t = sess.run([merged, rnn_model['train_op'], rnn_model['loss']], feed_dict)
            train_writer.add_summary(summary, step)
            print ('step:%d, loss:%f' % (step, loss_t))
        else:
            sess.run([rnn_model['train_op']], feed_dict)
           
        if step == nb_iters:
            checkpoint_file = log_dir + '/model.ckpt'
            temp_saver.save(sess, checkpoint_file, global_step=step)
    
    
    with tf.Session() as sess:
        
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        # trainning
        if not is_predict:
            for t in range(nb_iters + 1):
                train_batch(t)
        # predicting
        else:
            
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                temp_saver.restore(sess, ckpt.model_checkpoint_path)

                test_sample = scale_train[-30:]
                batch_test = test_sample.reshape(30, 1, input_dim)
                feed_dict = {rnn_model['enc_inp'][t]: batch_test[t] for t in range(seq_length)}
                feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, output_dim], dtype=np.float32) for t in range(seq_length)})
                feed_dict.update({rnn_model['keep_prob']:1.0})
                outputs = np.array(sess.run([rnn_model['reshaped_outputs']], feed_dict)[0])
                outputs = outputs.reshape((seq_length, output_dim))
                rescale_outputs = result_scaler.inverse_transform(outputs)
                result_list = []
                date = pd.to_datetime('20140901')
                for i in range(seq_length):
                    result_list.append({'date':date.strftime('%Y%m%d'),
                                        'total_purchase':rescale_outputs[i][0],
                                        'total_redeem':rescale_outputs[i][1]})
                    date = date + timedelta(days=1)
                
                result = pd.DataFrame(result_list)
                columns = ['date','total_purchase','total_redeem']
                result['total_purchase'] = result['total_purchase'].astype(np.int64)
                result['total_redeem'] = result['total_redeem'].astype(np.int64)
                result.to_csv('result/tc_comp_predict_table.csv', index=False, columns=columns)
                
                data = pd.read_csv('result/test.csv')
                plotdata = pd.concat([data, result])
                plotdata['date'] = plotdata['date'].apply(lambda x:pd.to_datetime(str(x)))
                plotdata.plot(x=['date'], y=['total_purchase','total_redeem'])
            else:
                print ('no checkpoint.')
        
        train_writer.close()




if __name__ == '__main__':
    
    scaler = MinMaxScaler()
    result_scaler = MinMaxScaler()
    raw_data = pd.read_csv('train/train_data.csv')
    train_data = raw_data.iloc[:, 1:input_dim+1]
    test_data = train_data.iloc[:, 0:output_dim]
    scale_train = scaler.fit_transform(train_data)
    scale_test = result_scaler.fit_transform(test_data)
    scale_train = scale_train[-420:]
    scale_test = scale_test[-420:]
    batch_x, batch_y = generate_data(scale_train, scale_test, batch_size, input_dim, output_dim)
    
    train_predict(True)

