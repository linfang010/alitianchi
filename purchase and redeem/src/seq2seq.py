# filename: seq2seq.py
# -*- coding: utf-8 -*-

import tensorflow as tf  # Version 1.0 or 0.12
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datasets import generate_data
from datetime import timedelta


# ## Neural network's hyperparameters


# Internal neural network parameters
# Time series will have the same past and future (to be predicted) lenght.
seq_length = 30
batch_size = 14  # Low value used for live demo purposes - 100 and 1000 would be possible too, crank that up!

# Output dimension (e.g.: multiple signals at once, tied in time)
input_dim = 26
output_dim = 2
hidden_dim = 64  # Count of hidden neurons in the recurrent units.
# Number of stacked recurrent cells, on the neural depth axis.
layers_stacked_count = 2

# Optmizer:
learning_rate = 0.007  # Small lr helps not to diverge during training.
# How many times we perform a training step (therefore how many times we
# show a batch).
nb_iters = 5000
lr_decay = 0.92  # default: 0.9 . Simulated annealing.
momentum = 0.5  # default: 0.0 . Momentum technique in weights update
lambda_l2_reg = 0.003  # L2 regularization of weights - avoids overfitting


# ## Definition of the seq2seq neuronal architecture
#
# <img src="https://www.tensorflow.org/images/basic_seq2seq.png" />
#
# Comparatively to what we see in the image, our neural network deals with
# signal rather than letters. Also, we don't have the feedback mechanism
# yet, which is to be implemented in the exercise 4. We do have the "GO"
# token however.

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

# Backward compatibility for TensorFlow's version 0.12:
try:
    tf.nn.seq2seq = tf.contrib.legacy_seq2seq
    tf.nn.rnn_cell = tf.contrib.rnn
    tf.nn.rnn_cell.GRUCell = tf.contrib.rnn.GRUCell
    print("TensorFlow's version : 1.0 (or more)")
except:
    print("TensorFlow's version : 0.12")

# train or predict
is_train = False

tf.reset_default_graph()
# sess.close()
sess = tf.InteractiveSession()

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


with tf.name_scope('Seq2seq'):

    # Encoder: inputs
    enc_inp = [
        tf.placeholder(tf.float32, shape=(
            None, input_dim), name="inp_{}".format(t))
        for t in range(seq_length)
    ]
    # dropout prob
    input_keep_prob = tf.placeholder(tf.float32)
    output_keep_prob = tf.placeholder(tf.float32)
    
    # Decoder: expected outputs
    expected_sparse_output = [
        tf.placeholder(tf.float32, shape=(None, output_dim),
                       name="expected_sparse_output_".format(t))
        for t in range(seq_length)
    ]

    # Give a "GO" token to the decoder.
    # You might want to revise what is the appended value "+ enc_inp[:-1]".
    dec_inp = [tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")] + enc_inp[:-1]

    # Create a `layers_stacked_count` of stacked RNNs (GRU cells here).
    cells = []
    for i in range(layers_stacked_count):
        with tf.variable_scope('RNN_{}'.format(i)):
            gru_cell = tf.nn.rnn_cell.GRUCell(hidden_dim)
            cells.append(tf.nn.rnn_cell.DropoutWrapper(gru_cell, input_keep_prob=input_keep_prob,
                                                       output_keep_prob=output_keep_prob))
            # cells.append(tf.nn.rnn_cell.BasicLSTMCell(...))
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
     

    # Here, the encoder and the decoder uses the same cell, HOWEVER,
    # the weights aren't shared among the encoder and decoder, we have two
    # sets of weights created under the hood according to that function's def.
    dec_outputs, dec_memory = tf.nn.seq2seq.basic_rnn_seq2seq(
        enc_inp,
        dec_inp,
        cell
    )
    
    # For reshaping the input and output dimensions of the seq2seq RNN:
    with tf.name_scope('w_out'):
        w_out = tf.Variable(tf.random_normal([hidden_dim, output_dim]))
        variable_summaries(w_out)
    with tf.name_scope('b_out'):
        b_out = tf.Variable(tf.random_normal([output_dim]))
        variable_summaries(b_out)
    
    output_scale_factor = tf.Variable(1.0, name="Output_ScaleFactor")
    # Final outputs: with linear rescaling similar to batch norm,
    # but without the "norm" part of batch normalization hehe.
    reshaped_outputs = [output_scale_factor *
                        (tf.matmul(i, w_out) + b_out) for i in dec_outputs]



# Training loss and optimizer

with tf.name_scope('Loss'):
    # L2 loss
    output_loss = 0
    for _y, _Y in zip(reshaped_outputs, expected_sparse_output):
        output_loss += tf.reduce_mean(tf.nn.l2_loss(_y - _Y))

    # L2 regularization (to avoid overfitting and to have a  better
    # generalization capacity)
    reg_loss = 0
    for tf_var in tf.trainable_variables():
        if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
            reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

    loss = output_loss + lambda_l2_reg * reg_loss
    tf.summary.scalar('loss', loss)

with tf.name_scope('Optimizer'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=lr_decay, momentum=momentum)
    train_op = optimizer.minimize(loss)

# Save model
saver = tf.train.Saver()

# ## Training of the neural net
def train_batch(step):
    """
    Training step that optimizes the weights
    provided some batch_size X and Y examples from the dataset.
    """
    feed_dict = {enc_inp[t]: batch_x[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: batch_y[
                     t] for t in range(len(expected_sparse_output))})
    feed_dict.update({input_keep_prob:0.9, output_keep_prob:0.5})
    
    if step % 100 == 0:  
        summary, _, loss_t = sess.run([merged, train_op, loss], feed_dict)
        train_writer.add_summary(summary, step)
        print ('step:%d, loss:%f' % (step, loss_t))
    else:
       sess.run([train_op, loss], feed_dict)
      
    if step == nb_iters:
        checkpoint_file = 'E:/Tensorflow/logs/train/model.ckpt'
        saver.save(sess, checkpoint_file, global_step=step)
    


def test_batch():
    """
    Test step, does NOT optimizes. Weights are frozen by not
    doing sess.run on the train_op.
    """
    feed_dict = {enc_inp[t]: batch_x[t] for t in range(len(enc_inp))}
    feed_dict.update({expected_sparse_output[t]: batch_y[
                     t] for t in range(len(expected_sparse_output))})
    loss_t = sess.run([loss], feed_dict)
    return loss_t[0]


if is_train:
    # Training
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('E:/Tensorflow/logs/train', sess.graph)
    #test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    sess.run(tf.global_variables_initializer())
    for t in range(nb_iters + 1):
        train_batch(t)
        
        '''
        if t % 10 == 0:
            # Tester
            test_loss = test_batch(1)
            test_losses.append(test_loss)
            print("Step {}/{}, train loss: {}, \tTEST loss: {}".format(t,
                                                                       nb_iters, train_loss, test_loss))
        '''
    train_writer.close()

else:
    # predict
    ckpt = tf.train.get_checkpoint_state('E:/Tensorflow/logs/train')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    
        test_sample = scale_train[-30:]
        batch_test = test_sample.reshape(30, 1, input_dim)
        feed_dict = {enc_inp[t]: batch_test[t] for t in range(seq_length)}
        feed_dict.update({input_keep_prob:1.0, output_keep_prob:1.0})
        outputs = np.array(sess.run([reshaped_outputs], feed_dict)[0])
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
   
sess.close()

