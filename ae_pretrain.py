from scipy import signal
import tensorflow as tf
import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt

#tf.device('/device:GPU:0')

def ext_spectrogram(epoch, fs=1000, window='hamming', nperseg=2000, noverlap=1975, nfft=3000):
    # epoch.shape = channel number, timepoint, trials
    # extract sepctrogram with time point

    dat = []
    for i in range(epoch.shape[2]):
        tfreq = []
        for j in range(epoch.shape[0]):
            f, t, Sxx = signal.stft(epoch[j,:,i], fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            interval = f[-1]/(len(f)-1)
            req_len = int(40 / interval)
            # use frequency(~121th) and tiem(-41th~0)
            tfreq.append(np.abs(Sxx[:121, -41:]).transpose())
        dat.append(np.asarray(tfreq))

    # shape : (trials, channel number, time, freq), time and freq should be : 41, 121
    return np.array(dat)

def get_batch_num(data, batch_size):
    total_len = data.shape[0]
    return math.ceil(total_len/batch_size)

def get_batch(data, batch_size, idx):
    batch_num = get_batch_num(data, batch_size)
    if idx == batch_num - 1:
        return data[batch_size*idx:]
    else:
        return data[batch_size*idx:batch_size*(idx+1)]

# input: location, ten fold number, inversion of label
# output: train_x, train_y, test_x, test_y
def load_data(location = 'dataset1_parsed.mat', ten=9, inv=False, getAll = False):
    # load eeg data
    data = sio.loadmat(location)
    # get eeg data and extract spectogram
    # reshpae spectogram data as shape (num_trials, features) to use it in FC
    ep = ext_spectrogram(data['ep']).reshape(data['ep'].shape[2], -1)
    # get label data
    # reshpae it to (num_trials, 1) to use it in FC
    lb = data['lb'].T
    if inv:
      lb = np.where(lb == 0, 1, 0)
    # generate random index, for unbiased dataset
    shuffle_idx = np.arange(ep.shape[0])
    # fix the seed for consistency
    np.random.seed(2020)
    np.random.shuffle(shuffle_idx)
    # shuffle ep and lb in the same order
    ep = ep[shuffle_idx]
    lb = lb[shuffle_idx]
    if getAll:
        return ep, lb, ep, lb
    # for ten fold
    bs = ep.shape[0]//10
    return ep[(ten+1)*bs:] if ten == 0 else \
    (ep[:ten*bs] if ten == 9 else np.concatenate((ep[:ten*bs],ep[(ten+1)*bs:]), axis = 0)), \
    lb[(ten+1)*bs:] if ten == 0 else \
    (lb[:ten*bs] if ten == 9 else np.concatenate((lb[:ten*bs],lb[(ten+1)*bs:]), axis = 0)), \
    ep[ten*bs:(ten+1)*bs], \
    lb[ten*bs:(ten+1)*bs]

ep_size, _, ep_test, _ = load_data(getAll=True)

# setting
learning_rate = 1e-5
training_epoch = 1000

# get number of input features
# equal for all dataset because of same number of channel(which is 4)
num_input = ep_size.shape[1]

for data_num in (0,):
  for inv in (False,):
    # reset graph to free unused memory
    tf.reset_default_graph()

    # placeholder for input
    X = tf.placeholder(tf.float32, [None, num_input])

    # build nn
    e_W1 = tf.get_variable("e_W1", shape=[num_input, 4096], initializer=tf.contrib.layers.xavier_initializer())
    e_b1 = tf.get_variable("e_b1", shape=[4096], initializer=tf.contrib.layers.xavier_initializer())
    e_W2 = tf.get_variable("e_W2", shape=[4096, 2048], initializer=tf.contrib.layers.xavier_initializer())
    e_b2 = tf.get_variable("e_b2", shape=[2048], initializer=tf.contrib.layers.xavier_initializer())
    e_W3 = tf.get_variable("e_W3", shape=[2048, 1024], initializer=tf.contrib.layers.xavier_initializer())
    e_b3 = tf.get_variable("e_b3", shape=[1024], initializer=tf.contrib.layers.xavier_initializer())
    e_W4 = tf.get_variable("e_W4", shape=[1024, 512], initializer=tf.contrib.layers.xavier_initializer())
    e_b4 = tf.get_variable("e_b4", shape=[512], initializer=tf.contrib.layers.xavier_initializer())
    #e_W5 = tf.get_variable("e_W5", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
    #e_b5 = tf.get_variable("e_b5", shape=[256], initializer=tf.contrib.layers.xavier_initializer())
    encoder = tf.nn.tanh(tf.matmul(X, e_W1) + e_b1)
    encoder = tf.nn.tanh(tf.matmul(encoder, e_W2) + e_b2)
    encoder = tf.nn.tanh(tf.matmul(encoder, e_W3) + e_b3)
    encoder = tf.nn.tanh(tf.matmul(encoder, e_W4) + e_b4)
    #encoder = tf.nn.tanh(tf.matmul(encoder, e_W5) + e_b5)

    d_b4 = tf.get_variable("d_b4", shape=[1024], initializer=tf.contrib.layers.xavier_initializer())
    d_b3 = tf.get_variable("d_b3", shape=[2048], initializer=tf.contrib.layers.xavier_initializer())
    d_b2 = tf.get_variable("d_b2", shape=[4096], initializer=tf.contrib.layers.xavier_initializer())
    d_b1 = tf.get_variable("d_b1", shape=[num_input], initializer=tf.contrib.layers.xavier_initializer())
    #share version
    decoder = tf.nn.tanh(tf.matmul(encoder, tf.transpose(e_W4)) + d_b4)
    decoder = tf.nn.tanh(tf.matmul(decoder, tf.transpose(e_W3)) + d_b3)
    decoder = tf.nn.tanh(tf.matmul(decoder, tf.transpose(e_W2)) + d_b2)
    decoder = tf.matmul(decoder, tf.transpose(e_W1)) + d_b1

    cost = tf.reduce_mean(tf.square(X - decoder))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
    print(f'------------- [AE] dataset{data_num} & inv {inv} -------------')
    saver = tf.train.Saver(tf.global_variables())
    sess.run(tf.global_variables_initializer())
    ep_tot, lb_tot, _, _ = load_data(location = f'dataset{data_num}_parsed.mat', inv=inv, getAll=True)
    ep = ep_tot
    lb = lb_tot
    # ep, lb = ep_tot[train_ind,:], lb_tot[train_ind]
    # test_x,  test_y  = ep_tot[test_ind, :], lb_tot[test_ind]
    batch_num = get_batch_num(ep, batch_size)
    for epoch in range(training_epoch):
        total_cost = 0
        for i in range(batch_num):
            batch_ep = get_batch(ep, batch_size, i)
            _, batch_cost = sess.run([optimizer, cost], feed_dict={X: batch_ep})
            total_cost += batch_cost
        print(f'[AE] Epoch: {epoch+1} & Avg_cost = {total_cost/batch_num}')
        if epoch+1 in (100, 500, 1000):
          saver.save(sess, f'./model_{dataset}_{inv}_ae_{epoch+1}/dnn.ckpt')
    print(f'Final Train Reconstruction Cost: {sess.run(cost, feed_dict={X: ep})}')

print('Done!')