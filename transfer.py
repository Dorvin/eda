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

ep_size, _, ep_test, _ = load_data()

# setting
c_learning_rate = 1e-5
c_training_epoch = 1000

# get number of input features
# equal for all dataset because of same number of channel(which is 4)
num_input = ep_size.shape[1]

for data_num in (0,):
  for inv in (False,):
    for until in (0, 1, 5):
        for weight in ("./model",):
            # reset graph to free unused memory
            tf.reset_default_graph()

            # placeholder for input and label
            X = tf.placeholder(tf.float32, [None, num_input])
            Y = tf.placeholder(tf.float32, [None, 1])

            # build nn of leftside(fixed)
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

            #c_W = tf.get_variable("c_W", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
            #c_b = tf.get_variable("c_b", shape=[256], initializer=tf.contrib.layers.xavier_initializer())
            c_W2 = tf.get_variable("c_W2", shape=[512, 1], initializer=tf.contrib.layers.xavier_initializer())
            c_b2 = tf.get_variable("c_b2", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
            #c_logit = tf.nn.tanh(tf.matmul(bottleneck,c_W) + c_b)
            c_logit = tf.matmul(encoder, c_W2) + c_b2
            c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=c_logit))
            # for tensorboard
            tf.summary.scalar('loss', c_loss)

            # optimizer
            c_optimizer = tf.train.AdamOptimizer(c_learning_rate).minimize(c_loss)
            # prediction and accuracy
            predicted = tf.cast(tf.nn.sigmoid(c_logit) > 0.5, dtype=tf.float32)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
            # for tensorboard
            tf.summary.scalar('accuracy', accuracy)
            merged = tf.summary.merge_all()

            with tf.Session() as sess:
                print(f'----------dataset{data_num} & inv {inv} & weight {weight} & until {until}----------')
                # use different optimizer according to datasets, and also training epoch(optional)
                #if data_num == 0:
                #c_training_epoch = 1500
                #c_optimizer = c_optimizer0
                #elif data_num == 1:
                #c_training_epoch = 1200
                #c_optimizer = c_optimizer1
                #elif data_num == 3:
                #c_training_epoch = 700
                #c_optimizer = c_optimizer3
                #elif data_num == 6:
                #c_training_epoch = 700
                #c_optimizer = c_optimizer6

                # for Avg of ten fold result
                total_test = 0
                # originally range(0,10)
                for ten_num in range(0, 10):
                    # first, initialize all networks
                    sess.run(tf.global_variables_initializer())
                    saver = None
                    # make saver to restore leftside nn, only not for until 0
                    if until == 1:
                        saver = tf.train.Saver(var_list = [e_W1, e_b1])
                    elif until == 2:
                        saver = tf.train.Saver(var_list = [e_W1, e_b1, e_W2, e_b2])
                    elif until == 3:
                        saver = tf.train.Saver(var_list = [e_W1, e_b1, e_W2, e_b2, e_W3, e_b3])
                    elif until == 4:
                        saver = tf.train.Saver(var_list = [e_W1, e_b1, e_W2, e_b2, e_W3, e_b3, e_W4, e_b4])
                    # restore only not for until0
                    if until != 0:
                        ckpt = tf.train.get_checkpoint_state(weight)
                        saver.restore(sess, ckpt.model_checkpoint_path)
                    # load data according to dataset number, ten number, inv
                    ep, lb, test_x, test_y = load_data(location = f'dataset{data_num}_parsed.mat', ten=ten_num, inv=inv)
                    # fix batch_size to 32 to avoid oom
                    batch_size = 32
                    batch_num = get_batch_num(ep, batch_size)
                    # for convenience
                    feed_dict_train = {X: ep, Y: lb}
                    feed_dict_val = {X: test_x, Y: test_y}

                    # make writer for tensorboard, for each datasets, inv, until, ten_num
                    train_summary_path = f"./logs/{data_num}_{inv}_{weight}_d{until}/{ten_num}/train"
                    val_summary_path = f"./logs/{data_num}_{inv}_{weight}_d{until}/{ten_num}/val"
                    train_writer = tf.summary.FileWriter(train_summary_path)
                    val_writer = tf.summary.FileWriter(val_summary_path)
                    for epoch in range(c_training_epoch):
                        total_cost = 0
                        for i in range(batch_num):
                            batch_ep = get_batch(ep, batch_size, i)
                            batch_lb = get_batch(lb, batch_size, i)
                            _, batch_cost = sess.run([c_optimizer, c_loss], feed_dict={X: batch_ep, Y: batch_lb})
                            total_cost += batch_cost
                        # for manual logging
                        #print(f'[Classifier] Epoch: {epoch+1} & Avg_cost = {total_cost/batch_num}')

                        # write log for tensorboard
                        train_summary = sess.run(merged, feed_dict=feed_dict_train)
                        val_summary = sess.run(merged, feed_dict=feed_dict_val)
                        train_writer.add_summary(train_summary, global_step=epoch+1)
                        val_writer.add_summary(val_summary, global_step=epoch+1)
                        train_writer.flush()
                        val_writer.flush()
                        
                    # get final accuracy of tenfold
                    print(f"ten fold: {ten_num}")
                    print(f'Test Accuracy: {sess.run(accuracy * 100, feed_dict={X: test_x, Y: test_y})}')
                    print(f'Train Accuracy: {sess.run(accuracy * 100, feed_dict={X: ep, Y: lb})}')
                    total_test += sess.run(accuracy * 100, feed_dict={X: test_x, Y: test_y})
                # get final Avg accuracy
                print(f'Final Avg Test Accuracy: {total_test/10}')
print('Done!')