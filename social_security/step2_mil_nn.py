import numpy as np
import cPickle
import random
import argparse
import os
import tensorflow as tf

from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn import metrics

random.seed(12345)


def split_data(dataset_x, dataset_y, split_ratio):
    """
    split data to training set and validation set
    :param split_ratio:
    :return:
    """
    num_examples = len(dataset_x)
    training_x = dataset_x[:int(num_examples*split_ratio)]
    training_y = dataset_y[:int(num_examples*split_ratio)]

    validation_x = dataset_x[int(num_examples*split_ratio): num_examples]
    validation_y = dataset_y[int(num_examples*split_ratio): num_examples]

    training_y = np.asarray(training_y, dtype='float32')
    validation_y = np.asarray(validation_y, dtype='float32')
    return training_x, training_y, validation_x, validation_y


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def _make_hparam_string(learning_rate, num_fc):
    fc_param = "fc=%d" % num_fc
    return "lr_%.0E,%s" % (learning_rate, fc_param)


def train_model(training_x, training_y, validation_x, validation_y,
                num_feature,
                num_fc,
                learning_rate,
                log_dir,
                num_epoch,
                classes=2):

    tf.reset_default_graph()
    sess = tf.Session()

    print '...... building model .......'
    x = tf.placeholder(tf.float32, [None, num_feature])
    y = tf.placeholder(tf.float32, [None, classes])

    if num_fc == 3:
        fc1 = fc_layer(x, num_feature, 100, "fc1")
        fc2 = fc_layer(fc1, 100, 20, "fc2")
        logits = fc_layer(fc2, 20, classes, "fc3")
    elif num_fc == 2:
        fc1 = fc_layer(x, num_feature, 100, "fc1")
        logits = fc_layer(fc1, 100, classes, "fc2")
    else:# num_fc == 1
        logits = fc_layer(x, num_feature, classes, "fc1")

    with tf.name_scope("softmax"):
        y_pred = tf.nn.softmax(tf.reduce_sum(logits, axis=0))
        tf.summary.histogram("y_pred", y_pred)

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.square(y - y_pred))
        tf.summary.scalar("cost", cost)

    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.name_scope('accuracy'):
        # compute the MIL accuracy
        correct_prediction = tf.equal(tf.argmax(tf.expand_dims(y_pred, 0), 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    print '...... initializing ......'
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(log_dir + _make_hparam_string(learning_rate, num_fc))
    writer.add_graph(sess.graph)

    for epoch in tqdm(range(num_epoch)):

        train_acc = 0.
        train_cost = 0.
        validation_acc = 0.
        validation_cost = 0.

        # training
        for mini_batch in range(len(training_x)):
            batch_x, batch_y = training_x[mini_batch], training_y[mini_batch]
            batch_y = batch_y[np.newaxis, :]
            # print batch_x.shape
            # print batch_y.shape
            _, _accuracy, _cost = sess.run([train_op, accuracy, cost], feed_dict={x: batch_x,
                                                                                  y: batch_y})

            train_acc += _accuracy
            train_cost += _cost

            # write summary
            if mini_batch == 100:
                [s] = sess.run([summ],
                               feed_dict={x: batch_x,
                                          y: batch_y})
                writer.add_summary(s, epoch)

        # validation
        for mini_batch in range(len(validation_x)):
            batch_x, batch_y = validation_x[mini_batch], validation_y[mini_batch]
            batch_y = batch_y[np.newaxis, :]
            _accuracy, _cost = sess.run([accuracy, cost], feed_dict={x: batch_x,
                                                                     y: batch_y})
            validation_acc += _accuracy
            validation_cost += _cost

        train_acc /= len(training_x)
        train_cost /= len(training_x)
        validation_acc /= len(validation_x)
        validation_cost /= len(validation_x)

        print 'epoch %d, train_acc = %f, train_cost = %f, validation_acc = %f, validation_cost = %f' % \
              (epoch, train_acc, train_cost, validation_acc, validation_cost)

if __name__ == '__main__':

    from data_reader import read_training_set
    # parse args
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", help="the root of csv files",
                        type=str, default=os.path.join(os.getcwd(), 'data'))
    parser.add_argument("--split_ratio", help="split data to training and validation data",
                        type=str, default=0.8)
    parser.add_argument("--num_feature", help="number of features",
                        type=int, default=63)
    parser.add_argument("--log_dir", help="path to store logs",
                        type=str, default=os.path.join(os.getcwd(), 'tb_log')+'/')
    parser.add_argument("--num_epoch", help="number of epoch to train model",
                        type=int, default=200)
    args = parser.parse_args()

    print args

    data_dir = args.data_dir
    split_ratio = args.split_ratio
    num_feature = args.num_feature
    log_dir = args.log_dir
    num_epoch = args.num_epoch

    dataset_x, dataset_y = read_training_set(data_dir, save_data='false')
    training_x, training_y, validation_x, validation_y = split_data(dataset_x, dataset_y, split_ratio)

    for learning_rate in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-8, 1e-10]:
        for num_fc in [1, 2, 3]:
            train_model(training_x, training_y, validation_x, validation_y,
                        num_feature=num_feature, num_fc=num_fc, learning_rate=learning_rate,
                        log_dir=log_dir, num_epoch=num_epoch)

