# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import argparse
import random
import tensorflow as tf

from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.externals import joblib # model presistant
from data_reader import _shuffle_data


random.seed(12345)


def _one_hot_encoder(label, class_count=2):
    mat = np.asarray(np.zeros(class_count), dtype='float32').reshape(1, class_count)
    for i in range(class_count):
        if i == label:
            mat[0, i] = 1
    return mat.flatten()


def load_train_df(data_dir):
    print '...... reading csv ......'
    train_df = pd.read_csv(data_dir + '/df_train.csv')
    train_id_df = pd.read_csv(data_dir + '/df_id_train.csv', names=['个人编码', '标签'])
    fee_df = pd.read_csv(data_dir + '/fee_detail.csv')

    # merge
    print '...... merge data ......'
    train_merged_df = pd.merge(train_df, fee_df, on='顺序号', how='inner')
    train_dataset_df = pd.merge(train_merged_df, train_id_df, on='个人编码', how='inner')

    train_dataset_df.drop(train_dataset_df.columns[[0, 47, 55, 56, 58, 59,
                                                    68, 69, 71, 72, 73, 74,
                                                    77, 78, 79]],
                          axis=1, inplace=True)
    train_dataset_df.fillna(0, inplace=True)

    return train_dataset_df, train_id_df


def cluster_2_vec(dataset_df,
                  id_df,
                  downsample_ratio,
                  model_name,
                  k_means_method='k-means++',
                  num_cluster=20,
                  ):
    """
    :param dataset_df:
    :param id_df:
    :param k_means_method: k-means++ or random
    :param num_cluster:
    :return:
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans

    dataset_x = np.asarray(dataset_df.ix[:, 1: len(dataset_df.columns) - 1].as_matrix(), dtype='float32')
    dataset_x = normalize(dataset_x, axis=1)

    print '...... kmeans ......'
    mbk = MiniBatchKMeans(init=k_means_method, n_clusters=num_cluster, batch_size=1000,
                          n_init=10, max_no_improvement=10, verbose=0)

    mbk.fit(dataset_x)

    dataset_df['cluster'] = mbk.labels_

    # save kmeans model
    joblib.dump(mbk, "kmeans_model/%s.pkl" % model_name)

    print '...... to feature vectors ......'

    train_person_ids = set(id_df['个人编码'])

    feature_vecs = []
    feature_labels = []

    for person_id in tqdm(train_person_ids):
        person_pd = dataset_df[dataset_df['个人编码'] == person_id]
        person_vec = np.zeros(num_cluster)
        person_label = np.asarray(person_pd.iloc[0][len(person_pd.columns) - 2], dtype='float32')

        # down sampling
        if int(person_label)==0:
            if np.random.rand() <= downsample_ratio:
                for c in range(num_cluster):
                    person_vec[c] = np.sum(person_pd['cluster'] == c)

                feature_vecs.append(person_vec)
                feature_labels.append(_one_hot_encoder(int(person_label)))
        else:
            for c in range(num_cluster):
                person_vec[c] = np.sum(person_pd['cluster'] == c)

            feature_vecs.append(person_vec)
            feature_labels.append(_one_hot_encoder(int(person_label)))

    _shuffle_data(feature_vecs, feature_labels)

    feature_vecs = np.asarray(feature_vecs, dtype='float32')
    feature_labels = np.asarray(feature_labels, dtype='float32')

    print 'postive sample:%d, negative sample:%d' % (np.sum(feature_labels[:, 1]==1), np.sum(feature_labels[:, 0]==1))
    return feature_vecs, feature_labels, mbk


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


def _make_hparam_string(learning_rate, num_fc, num_feature):
    fc_param = "fc=%d" % num_fc
    num_feat = "num_feat=%d" % num_feature
    return "lr_%.0E,%s,%s" % (learning_rate, fc_param, num_feat)


def train_model(training_x, training_y, validation_x, validation_y,
                num_feature,
                num_fc,
                learning_rate,
                log_dir,
                num_epoch,
                classes=2,
                batch_size=50,
                output=True):

    tf.reset_default_graph()
    sess = tf.Session()

    print '...... building model .......'
    x = tf.placeholder(tf.float32, [None, num_feature])
    y = tf.placeholder(tf.float32, [None, classes])
    keep_prob = tf.placeholder(tf.float32)

    if num_fc == 3:
        fc1 = fc_layer(x, num_feature, 100, "fc1")
        fc2 = fc_layer(fc1, 100, 20, "fc2")
        logits = fc_layer(fc2, 20, classes, "fc3")
    elif num_fc == 2:
        fc1 = fc_layer(x, num_feature, 100, "fc1")
        logits = fc_layer(fc1, 100, classes, "fc2")
    else:# num_fc == 1
        logits = fc_layer(x, num_feature, classes, "fc1")

    with tf.name_scope("dropout"):
        logits = tf.nn.dropout(logits, keep_prob=keep_prob)

    with tf.name_scope("softmax"):
        y_pred = tf.nn.softmax(logits)

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.square(y - y_pred))
        tf.summary.scalar("cost", cost)

    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.name_scope('accuracy'):
        # compute the MIL accuracy
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()
    saver = tf.train.Saver()

    print '...... initializing ......'
    sess.run(tf.global_variables_initializer())
    log_to = log_dir + _make_hparam_string(learning_rate, num_fc, num_feature)
    writer = tf.summary.FileWriter(log_to)
    writer.add_graph(sess.graph)

    best_f1 = 0.
    best_epoch = 0

    for epoch in tqdm(range(num_epoch)):

        train_acc = 0.
        train_cost = 0.
        validation_acc = 0.
        validation_cost = 0.
        y_train_pred = []
        y_validation_pred = []

        # training
        for mini_batch in range(int(len(training_x)/batch_size)):
            batch_x = training_x[mini_batch*batch_size: (mini_batch+1)*batch_size]
            batch_y = training_y[mini_batch*batch_size: (mini_batch+1)*batch_size]
            # print batch_x.shape
            # print batch_y.shape

            _y_pred, _, _accuracy, _cost = sess.run([y_pred, train_op, accuracy, cost], feed_dict={x: batch_x,
                                                                                                   y: batch_y,
                                                                                                   keep_prob: 0.8})
            y_train_pred.extend(list(np.argmax(_y_pred, axis=1)))

            train_acc += _accuracy
            train_cost += _cost

            # write summary
            if mini_batch == 100:
                [s] = sess.run([summ],
                               feed_dict={x: batch_x,
                                          y: batch_y})
                writer.add_summary(s, epoch)

        if int(len(training_x)/batch_size) % batch_size != 0:
            batch_x = training_x[batch_size*int(len(training_x)/batch_size):]
            batch_y = training_y[batch_size*int(len(training_x)/batch_size):]
            _y_pred, _, _accuracy, _cost = sess.run([y_pred, train_op, accuracy, cost], feed_dict={x: batch_x,
                                                                                                   y: batch_y,
                                                                                                   keep_prob: 0.8})
            y_train_pred.extend(list(np.argmax(_y_pred, axis=1)))

            train_acc += _accuracy
            train_cost += _cost

        # validation
        for mini_batch in range(int(len(validation_x)/batch_size)):
            batch_x = validation_x[mini_batch*batch_size: (mini_batch+1)*batch_size]
            batch_y = validation_y[mini_batch*batch_size: (mini_batch+1)*batch_size]
            _y_pred, _accuracy, _cost = sess.run([y_pred, accuracy, cost], feed_dict={x: batch_x,
                                                                                      y: batch_y,
                                                                                      keep_prob: 1.0})
            y_validation_pred.extend(list(np.argmax(_y_pred, axis=1)))
            validation_acc += _accuracy
            validation_cost += _cost

        if int(len(validation_x)/batch_size) % batch_size != 0:
            batch_x = validation_x[batch_size*int(len(validation_x)/batch_size):]
            batch_y = validation_y[batch_size*int(len(validation_x)/batch_size):]
            _y_pred, _accuracy, _cost = sess.run([y_pred, accuracy, cost], feed_dict={x: batch_x,
                                                                                      y: batch_y,
                                                                                      keep_prob: 1.0})
            y_validation_pred.extend(list(np.argmax(_y_pred, axis=1)))
            validation_acc += _accuracy
            validation_cost += _cost

        train_acc /= int(len(training_x)/batch_size)
        train_cost /= int(len(training_x)/batch_size)
        validation_acc /= int(len(validation_x)/batch_size)
        validation_cost /= int(len(validation_x)/batch_size)

        if output:
            print 'epoch %d, train_acc = %f, train_cost = %f, validation_acc = %f, validation_cost = %f' % \
              (epoch, train_acc, train_cost, validation_acc, validation_cost)

            print 'trainging set, precision:%f, recall:%f, f1:%f' % (metrics.precision_score(y_true=np.argmax(training_y, axis=1), y_pred=y_train_pred),
                                                                 metrics.recall_score(y_true=np.argmax(training_y, axis=1), y_pred=y_train_pred),
                                                                 metrics.f1_score(y_true=np.argmax(training_y, axis=1), y_pred=y_train_pred))
            print 'validation set, precision:%f, recall:%f, f1:%f' % (metrics.precision_score(y_true=np.argmax(validation_y,axis=1), y_pred=y_validation_pred),
                                                                 metrics.recall_score(y_true=np.argmax(validation_y, axis=1), y_pred=y_validation_pred),
                                                                 metrics.f1_score(y_true=np.argmax(validation_y, axis=1), y_pred=y_validation_pred))
        if metrics.f1_score(y_true=np.argmax(validation_y, axis=1), y_pred=y_validation_pred) > best_f1:
            best_f1 = metrics.f1_score(y_true=np.argmax(validation_y, axis=1), y_pred=y_validation_pred)
            best_epoch = epoch
            saver.save(sess, os.path.join(log_to, "model.ckpt"), epoch)


    print "num_feature=%d, num_fc=%d, learning_rate=%f" % (num_feature, num_fc, learning_rate)
    print "best_f1:%f, best_epoch:%d" % (best_f1, best_epoch)

    return best_f1, best_epoch

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", help="the root of csv files",
                        type=str, default=os.path.join(os.getcwd(), 'data'))
    parser.add_argument("--split_ratio", help="split data to training and validation data",
                        type=str, default=0.8)
    parser.add_argument("--log_dir", help="path to store logs",
                        type=str, default=os.path.join(os.getcwd(), 'tb_log') + '/')
    parser.add_argument("--num_epoch", help="number of epoch to train model",
                        type=int, default=2000)
    parser.add_argument("--down_sample", help="downsample rate",
                        type=float, default=0.13)
    args = parser.parse_args()

    print args

    train_dataset_df, train_id_df = load_train_df(data_dir=args.data_dir)


    learning_rate = 1e-5
    num_feature = 40
    num_fc=3

    feature_vecs, feature_labels, mbk_model = cluster_2_vec(train_dataset_df,
                                                            train_id_df,
                                                            downsample_ratio=args.down_sample,
                                                            num_cluster=num_feature,
                                                            model_name=_make_hparam_string(learning_rate, num_fc,
                                                                                           num_feature))
    feature_vecs = normalize(feature_vecs, axis=1)


    """
    for learning_rate in [1e-5, 1e-6, 1e-7, 1e-8]:
        for num_feature in [20, 30, 40, 50]:
            for num_fc in [1, 2, 3]:
                feature_vecs, feature_labels, mbk_model = cluster_2_vec(train_dataset_df,
                                                                        train_id_df,
                                                                        downsample_ratio=args.down_sample,
                                                                        num_cluster=num_feature,
                                                                        model_name=_make_hparam_string(learning_rate, num_fc, num_feature))

                feature_vecs = normalize(feature_vecs, axis=1)

                training_x, training_y, validation_x, validation_y = split_data(feature_vecs, feature_labels,
                                                                                args.split_ratio)
                best_f1, best_epoch = train_model(training_x, training_y, validation_x, validation_y,
                        num_feature=num_feature,
                        num_fc=num_fc,
                        learning_rate=learning_rate,
                        log_dir=args.log_dir,
                        num_epoch=args.num_epoch,
                        output=False)
                print "\n\n\n\n"
    """
