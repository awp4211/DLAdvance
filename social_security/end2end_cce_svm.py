# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import argparse

from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn import metrics

from csv_loader import load_train_df, load_test_df, split_data
from kmeans_pack import kmeans_cluster_2_vec, kmeans_cluster_2_vec_test

np.random.seed(12345)


def prepare_data(data_dir,
                 down_sample_rate,
                 num_feature,
                 split_ratio):
    # training data and validation data
    train_dataset_df, train_id_df = load_train_df(data_dir=data_dir)
    train_feature_vecs, train_feature_labels, mbk_model = kmeans_cluster_2_vec(train_dataset_df,
                                                                               train_id_df,
                                                                               downsample_ratio=down_sample_rate,
                                                                               k_means_method='k-means++',
                                                                               num_cluster=num_feature)
    train_feature_vecs = normalize(train_feature_vecs, axis=1)

    training_x, training_y, validation_x, validation_y = split_data(train_feature_vecs,
                                                                    train_feature_labels,
                                                                    split_ratio)
    # testing data
    test_dataset_df, test_id_df = load_test_df(data_dir=data_dir)

    test_feature_vecs, test_ids = kmeans_cluster_2_vec_test(test_dataset_df, test_id_df, mbk_model, num_feature)
    test_feature_vecs = normalize(test_feature_vecs, axis=1)

    return training_x, training_y, validation_x, validation_y, test_feature_vecs, test_ids


def train_svm(datas,
              svm_kernel,
              class_weight={1: 10}):
    clf = svm.SVC(kernel=svm_kernel, class_weight=class_weight, C=1.0)

    training_x, training_y, validation_x, validation_y, test_feature_vecs, test_ids = datas

    training_y = np.argmax(training_y, axis=1)
    validation_y = np.argmax(validation_y, axis=1)

    print '...... training SVM ......'
    clf.fit(training_x, training_y)
    print '...... predict .......'

    validation_predict = clf.predict(validation_x)
    print 'accuracy = {0}'.format(metrics.accuracy_score(y_true=validation_y, y_pred=validation_predict))
    print 'precision = {0}'.format(metrics.precision_score(y_true=validation_y, y_pred=validation_predict))
    print 'recall = {0}'.format(metrics.recall_score(y_true=validation_y, y_pred=validation_predict))
    print 'f1 = {0}'.format(metrics.f1_score(y_true=validation_y, y_pred=validation_predict))



if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", help="the root of csv files",
                        type=str, default=os.path.join(os.getcwd(), 'data'))
    parser.add_argument("--split_ratio", help="split data to training and validation data",
                        type=str, default=0.8)
    args = parser.parse_args()

    print args


    for num_feature in [20, 30, 40, 50, 60]:
        for split_ratio in [0.8, 0.7]:
            training_x, training_y, validation_x, validation_y, test_feature_vecs, test_ids = prepare_data(args.data_dir,
                                                                                                   down_sample_rate=1.0,
                                                                                                   num_feature=num_feature,
                                                                                                   split_ratio=split_ratio)
            for svm_kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
                for class_weight in [{1:3}, {1:5}, {1:10}, {1:20}, {1:50}]:
                    print '...... num_feature: %d, split_ratio: %f, svm_kernel:%s,  classweight :%s' % \
                          (num_feature, split_ratio, svm_kernel, class_weight)
                    train_svm(datas=[training_x, training_y, validation_x, validation_y, test_feature_vecs, test_ids],
                        svm_kernel=svm_kernel,
                        class_weight=class_weight)




