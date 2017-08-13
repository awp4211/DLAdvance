# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import random

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


def load_test_df(data_dir):
    print '...... reading csv ......'
    test_df = pd.read_csv(data_dir + '/df_test.csv')
    test_id_df = pd.read_csv(data_dir + '/df_id_test.csv', names=['个人编码', '标签'])
    fee_df = pd.read_csv(data_dir + '/fee_detail.csv')

    # merge
    print '...... merge data ......'
    test_merged_df = pd.merge(test_df, fee_df, on='顺序号', how='inner')

    test_merged_df.drop(test_merged_df.columns[[0, 47, 55, 56, 58, 59,
                                                68, 69, 71, 72, 73, 74,
                                                77, 78, 79]],
                        axis=1, inplace=True)
    test_merged_df.fillna(0, inplace=True)

    return test_merged_df, test_id_df


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