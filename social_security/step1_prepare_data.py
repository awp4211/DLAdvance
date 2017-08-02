# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cPickle
import argparse

from tqdm import tqdm


if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", help="the root of csv files",
                        type=str, default=os.path.join(os.getcwd(),'data'))

    args = parser.parse_args()

    data_dir = args.data_dir

    print '...... loading data ......'

    train_df = pd.read_csv(data_dir + '/df_train.csv')
    train_id_df = pd.read_csv(data_dir + '/df_id_train.csv', names=['个人编码', '标签'])

    test_df = pd.read_csv(data_dir + '/df_test.csv')
    test_id_df = pd.read_csv(data_dir + '/df_id_test.csv', names=['个人编码'])
    fee_df = pd.read_csv(data_dir + '/fee_detail.csv')


    print '...... merge data ......'
    # merge data
    train_merged_df = pd.merge(train_df, fee_df, on='顺序号', how='inner')
    train_dataset_df = pd.merge(train_merged_df, train_id_df, on='个人编码', how='inner')

    test_merged_df = pd.merge(test_df, fee_df, on='顺序号', how='inner')
    test_dataset_df = pd.merge(test_merged_df, test_id_df, on='个人编码', how='inner')

    # wash data
    print '...... wash data .......'
    # delete u'顺序号'(0), u'交易时间'(47), u'住院开始时间'(55),u'住院终止时间'(56), u'申报受理时间'(58),
    #        u'出院诊断病种名称'(59),u'操作时间'(68), u'医院编码_y'(69), u'三目服务项目名称'(71),u'三目医院服务项目名称'(72),
    #        u'剂型'(73), u'规格'(74), u'拒付原因编码'(77),u'拒付原因'(78),u'费用发生时间'(79),
    train_dataset_df.drop(train_dataset_df.columns[[0, 47, 55, 56, 58, 59,
                                                    68, 69, 71, 72, 73, 74,
                                                    77, 78, 79]],
                          axis=1, inplace=True)

    test_dataset_df.drop(test_dataset_df.columns[[0, 47, 55, 56, 58, 59,
                                                  68, 69, 71, 72, 73, 74,
                                                  77, 78, 79]],
                          axis=1, inplace=True)

    # fill nan
    # u'一次性医用材料自费金额'(37) NAN==>0
    # u'一次性医用材料拒付金额'(38) NAN==>0
    # u'一次性医用材料申报金额'(39) NAN==>0
    # u'农民工医疗救助计算金额'(48) NAN==>0
    # u'公务员医疗补助基金支付金额'(49) NAN==>0
    # u'城乡救助补助金额'(50) NAN==>0
    # u'本次审批金额'(60) NAN==>0
    # u'补助审批金额'(61) NAN==>0
    # u'医疗救助医院申请'(62) NAN==>0
    # 残疾军人医疗补助基金支付金额(63) NAN==>0
    # 民政救助补助金额(64) NAN==>0
    # 城乡优抚补助金额(65) NAN==>0
    # 非典补助补助金额(66) NAN==>0
    # 家床起付线剩余(67) NAN==>0
    train_dataset_df.fillna(0, inplace=True)
    test_dataset_df.fillna(0, inplace=True)

    # save data
    print '...... save data ......'
    train_person_ids = set(train_dataset_df['个人编码'])
    train_bags = []
    train_labels = []

    print '...... process training data ......'
    for person_id in tqdm(train_person_ids):
        person_pd = train_dataset_df[train_dataset_df['个人编码'] == person_id]
        train_bag = np.asarray(person_pd.ix[:, 1: len(person_pd.columns) - 2].as_matrix(), dtype='float32')
        # assert(train_bag.shape[1]==63)
        train_label = np.asarray(person_pd.iloc[0][len(person_pd.columns) - 1], dtype='float32')
        train_bags.append(train_bag)
        train_labels.append(train_label)

    print '...... save training data .......'
    cPickle.dump(train_bags, open(data_dir + "/train_bags.pkl", "wb"))
    cPickle.dump(train_labels, open(data_dir + "/train_labels.pkl", "wb"))

    print '...... process testing data ......'
    test_person_ids = set(test_dataset_df['个人编码'])
    test_bags = []
    test_person_ids_list = []
    for person_id in tqdm(test_person_ids):
        person_pd = test_dataset_df[test_dataset_df['个人编码'] == person_id]
        test_bag = np.asarray(person_pd.ix[:, 1: len(person_pd.columns) - 1].as_matrix(), dtype='float32')
        test_bags.append(test_bag)
        test_person_ids_list.append(person_id)

    print '...... save testing data ......'
    cPickle.dump(test_bags, open(data_dir + "/test_bags.pkl", "wb"))
    cPickle.dump(test_person_ids_list, open(data_dir + "/test_person_ids.pkl", "wb"))
