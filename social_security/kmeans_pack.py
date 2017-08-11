# -*- coding:utf-8 -*-
import numpy as np

from sklearn.preprocessing import normalize
from tqdm import tqdm
from csv_loader import _one_hot_encoder, _shuffle_data


def kmeans_cluster_2_vec(dataset_df,
                         id_df,
                         downsample_ratio,
                         k_means_method='k-means++',
                         num_cluster=20):
    """
    :param dataset_df:
    :param id_df:
    :param k_means_method: k-means++ or random
    :param num_cluster:
    :return:
    """
    from sklearn.cluster import MiniBatchKMeans

    dataset_x = np.asarray(dataset_df.ix[:, 1: len(dataset_df.columns) - 1].as_matrix(), dtype='float32')
    dataset_x = normalize(dataset_x, axis=1)

    print '...... kmeans ......'
    mbk = MiniBatchKMeans(init=k_means_method, n_clusters=num_cluster, batch_size=1000,
                          n_init=10, max_no_improvement=10, verbose=0)

    mbk.fit(dataset_x)

    dataset_df['cluster'] = mbk.labels_

    print '...... to feature vectors ......'

    train_person_ids = set(id_df['个人编码'])

    feature_vecs = []
    feature_labels = []

    for person_id in tqdm(train_person_ids):
        person_pd = dataset_df[dataset_df['个人编码'] == person_id]
        person_vec = np.zeros(num_cluster)
        person_label = np.asarray(person_pd.iloc[0][len(person_pd.columns) - 2], dtype='float32')

        # down sampling
        if int(person_label) == 0:
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


def kmeans_cluster_2_vec_test(dataset_df,
                       id_df,
                       model,
                       num_cluster):

    dataset_x = np.asarray(dataset_df.ix[:, 1: len(dataset_df.columns)].as_matrix(), dtype='float32')
    dataset_x = normalize(dataset_x, axis=1)

    y_clusters = model.predict(dataset_x)

    dataset_df['cluster'] = y_clusters

    print '...... to feature vectors ......'

    feature_vecs = []
    person_ids = []

    for person_id in tqdm(id_df['个人编码']):
        person_pd = dataset_df[dataset_df['个人编码'] == person_id]
        person_vec = np.zeros(num_cluster)

        # down sampling
        for c in range(num_cluster):
            person_vec[c] = np.sum(person_pd['cluster'] == c)

        feature_vecs.append(person_vec)
        person_ids.append(person_id)

    return feature_vecs, person_ids