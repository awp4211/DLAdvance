import tensorflow as tf
import numpy as np

def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.nn.softmax(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def one_hot_encoder(label, class_count=2):
    mat = np.asarray(np.zeros(class_count), dtype='float32').reshape(1, class_count)
    for i in range(class_count):
        if i == label:
            mat[0, i] = 1
    return mat.flatten()


def train(n_epoch=200, learning_rate=0.001):
    x = tf.placeholder(tf.float32, [None, 10])
    y = tf.placeholder(tf.float32, [None, 2])

    fc = fc_layer(x, 10, 2, name="fc")

    y_pred = tf.nn.softmax((tf.reduce_sum(fc, axis=0)))

    cost = tf.reduce_mean(tf.square(y - y_pred))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(tf.expand_dims(y_pred, 0), 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    dataset_x = np.asarray(np.random.rand(20, 20, 10), dtype='float32')
    dataset_y = np.random.randint(0, 2, size=(20, 1))

    label_mat = []
    for l in dataset_y:
        label_mat.append(one_hot_encoder(int(l)))

    dataset_y = np.asarray(label_mat, dtype='float32')

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epoch):

            train_acc = 0.
            train_cost = 0.

            for ex in range(20):
                batch_x = dataset_x[ex]
                batch_y = dataset_y[ex][:, np.newaxis].T
                #print batch_y
                _y_fc, _y_pred, _, _cost, _accuracy = sess.run([fc, y_pred, optimizer, cost, accuracy],
                                           feed_dict={x: batch_x,
                                                      y: batch_y})
                train_acc += _accuracy
                train_cost += _cost
                #print _y_fc
                #print _y_pred

            train_acc /= 20.
            train_cost /=20.

            print 'epoch %d, training acc = %f, training cost = %f' % (epoch, train_acc, train_cost)


if __name__ == '__main__':
    train()
