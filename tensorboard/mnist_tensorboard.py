import tensorflow as tf
import os

from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
mnist_data_path = os.path.join(os.path.dirname(os.getcwd()), 'MNIST_DATA')

print 'loading data from %s ' % mnist_data_path
mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)
print 'loading data done'


LOG_DIR = os.path.join(os.getcwd(), 'tb_log_basic')


def conv_layer(input, channels_in, channel_out, name='conv'):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([5, 5, channels_in, channel_out]))
        b = tf.Variable(tf.zeros([channel_out]))
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME')
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weight", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def fc_layer(input, channels_in, channels_out, name='fc'):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([channels_in, channels_out]))
        b = tf.Variable(tf.zeros([channels_out]))
        act = tf.nn.relu(tf.add(tf.matmul(input, w), b))
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def mnist_model():

    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)

    conv1 = conv_layer(x_image, 1, 32, name='conv1')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = conv_layer(pool1, 32, 64, name='conv2')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    flattened = tf.reshape(pool2, [-1, 7*7*64])

    fc1 = fc_layer(flattened, 7*7*64, 1024, name='fc1')
    logits = fc_layer(fc1, 1024, 10, name='fc2')

    with tf.name_scope('xent'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar("xent", cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


    with tf.name_scope('accuracy'):
        # compute the accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOG_DIR)
    writer.add_graph(sess.graph)

    for i in tqdm(range(2000)):
        batch_x, batch_y = mnist.train.next_batch(100)

        if i % 5 == 0:
            [s] = sess.run([summ],
                            feed_dict={x: batch_x,
                                           y: batch_y})
            writer.add_summary(s, i)

        if (i+1) % 500 == 0:
            [train_accuracy] = sess.run([accuracy], feed_dict={x: batch_x,
                                                               y: batch_y})
            print 'step %d, training accuracy %g' % (i, train_accuracy)

        sess.run(train_step, feed_dict={x: batch_x,
                                        y: batch_y})


if __name__ == '__main__':
    mnist_model()