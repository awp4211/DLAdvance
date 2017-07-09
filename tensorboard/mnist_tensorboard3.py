import tensorflow as tf
import os
import urllib

from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
mnist_data_path = os.path.join(os.path.dirname(os.getcwd()), 'MNIST_DATA')

print 'loading data from %s ' % mnist_data_path
mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)
print 'loading data done'

GIST_URL = 'https://gist.githubusercontent.com/dandelionmane/4f02ab8f1451e276fea1f165a20336f1/raw/dfb8ee95b010480d56a73f324aca480b3820c180'
LOG_DIR = os.path.join(os.getcwd(), 'tb_log_embedding')+'/'
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)

print 'download embedding data'
urllib.urlretrieve(GIST_URL + 'labels_1024.tsv', LOG_DIR + 'labels_1024.tsv')
urllib.urlretrieve(GIST_URL + 'sprite_1024.png', LOG_DIR + 'sprite_1024.png')
print 'download embedding data done'

def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def mnist_model(learning_rate, use_two_conv, use_two_fc, hparam):
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    if use_two_conv:
        conv1 = conv_layer(x_image, 1, 32, "conv1")
        conv_out = conv_layer(conv1, 32, 64, "conv2")
    else:
        conv1 = conv_layer(x_image, 1, 64, "conv1")
        conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    flattened = tf.reshape(conv_out, [-1, 7*7*64])

    if use_two_fc:
        fc1 = fc_layer(flattened, 7*7*64, 1024, "fc1")
        embedding_input = fc1
        embedding_size = 1024
        logits = fc_layer(fc1, 1024, 10, "fc2")
    else:
        embedding_input = flattened
        embedding_size = 7*7*64
        logits = fc_layer(flattened, 7*7*64, 10, "fc")

    with tf.name_scope("xnet"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name='xent')
        tf.summary.scalar("xent", xent)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(xent)

    with tf.name_scope('accuracy'):
        # compute the accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    # embedding
    embedding = tf.Variable(tf.zeros([1024, embedding_size]), name='test_embedding')
    assignment = embedding.assign(embedding_input)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOG_DIR + hparam)
    writer.add_graph(sess.graph)

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = LOG_DIR + 'sprite_1024.png'
    embedding_config.metadata_path = LOG_DIR + 'labels_1024.tsv'
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    for i in tqdm(range(2001)):
        batch_x, batch_y = mnist.train.next_batch(100)

        if i % 5 == 0:
            [s] = sess.run([summ],
                           feed_dict={x: batch_x,
                                      y: batch_y})
            writer.add_summary(s, i)

        if (i + 1) % 500 == 0:
            # new ----------------------------------
            sess.run(assignment, feed_dict={x: mnist.test.images[:1024],
                                            y: mnist.test.labels[:1024]})
            saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), i)

            [train_accuracy] = sess.run([accuracy], feed_dict={x: batch_x,
                                                               y: batch_y})
            print 'step %d, training accuracy %g' % (i, train_accuracy)

        sess.run(train_step, feed_dict={x: batch_x,
                                        y: batch_y})


def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)


def main():
    for learning_rate in [1E-4, 1E-5, 1E-6]:
        # Include "False" as a value to try different model architectures
        for use_two_fc in [True, False]:
            for use_two_conv in [True, False]:
                # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
                hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                print('Starting run for %s' % hparam)
                # Actually run with the new settings
                mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)


if __name__ == '__main__':
    main()
