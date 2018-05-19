import tensorflow as tf
from tensorflow.contrib.layers import flatten


class LeNet:
    @staticmethod
    def build(X):
        # CONV1: input -> 32x32x3  output -> 28x28x6
        conv1_W = init_weight((5, 5, 3, 6))
        conv1_b = init_bias(6)
        conv1 = tf.nn.conv2d(X, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        conv1 = tf.nn.relu(conv1)

        # POOL1: input -> 28x28x6  output -> 14x14x6
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # CONV2: INPUT -> 14X14X6 OUTPUT -> 10X10X16
        conv2_W = init_weight((5, 5, 6, 16))
        conv2_b = init_bias(16)
        conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        conv2 = tf.nn.relu(conv2)

        # POOL2
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flatten Input -> 5x5x16 output -> 400
        fc0 = flatten(conv2)

        # FC Layer Input -> 400 output ->120
        fc1_W = init_weight((400, 120))
        fc1_b = init_bias(120)
        fc1 = tf.matmul(fc0, fc1_W) + fc1_b
        fc1 = tf.nn.relu(fc1)
        # fc1_dropout = tf.nn.dropout(fc1, keep_prob=0.5)

        # FC layer Input -> 120  Output 84
        fc2_W = init_weight((120, 84))
        fc2_b = init_bias(84)
        fc2 = tf.matmul(fc1, fc2_W) + fc2_b
        fc2 = tf.nn.relu(fc2)
        fc2_dropout = tf.nn.dropout(fc2, keep_prob=0.5)

        fc3_W = init_weight((84, 43))
        fc3_b = init_bias(43)
        Y_pred = tf.matmul(fc2_dropout, fc3_W) + fc3_b

        return Y_pred


def init_weight(shape):
    """
    Initialize the weights for a layer given the shape, using xavier initialization
    :param shape: the shape of the weights
    :return: a tensorflow Variable with the initial weights
    """
    initializer = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initializer(shape))


def init_bias(shape):
    """
    Initialize bias tensor given the shape
    :param shape: shape of the bias
    :return: a tensorflow Variable with the initial bias
    """
    b = tf.zeros(shape)
    return tf.Variable(b)