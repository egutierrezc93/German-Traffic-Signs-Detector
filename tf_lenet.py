import tensorflow as tf

import numpy as np
from lenet5 import LeNet
from sklearn.metrics import classification_report


def model(X_train, Y_train, epochs, class_weights, model_path, classes=43, learning_rate=0.001, minibatch_size=128):
    """
    Train the lenet network
    :param X_train: X for training
    :param Y_train: Y for training, in one_hot mode
    :param epochs: int, number of epochs
    :param classes: int, number of classes
    :param learning_rate: float, learning rate
    :param minibatch_size: int, batch size
    :param class_weights: array, useful for handling class imbalance
    :return: 
    """

    X, Y = create_placeholders(n_x=32, classes=classes)

    Y_pred = LeNet().build(X)

    cost = compute_cost(Y_pred, Y, class_weights)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]

        for epoch in range(epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)

            for offset in range(0, m, minibatch_size):
                end = offset + minibatch_size
                batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
                _, batch_cost = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})

                epoch_cost += batch_cost / num_minibatches

            print('cost after epoch %i: %f' % (epoch, epoch_cost))

        saver.save(sess, model_path)
        print('Model saved...')


def predict(X_test, model_path, Y_test=None, classes=43):
    """
    Makes predictions with the data provided.
    If X_test is provided, the model is evaluated with the test data, otherwise the predictions are the output
    :param X_test: data to make predictions from 
    :param Y_test: default None, labels in one_hot mode
    :param classes: number of classes
    :return: predictions
    """
    X, Y = create_placeholders(32, classes)
    Y_pred = LeNet().build(X)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_path)

        predictions = sess.run(Y_pred, feed_dict={X: X_test})
        predictions = np.array(predictions)

        if Y_test is not None:
            print(classification_report(Y_test.argmax(axis=1), predictions.argmax(axis=1)))

        return predictions


def create_placeholders(n_x, classes):
    """
    create tensorflow placeholders fro X and Y placeholders 
    :param n_x: number of pixels in one axis
    :param classes: number of classes
    :return: X, Y
    """
    X = tf.placeholder(tf.float32, (None, n_x, n_x, 3))
    Y = tf.placeholder(tf.int32, (None, classes))
    return X, Y


def compute_cost(Y_pred, Y, class_weights):
    """
    Computes cost function handling the class imablance problem
    :param Y_pred: tensor with values predicted from the network
    :param Y: tensor with the ground truth values
    :param class_weights: array with weights for each class  
    :return: the cost
    """
    labels = tf.argmax(Y, axis=1)
    class_weights = tf.constant(class_weights)
    weights = tf.gather(class_weights, labels)
    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=Y_pred, weights=weights))
    return cost
