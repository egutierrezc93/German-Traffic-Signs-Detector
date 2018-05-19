import tensorflow as tf
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import classification_report


def model(X_train, Y_train, model_path, num_epochs=1000, classes=43):
    """
    Train a linear softmax model in tensorflow
    :param X_train: data with shape (number of examples, number of pixels)
    :param Y_train: labels
    :param num_epochs: number of epochs to train 
    :param classes: number of classes
    :return: 
    """
    print('Training linear softmax in tensorflow...')
    n_x = X_train.shape[1]
    X, Y = create_placeholders(n_x, classes)
    parameters = initialize_parameters(n_x, classes)
    Y_pred = forward_propagation(X, parameters)
    cost = compute_cost(Y_pred, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(Y, axis=1), tf.argmax(Y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            _, c = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})

            if epoch % 10 == 0:
                print('cost after epoch %i %f' % (epoch, c))

        parameters = sess.run(parameters)

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))

        save_path = saver.save(sess, model_path)
        print('model saved in path: %s' % save_path)
        return parameters


def predict(X_test, model_path, Y_test=None, classes=43):
    """
    Makes predictions with the data passed
    If Y_test is specified, evaluate the model against the test data and print accuracy 
    :param X_test: data to makes predictions, numpy array of shape (# of examples, # of pixels)
    :param model_path: path where the model is saved
    :param Y_test: labels with the ground truth 
    :param classes: number of classes
    :return: predictions
    """
    n_x = X_test.shape[1]
    X, Y = create_placeholders(n_x, classes)
    parameters = initialize_parameters(n_x, classes)
    Y_pred = forward_propagation(X, parameters)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, model_path)
        # correct_prediction = tf.equal(tf.argmax(Y, axis=1), tf.argmax(Y_pred, axis=1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        predictions = sess.run(Y_pred, feed_dict={X: X_test})
        predictions = np.array(predictions)

        if Y_test is not None:
            print(classification_report(Y_test.argmax(axis=1), predictions.argmax(axis=1)))

        return predictions


def compute_cost(Y_pred, Y):
    """
    Computes the cost given the predicted and the ground truth
    :param Y_pred: 
    :param Y: Ground truth labels
    :return: 
    """
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))


def initialize_parameters(n_x, classes):
    """
    Initializes parameters W and b with zeros
    :param n_x: number of pixels
    :param classes: number of classes
    :return: parameters initialized to zeros
    """
    W = tf.Variable(tf.zeros([n_x, classes]))
    b = tf.Variable(tf.zeros([classes]))
    return W, b


def forward_propagation(X, parameters):
    """
    Computes the linear part of forward propagation. W*X+b 
    :param X: 
    :param parameters: 
    :return: linear result
    """
    W, b = parameters
    Y = tf.matmul(X, W) + b
    return Y


def create_placeholders(n_x, classes):
    """
    Creates tensorflow placeholders for the variables X and Y
    :param n_x: numer of pixels
    :param classes: number of classes
    :return: 
    """
    X = tf.placeholder(tf.float32, [None, n_x])
    Y = tf.placeholder(tf.float32, [None, classes])
    return X, Y


def convert_to_one_hot(Y):
    """
    Convert an array of labels into its corresponding one_hot representation
    :param Y: array with labels
    :return: one_hot encoded labels
    """
    return np_utils.to_categorical(Y)

