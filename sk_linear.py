from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import classification_report


def train(X, Y, model_path):
    """
    Train a linear model using sklearn
    m denotes the number of training examples
    n_x denotes the pixels flattened into one single array
    :param X: training data, a numpy array with shape (m, n_x)
    :param Y: labels, NOT in one_hot
    :param model_path: path where the model is going to be saved
    :return: 
    """
    print('Training with sklearn')
    logreg = LogisticRegression(C=1e5)
    logreg.fit(X, Y)
    joblib.dump(logreg, model_path)
    predictions = logreg.predict(X)
    print(classification_report(Y, predictions))
    print('Training finished.')


def predict(X, model_path, Y=None):
    """
    Makes predictions using the linear sklearn model. If Y is specified,
    evaluate the model with the test data
    m denotes the number of training examples
    n_x denotes the pixels flattened into one single array
    :param X: test data, a numpy array of shape (m, n_x)
    :param model_path: path where the model is saved
    :param Y: default None, 
    :return: predictions, the classes predicted from the model
    """
    print('Predicting with sklearn')
    classifier = joblib.load(model_path)
    predictions = classifier.predict(X)
    if Y is not None:
        print(classification_report(Y, predictions))
    print('Finishing predictions')
    return predictions.astype(int)
