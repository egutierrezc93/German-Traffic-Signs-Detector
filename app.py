import click
import tf_lenet
import sk_linear
import tf_linear
from utilities import load_prediction_data, show_data_predictions, load_data, unzip_file, download_full_dataset
from utilities import select_classification_data, generate_new_data
import numpy as np
import os

MODEL1_PATH = 'models/model1/saved/model1.pkl'
MODEL2_PATH = 'models/model2/saved/model2.ckpt'
MODEL3_PATH = 'models/model3/saved/lenet.ckpt'
IMAGE_EXTENSION = '.ppm'
DATA_URL = 'http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip'
RAW_DATA_DIRECTORY = 'FullIJCNN2013'
ZIP_FILE_NAME = 'FullIJCNN2013.zip'
IMAGES_BASE_DIRECTORY = 'images'
ZIP_FILE_PATH = os.path.join(IMAGES_BASE_DIRECTORY, ZIP_FILE_NAME)
TRAIN = 'train'
TEST = 'test'
TRAIN_DATA_PATH = os.path.join(IMAGES_BASE_DIRECTORY, TRAIN)
TEST_DATA_PATH = os.path.join(IMAGES_BASE_DIRECTORY, TEST)
MODEL3 = 'model3'
MODEL2 = 'model2'
MODEL1 = 'model1'
GERMAN_TRAFFIC_LABELS = labels_complete = [
    'speed limit 20 (prohibitory)',
    'speed limit 30 (prohibitory)',
    'speed limit 50 (prohibitory)',
    'speed limit 60 (prohibitory)',
    'speed limit 70 (prohibitory)',
    'speed limit 80 (prohibitory)',
    'restriction ends 80 (other)',
    'speed limit 100 (prohibitory)',
    'speed limit 120 (prohibitory)',
    'no overtaking (prohibitory)',
    'no overtaking (trucks) (prohibitory)',
    'priority at next intersection (danger)',
    'priority road (other)',
    'give way (other)',
    'stop (other)',
    'no traffic both ways (prohibitory)',
    'no trucks (prohibitory)',
    'no entry (other)',
    'danger (danger)',
    'bend left (danger)',
    'bend right (danger)',
    'bend (danger)',
    'uneven road (danger)',
    'slippery road (danger)',
    'road narrows (danger)',
    'construction (danger)',
    'traffic signal (danger)',
    'pedestrian crossing (danger)',
    'school crossing (danger)',
    'cycles crossing (danger)',
    'snow (danger)',
    'animals (danger)',
    'restriction ends (other)',
    'go right (mandatory)',
    'go left (mandatory)',
    'go straight (mandatory)',
    'go right or straight (mandatory)',
    'go left or straight (mandatory)',
    'keep right (mandatory)',
    'keep left (mandatory)',
    'roundabout (mandatory)',
    'restriction ends (overtaking) (other)',
    'restriction ends (overtaking (trucks)) (other)'
]


@click.group()
def cli():
    pass


@cli.command()
def download():
    """
    This command downloads the dataset from the remote repository, unzip the file 
    downloaded in a directory, select the classification data only, split this data
    into 20% for test and 80% for training suffling the data prior to split, and 
    finally generate new synthetic data from the training data
    :return: 
    """
    dataset_path = os.path.join(IMAGES_BASE_DIRECTORY, RAW_DATA_DIRECTORY)
    download_full_dataset(DATA_URL, ZIP_FILE_PATH)
    unzip_file(ZIP_FILE_PATH, IMAGES_BASE_DIRECTORY)
    select_classification_data(dataset_path, IMAGE_EXTENSION, TRAIN_DATA_PATH, TEST_DATA_PATH)
    generate_new_data(TRAIN_DATA_PATH)


@cli.command()
@click.option('-m', '--model')
@click.option('-d', '--directory')
def train(model, directory):
    """
    This command load the images from the given directory and train a chosen model
     'model1' corresponds to the linear sklearn model
     'model2' corresponds to the linear tensorflow model
     'model3' corresponds to the lenet tensorflow model
    :param model: the model to be used. Either 'model1', 'model2', and 'model3'
    :param directory: the directory where the training data is saved
    :return: 
    """
    data, labels, class_weights, one_hot_labels = load_data(directory, IMAGE_EXTENSION)
    data_reshaped = data.reshape((data.shape[0], 3072))
    if model == MODEL1:
        sk_linear.train(data_reshaped, labels, MODEL1_PATH)
    elif model == MODEL2:
        tf_linear.model(data_reshaped, one_hot_labels, MODEL2_PATH)
    elif model == MODEL3:
        tf_lenet.model(data, one_hot_labels, epochs=400, class_weights=class_weights, model_path=MODEL3_PATH)


@cli.command()
@click.option('-m', '--model')
@click.option('-d', '--directory')
def test(model, directory):
    """
    This command loads the images from the given directory and evaluates a model
    'model1' corresponds to the linear sklearn model
    'model2' corresponds to the linear tensorflow model
    'model3' corresponds to the lenet tensorflow model
    :param model: the model to be used. Either 'model1', 'model2', and 'model3'
    :param directory: the directory where images are saved
    :return: 
    """
    data, labels, _, one_hot_labels = load_data(directory, IMAGE_EXTENSION)
    data_reshaped = data.reshape((data.shape[0], 3072))
    if model == MODEL1:
        sk_linear.predict(data_reshaped, MODEL1_PATH, labels)
    elif model == MODEL2:
        tf_linear.predict(data_reshaped, MODEL2_PATH, Y_test=one_hot_labels)
    elif model == MODEL3:
        tf_lenet.predict(data, MODEL3_PATH, Y_test=one_hot_labels)


@cli.command()
@click.option('-m', '--model')
@click.option('-d', '--directory')
def infer(model, directory):
    """
    This command loads images from the given directory and makes predictions about
    those images using a given model, showing  a window with the image and its corresponding class label
     'model1' corresponds to the linear sklearn model
     'model2' corresponds to the linear tensorflow model
     'model3' corresponds to the lenet tensorflow model
    :param model: the model to be used. Either 'model1', 'model2', and 'model3'
    :param directory: the directory where images are saved
    :return: 
    """
    data = load_prediction_data(directory)
    if model == MODEL1:
        predictions = sk_linear.predict(data.reshape((data.shape[0], 3072)), MODEL1_PATH)
        show_data_predictions(data, predictions, GERMAN_TRAFFIC_LABELS)
    elif model == MODEL2:
        predictions = tf_linear.predict(data.reshape((data.shape[0], 3072)), MODEL2_PATH)
        predictions = np.argmax(predictions, axis=1)
        show_data_predictions(data, predictions, GERMAN_TRAFFIC_LABELS)
    elif model == MODEL3:
        predictions = tf_lenet.predict(data, MODEL3_PATH)
        predictions = np.argmax(predictions, axis=1)
        show_data_predictions(data, predictions, GERMAN_TRAFFIC_LABELS)


if __name__ == '__main__':
    cli()
