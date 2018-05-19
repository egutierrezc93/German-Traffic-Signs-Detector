from tf_linear import convert_to_one_hot
import requests
import augmentation
import zipfile
from imutils import paths
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import cv2


def compute_class_weights(labels):
    """
    Compute the weight for each class in order to handle class imbalance
    :param labels: labels
    :return: class_weights, numpy array with the weight for each class
    """
    Y = convert_to_one_hot(labels)
    class_totals = Y.sum(axis=0)
    class_weights = class_totals.max() / class_totals
    return class_weights


def generate_new_data(base_directory):
    """
    Generates new images given a base directory. For each image, add some transformations
      and get a total of 10 images for each image in the base directory
    :param base_directory: the directory where images are saved
    :return: 
    """
    augmentation.generate_new_data(base_directory, base_directory)


def show_data_predictions(data, predictions, class_names):
    """
    Shows the images that are given in data, along with their respective class labels
    predictions is in range 0-42 for a 43 total classes
    :param data: array of images, numpy array
    :param predictions: the predictions (integers) specifying the class per image.    
    :return: 
    """
    for (idx, image) in zip(predictions, data):
        image = cv2.resize(image, (400, 400))
        cv2.putText(image, 'Label: {}'.format(class_names[idx]),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        print(class_names[idx])


def download_full_dataset(url, filepath):
    """
    Download the dataset from a remote url to a given file path.
    :param url: remote url
    :param filepath: local path, should include the filename with extension
    :return: 
    """
    print('starting download...')
    r = requests.get(url)
    f = open(filepath, "wb")
    f.write(r.content)
    f.close()
    print('download finished')


def unzip_file(input_file, output_directory):
    """
    Extracts the content from a zip file in a target directory
    :param input_file: file path of the zip file
    :param output_directory: the directory where you want to extract the file
    :return: 
    """
    zip_ref = zipfile.ZipFile(input_file, 'r')
    zip_ref.extractall(output_directory)
    zip_ref.close()


def select_classification_data(dataset_path, allowed_ext, train_data_path, test_data_path):
    """
    Select data for classification, shuffle them, split into training and testing and
     save according to the given training path and test path
    :param dataset_path: The path where data is saved
    :param allowed_ext: The extension of the image files
    :param train_data_path: The path where the training data is going to be saved
    :param test_data_path: The path where the testing data is going to be saved
    :return: 
    """
    image_paths = list(paths.list_files(dataset_path, validExts=(allowed_ext)))
    image_paths = [p for p in image_paths if len(p.split(os.path.sep)) > 3]
    train, test = train_test_split(image_paths, test_size=0.2)
    save_data_on_disk(train, train_data_path)
    save_data_on_disk(test, test_data_path)


def save_data_on_disk(image_paths, directory):
    """
    Save images in the output target directory given a list of image paths
    :param image_paths: list with paths of each image
    :param directory: target directory where the images are going to be saved
    :return: 
    """
    for path in image_paths:
        im = Image.open(path)
        im_path_array = path.split(os.path.sep)
        image_name = im_path_array[-2] + '-' + im_path_array[-1]
        im.save(os.path.join(directory, image_name))


def load_data(directory, image_ext):
    """
    Load the training data, given a directory and the images file extension. 
    Returns the data as a numpy a array, the labels and class_weights. class_weights
      are necesary to deal with class imabalance problems
    :param directory: directory where the data is saved 
    :param image_ext: extension for the images to train
    :param one_hot: default False, whether Y should be converted to one_hot
    :return: data, labels, class_weights
    """
    image_paths = list(paths.list_files(directory, validExts=(image_ext)))
    data = []
    labels = []
    for(i, image_path) in enumerate(image_paths):
        label = image_path.split(os.path.sep)[-1].split('-')[0]
        image = cv2.imread(image_path)
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        image = img_to_array(image)
        # image = np.expand_dims(image, axis=0)
        data.append(image)
        labels.append(label)
    data = np.array(data).astype('float') / 255.0
    labels = np.array(labels)
    class_weights = compute_class_weights(labels)
    one_hot_labels = convert_to_one_hot(labels)
    return data, labels, class_weights, one_hot_labels


def load_prediction_data(directory):
    """
    Load the data for making predictions given a directory where this images are saved
    Returns the images as a numpy array
    :param directory: data where images are saved
    :return: data, a numpy array
    """
    image_paths = list(paths.list_files(directory, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".ppm")))
    data = []
    for (i, image_path) in enumerate(image_paths):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        image = img_to_array(image)
        data.append(image)
    data = np.array(data).astype('float') / 255.0
    return data