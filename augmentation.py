from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
from imutils import paths
import os


def generate_new_images_from_image(image_path, output_directory):
    """
    Given an image path, generates 10 new images applying some transformations
    to the original image
    :param image_path: path of the image
    :param output_directory: target directory where the new images are going to be saved
    :return: 
    """

    image_label = image_path.split(os.path.sep)[-1].split('-')[0]
    image_prefix = image_label + '-'

    image = load_img(image_path)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             fill_mode='nearest')
    total = 0

    image_gen = aug.flow(image, batch_size=1, save_to_dir=output_directory, save_prefix=image_prefix, save_format='ppm')

    for image in image_gen:
        total += 1

        if total == 10:
            break


def generate_new_data(base_directory, output_directory):
    """
    Generate new images given a base directory. For each image, apply some transformations
    to it and get 10 new images
    :param base_directory: the path where images are saved
    :param output_directory: the path where the new images are going to be saved
    :return: 
    """
    images_path = list(paths.list_files(base_directory, validExts=('.ppm')))
    for path in images_path:
        generate_new_images_from_image(path, output_directory)



