# German Traffic Signs Detector

This project was made as a requirement for participate in the Kiwi Campus Inc. machine learning challenge.

## Getting Started

This project downloads, delete the object detection images, and reorganizate the data from the [German Traffic Signs Dataset](http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset), this command by implementing this command:

```
python app.py download
```

The German Traffic Signs Detector project also create, train, save and test three different machine learning models:

* Model1: A logistic regression model using scikit-learn,
* Model2: A linear softmax model using TensorFlow,
* Model3: A from scrach LeNet model in TensorFlow, based in the [Yann LeCun Article](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).

All of this models can be trained and tested using this commands:

```
python app.py train -m [chosen model] -d [directory with trainig data]

python app.py test -m [chosen model] -d [directory with test data]
```

[chosen model] can be either: 'model1', 'model2' or 'model3' for the linear sklearn, linear softmax tensorflow and lenet tensorflow respectively. For example:

```
python app.py train -m model3 -d images/train/
```

Finally this code create an application that will use a chosen model and run inference on images saved in a particular directory. For each image in such directory shows a window with the image together with its label and is executed as follows:

```
python app.py infer -m [chosen model] -d [directory with user data]
```

## Built With

* [Python3](https://www.python.org/download/releases/3.0/)

And this libraries:

* [TensorFlow](https://www.tensorflow.org/)
* [Click](http://click.pocoo.org/)
* [Requests](http://docs.python-requests.org/)
* [zipfile](https://docs.python.org/3.4/library/zipfile.html)
* [imutils](https://pypi.org/project/imutils/)
* [scikit-learn](http://scikit-learn.org/)
* [Keras](https://keras.io/)
* [PIL](https://pypi.org/project/PIL/)
* [NumPy](http://www.numpy.org/)
* [OpenCV](https://pypi.org/project/opencv-python/)

## Authors

* **Esteban Gutiérrez Correa** - *Initial work* - [egutierrezc93](https://github.com/egutierrezc93)

See also the list of [contributors](https://github.com/egutierrezc93/German-Traffic-Signs-Detector/contributors) who participated in this project.

