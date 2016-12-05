# Behavioral Cloning

This project was created as an assessment for the [Self-Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) Program by Udacity. The goal is to drive a car autonomously in a simulator using a deep neuronal network (DNN) trained on human driving behavior. For that Udacity provided the simulator and a basic python script to connect a DNN with it. The simulator has two mode. In the "training mode" the car can be controlled through a keyboard or a game pad to generated data. More information about the data and it's structure can be found in the corresponding [section](https://github.com/pkern90/behavioral-cloning/blob/master/README.md#data). In the "autonomous mode" however the car receives it input commands by the python script.

The following animations shows the final model controlling the car on both tracks.

Track 1                       |  Track 2
:----------------------------:|:------------------------------:
![Track 1](images/track1.gif) | ![Track 2](images/track2.gif)


# Getting Started
## Prerequisites

This project requires **Python 3.5** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [matplotlib](http://matplotlib.org/)
- [pandas](http://pandas.pydata.org/)
- [TensorFlow](http://tensorflow.org)
- [Keras](https://keras.io/)
- [h5py](http://www.h5py.org/)

Only needed for driving in the simulator:

- [flask-socketio](https://flask-socketio.readthedocs.io/en/latest/)
- [eventlet](http://eventlet.net/)
- [pillow](https://python-pillow.org/)


## Run The Drive Script
## Retrain The Model
# Structure
## Data

![Sample Log](images/sample_log.png)

Left                                   |  Center                                   |  Right
:-------------------------------------:|:-----------------------------------------:|:-------------------------------------:
![Sample Left](images/left_sample.jpg) | ![Sample Center](images/center_sample.jpg)|![Sample Left](images/right_sample.jpg)


## Model

<a href="https://raw.githubusercontent.com/pkern90/behavioral-cloning/master/images/model_wide.png" target="_blank"><img src="images/model_wide.png"></img> </a>
