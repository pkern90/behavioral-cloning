# Imports
import json

import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Convolution2D, Input, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.regularizers import l2
from keras.utils.visualize_util import plot

import matplotlib.pyplot as plt

from utils import RegressionImageDataGenerator, get_cropped_shape, load_images

import numpy as np
np.random.seed(7)

# Constants
IMG_SIZE = [160, 320]
CROPPING = (54, 0, 0, 0)
SHIFT_OFFSET = 0.2
SHIFT_RANGE = 0.2


# Data loading
def get_generator(from_directory=False, batch_size=32, fit_sample_size=None):
    header = ['center_img', 'left_img', 'right_img', 'steering_angle', 'throttle', 'break', 'speed']
    data_paths = ['data/track1_central/driving_log.csv',
                  'data/track1_recovery/driving_log.csv',
                  'data/track1_reverse/driving_log.csv',
                  'data/track1_recovery_reverse/driving_log.csv',
                  'data/track2_central/driving_log.csv']

    val_paths = ['data/track1_test/driving_log.csv',
                 'data/track2_test/driving_log.csv']

    log = pd.concat([pd.read_csv(path, names=header) for path in data_paths])
    val_log = pd.concat([pd.read_csv(path, names=header) for path in val_paths])

    # Create feature value pairs for the left camera images by subtracting the offset from the steering angle.
    log_left = log[['left_img', 'steering_angle']].copy()
    log_left.loc[:, 'steering_angle'] -= SHIFT_OFFSET

    # Create feature value pairs for the right camera images by adding the offset to the steering angle.
    log_right = log[['right_img', 'steering_angle']].copy()
    log_right.loc[:, 'steering_angle'] += SHIFT_OFFSET

    paths = pd.concat([log.center_img, log_left.left_img, log_right.right_img]).str.strip()
    values = pd.concat([log.steering_angle, log_left.steering_angle, log_right.steering_angle])

    datagen = RegressionImageDataGenerator(rescale=lambda x: x / 127.5 - 1.,
                                           horizontal_flip=True,
                                           channel_shift_range=0.2,
                                           width_shift_range=SHIFT_RANGE,
                                           width_shift_value_transform=lambda val, shift: val - (
                                               (SHIFT_OFFSET / SHIFT_RANGE) * shift),
                                           horizontal_flip_value_transform=lambda val: -val,
                                           cropping=CROPPING)

    val_datagen = RegressionImageDataGenerator(rescale=lambda x: x / 127.5 - 1.,
                                               cropping=CROPPING)

    if fit_sample_size is not None:
        sample_to_fit = load_images(paths.sample(fit_sample_size), IMG_SIZE)
        datagen.fit(sample_to_fit)
        val_datagen.fit(sample_to_fit)
        del sample_to_fit

    if from_directory:
        return (datagen.flow_from_directory(paths.values, values.values, shuffle=True, target_size=IMG_SIZE, batch_size=batch_size),
                val_datagen.flow_from_directory(val_log.center_img.values, val_log.steering_angle.values, shuffle=True, target_size=IMG_SIZE, batch_size=batch_size))
    else:
        images = load_images(paths, IMG_SIZE)
        val_images = load_images(val_log.center_img, IMG_SIZE)

        return (datagen.flow(images, values.values, shuffle=True, batch_size=batch_size),
                val_datagen.flow(val_images, val_log.steering_angle.values, shuffle=True, batch_size=batch_size))


def get_model():
    input_layer = Input(shape=get_cropped_shape((*IMG_SIZE, 3), CROPPING))
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)

    # Remove the last block of the VGG16 net.
    [base_model.layers.pop() for _ in range(4)]
    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []

    # Make sure pre trained layers from the VGG net don't change while training.
    for layer in base_model.layers:
        layer.trainable = False

    # Add last block to the VGG model with modified sub sampling.
    layer = base_model.outputs[0]
    layer = Convolution2D(512, 3, 3, subsample=(2, 2), activation='relu', border_mode='same', name='block5_conv1')(layer)
    layer = Convolution2D(512, 3, 3, subsample=(1, 2), activation='relu', border_mode='same', name='block5_conv2')(layer)
    layer = Convolution2D(512, 3, 3, subsample=(1, 2), activation='relu', border_mode='same', name='block5_conv3')(layer)

    layer = Flatten()(layer)
    layer = Dropout(.2)(layer)
    layer = Dense(2048,  activation='relu', name='fc1')(layer)
    layer = Dropout(.2)(layer)
    layer = Dense(2048, activation='relu', name='fc2')(layer)
    layer = Dropout(.5)(layer)
    layer = Dense(1, activation='linear', name='predictions')(layer)

    return Model(input=base_model.input, output=layer)

if __name__ == '__main__':
    model = get_model()
    model.summary()
    plot(model, to_file='model.png', show_shapes=True)

    model.compile(optimizer='adam', loss='mse')

    # Persist trained model
    model_json = model.to_json()
    with open('model.json', 'w') as f:
        json.dump(model_json, f)

    rdi_train, rdi_val = get_generator(True, batch_size=128)

    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    # Train the model with exactly one version of each image
    history = model.fit_generator(rdi_train,
                                  samples_per_epoch=rdi_train.N,
                                  validation_data=rdi_val,
                                  nb_val_samples=rdi_val.N,
                                  nb_epoch=50,
                                  callbacks=[checkpoint, early_stopping])

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
