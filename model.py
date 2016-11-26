from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, ELU, Lambda

from utils import RegressionImageDataGenerator
import numpy as np
import json
import pandas as pd

import matplotlib.pyplot as plt

IMG_SIZE = [100, 200]
SHIFT_OFFSET = 0.2
SHIFT_RANGE = 0.2


header = ['center_img', 'left_img', 'right_img', 'steering_angle', 'throttle', 'break', 'speed']
log = pd.read_csv('data/track1_central/driving_log.csv', names=header)
log_recovery = pd.read_csv('data/track1_recovery/driving_log.csv', names=header)
log_reverse = pd.read_csv('data/track1_reverse/driving_log.csv', names=header)
log_recovery_reverse = pd.read_csv('data/track1_recovery_reverse/driving_log.csv', names=header)

log_val = pd.read_csv('data/track1_test/driving_log.csv', names=header)

log = pd.concat([log, log_reverse, log_recovery, log_recovery_reverse])
#log = pd.concat([log, log_reverse])

log_left = log[['left_img', 'steering_angle']].copy()
log_left.loc[:, 'steering_angle'] -= SHIFT_OFFSET

log_right = log[['right_img', 'steering_angle']].copy()
log_right.loc[:, 'steering_angle'] += SHIFT_OFFSET

paths = pd.concat([log.center_img, log_left.left_img, log_right.right_img]).str.strip()
values = pd.concat([log.steering_angle, log_left.steering_angle, log_right.steering_angle])

datagen = RegressionImageDataGenerator(rescale=lambda x: x/127.5 - 1.,
                                       horizontal_flip=True,
                                       channel_shift_range=0.1,
                                       width_shift_range=SHIFT_RANGE,
                                       width_shift_value_transform=lambda val, shift: val - ((SHIFT_OFFSET/SHIFT_RANGE)*shift),
                                       horizontal_flip_value_transform=lambda val: -val)

val_datagen = RegressionImageDataGenerator(rescale=lambda x: x/127.5 - 1.)

rdi_train = datagen.flow_from_directory(paths.values,
                                        values.values,
                                        target_size=IMG_SIZE,
                                        shuffle=False)

rdi_val = val_datagen.flow_from_directory(log_val['center_img'].values,
                                          log_val['steering_angle'].values,
                                          target_size=IMG_SIZE,
                                          shuffle=True)

# x, y = next(rdi_train)
# print(y[0])
# plt.imshow(x[0])
# plt.show()

# plt.hist(y)
# plt.show()


model = Sequential()
model.add(GaussianNoise(0.2, input_shape=(66, 200, 3)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(1, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(.2))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse')

model.fit_generator(rdi_train,
                    validation_data=rdi_train,
                    nb_val_samples=len(log_val),
                    samples_per_epoch=len(paths),
                    nb_epoch=5)

model_json = model.to_json()
with open('model.json', 'w') as f:
    json.dump(model_json, f)

model.save_weights('model.h5')

