import scipy.misc as spm
from keras.preprocessing.image import *


def normalize(images, new_max, new_min, old_max=None, old_min=None):
    if old_min is None:
        old_min = np.min(images)
    if old_max is None:
        old_max = np.max(images)

    return (images - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min


def crop_image(img, cropping):
    return img[cropping[0]:img.shape[0] - cropping[1], cropping[2]:img.shape[1] - cropping[3], :]


def get_cropped_shape(img_shape, cropping):
    return (img_shape[0] - cropping[0] - cropping[1],
            img_shape[1] - cropping[2] - cropping[3],
            img_shape[2])


def resize_image(img, size):
    return spm.imresize(img, size)


def extract_filename(path):
    return path.split('/')[-1]


def adjust_path(path, new_location):
    return '%s/%s' % (new_location, extract_filename(path))


def load_images(paths, target_size):
    images = np.zeros((len(paths), *target_size, 3))
    for i, p in enumerate(paths):
        img = load_img(p, target_size=target_size)
        img = img_to_array(img, dim_ordering='tf')
        images[i] = img

    return images


class RegressionImageDataGenerator(object):
    """Generate minibatches with
    real-time data augmentation.

    This implementation is a modified version of the ImageDataGenerator from Keras
    (https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py).

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        rotation_value_transform: function to modify the label based on the rotation.
        width_shift_range: fraction of total width.
        width_shift_value_transform: function to modify the label based on the width_shift.
        height_shift_range: fraction of total height.
        height_shift_value_transform: function to modify the label based on the height_shift.
        shear_range: shear intensity (shear angle in radians).
        shear_value_transform: function to modify the label based on the shear.
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        zoom_value_transform: function to modify the label based on the zoom.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        horizontal_flip_value_transform: function to modify the label based on the horizontal_flip.
        vertical_flip: whether to randomly flip images vertically.
        vertical_flip_value_transform: function to modify the label based on the vertical_flip.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation).
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 rotation_value_transform=None,
                 width_shift_range=0.,
                 width_shift_value_transform=None,
                 height_shift_range=0.,
                 height_shift_value_transform=None,
                 shear_range=0.,
                 shear_value_transform=None,
                 zoom_range=0.,
                 zoom_value_transform=None,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 horizontal_flip_value_transform=None,
                 vertical_flip=False,
                 vertical_flip_value_transform=None,
                 rescale=None,
                 dim_ordering='default',
                 cropping=(0, 0, 0, 0)):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row and '
                            'column) or "th" (channel before row and column). '
                            'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

        self.cropping = cropping

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return RegressionNumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def flow_from_directory(self, directory, values,
                            target_size=(256, 256), color_mode='rgb',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg'):
        return RegressionDirectoryIterator(
            directory, values, self,
            target_size=target_size, color_mode=color_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def crop(self, x):
        return crop_image(x, self.cropping)

    def standardize(self, x):
        if self.rescale:
            if callable(self.rescale):
                x = self.rescale(x)
            else:
                x *= self.rescale

        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

        return x

    def random_transform(self, x, y):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.rotation_value_transform:
            y = self.rotation_value_transform(y, theta)

        if self.height_shift_range:
            px = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            tx = px * x.shape[img_row_index]
        else:
            tx = 0

        if self.height_shift_value_transform:
            y = self.height_shift_value_transform(y, px)

        if self.width_shift_range:
            py = np.random.uniform(-self.width_shift_range, self.width_shift_range)
            ty = py * x.shape[img_col_index]
        else:
            ty = 0

        if self.width_shift_value_transform:
            y = self.width_shift_value_transform(y, py)

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.shear_value_transform:
            y = self.shear_value_transform(y, shear)

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        if self.zoom_value_transform:
            y = self.zoom_value_transform(y, zx, zy)

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip and np.random.random() < 0.5:
            x = flip_axis(x, img_col_index)
            if self.horizontal_flip_value_transform:
                y = self.horizontal_flip_value_transform(y)

        if self.vertical_flip and np.random.random() < 0.5:
            x = flip_axis(x, img_row_index)
            if self.vertical_flip_value_transform:
                y = self.vertical_flip_value_transform(y)

        return x, y

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        '''
        if seed is not None:
            np.random.seed(seed)

        X = np.copy(X)
        cropped = np.zeros((X.shape[0], *get_cropped_shape(X.shape[1:], self.cropping)))
        for i, img in enumerate(X):
            cropped[i] = self.crop(img)
        X = cropped

        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= (self.std + 1e-7)

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[0]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)


class RegressionNumpyArrayIterator(Iterator):
    """
        This implementation is a modified version of the NumpyArrayIterator from Keras
        (https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py).
    """

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if y is not None and len(X) != len(y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(RegressionNumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        output_shape = get_cropped_shape(self.X[0].shape, self.image_data_generator.cropping)
        batch_x = np.zeros((current_batch_size,) + output_shape)
        batch_y = np.zeros(current_batch_size)
        for i, j in enumerate(index_array):
            x = self.X[j]
            y = self.y[j]
            x = self.image_data_generator.crop(x)
            x, y = self.image_data_generator.random_transform(x, y)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = y
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        return batch_x, batch_y


class RegressionDirectoryIterator(Iterator):
    """
        This implementation is a modified version of the DirectoryIterator from Keras
        (https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py).
    """

    def __init__(self, paths, values, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.paths = paths
        self.values = values
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')

        self.color_mode = color_mode
        self.dim_ordering = dim_ordering

        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        elif self.dim_ordering == 'tf':
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = (1,) + self.target_size

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        self.nb_sample = len(paths)
        self.nb_values = len(values)
        if self.nb_sample != self.nb_values:
            raise ValueError("Number of values=%d does not match "
                             "number of samples=%d" % (self.nb_values, self.nb_sample))

        super(RegressionDirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        output_shape = get_cropped_shape(self.image_shape, self.image_data_generator.cropping)
        batch_x = np.zeros((current_batch_size,) + output_shape)
        batch_y = np.zeros(current_batch_size)
        grayscale = self.color_mode == 'grayscale'

        # build batch of image data
        for i, j in enumerate(index_array):
            path = self.paths[j]
            img = load_img(path, grayscale=grayscale, target_size=self.target_size)

            y = self.values[j]
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            x = self.image_data_generator.crop(x)
            x, y = self.image_data_generator.random_transform(x, y)
            x = self.image_data_generator.standardize(x)

            batch_x[i] = x
            batch_y[i] = y

        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        return batch_x, batch_y
