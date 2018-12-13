"""
File: utils.py
Project: msalib
File Created: Thursday, 13th December 2018 10:08:17 pm
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
-----
Copyright - 2018 Qianxiao Li, IHPC, A*STAR
License: MIT License
"""

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


class Dataset(object):
    """Dataset class
    """
    def __init__(self, data, target):
        """Dataset class

        Arguments:
            data {np float} -- inputs
            target {np float} -- targets
        """

        assert len(data.shape) > 1 and len(target.shape) > 1
        assert data.shape[0] == target.shape[0]
        self._num_examples = data.shape[0]
        self._data = data
        self._target = target
        self._epoch_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def target(self):
        return self._target

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def data_shape(self):
        return self._data.shape[1:]

    @property
    def target_shape(self):
        return self._target.shape[1:]

    def next_batch(self, batch_size):
        """Return next batch of data

        Arguments:
            batch_size {int} -- batch size

        Returns:
            tuple of np arrays -- batch input/targets
        """

        train_size = self._num_examples
        start = self._index_in_epoch
        end = self._index_in_epoch + batch_size \
            if self._index_in_epoch+batch_size < train_size else train_size
        if end > train_size:
            self._epoch_completed += 1
            self._data = shuffle(self._data)
            self._target = shuffle(self._target)
        x_batch, y_batch = \
            self._data[start:end], \
            self._target[start:end]
        self._index_in_epoch = end if end < train_size else 0
        return x_batch, y_batch


def convert_to_onehot(inputs, num_classes):
    """Convert to one-hot array

    Arguments:
        inputs {np array} -- inputs
        num_classes {int} -- number of classes

    Returns:
        np array -- inputs in one-hot form
    """

    return np.eye(num_classes)[np.array(inputs).reshape(-1)]


def load_dataset(name, num_train, num_test, lift_dim=1):
    """Load dataset

    Arguments:
        name {string} -- 'mnist' or 'fashion_mnist'
        num_train {int} -- number of training examples
        num_test {int} -- number of testing examples

    Keyword Arguments:
        lift_dim {int} -- lifting dimension (default: {1})

    Returns:
        tuple of Dataset objects -- train/test datasets
    """

    module = getattr(tf.keras.datasets, name)
    (x_train, y_train), (x_test, y_test) = module.load_data()

    def preprocess(x, y, num):
        x = x.reshape(x.shape + (1, )).astype(np.float32)
        x = np.repeat(x, axis=-1, repeats=lift_dim)
        x = x / 255.0
        y = convert_to_onehot(y, 10)
        x = x[:num]
        y = y[:num]
        return x, y

    x_train, y_train = preprocess(x_train, y_train, num_train)
    x_test, y_test = preprocess(x_test, y_test, num_test)

    trainset = Dataset(data=x_train, target=y_train)
    testset = Dataset(data=x_test, target=y_test)
    return trainset, testset
