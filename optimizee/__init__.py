import tensorflow as tf
import numpy as np
import math

class Optimizee:
    def __init__(self):
        self.internal_feed_dict = {}
        self.session = tf.get_default_session()
        self.coe = None
        self.against_vars = None
        self.ws = []

    def build(self):
        raise NotImplementedError()
    def get_x_dim(self):
        raise NotImplementedError()
    def get_initial_x(self):
        raise NotImplementedError()
    def next_internal_feed_dict(self):
        raise NotImplementedError()
    def next_feed_dict(self, n_iterations):
        raise NotImplementedError()
    def loss(self, i, x):
        raise NotImplementedError()
    def best_optimizer(self):
        raise NotImplementedError()

    def get_n(self, shape):
        n = 1
        for s in shape:
            n *= s
        return n

    def start_get_weights(self, weights_x):
        self.weights_x = weights_x
        self.weights_xl = 0
    def get_weights(self, shape):
        n = self.get_n(shape)
        self.weights_xl += n
        return tf.reshape(self.weights_x[self.weights_xl - n : self.weights_xl], shape=shape)

import lstm, mnist, tricks, trivial, vgg

__all__ = ['lstm', 'mnist', 'tricks', 'trivial', 'vgg']
