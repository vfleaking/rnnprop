import tensorflow as tf
import numpy as np
import optimizee

class Square(optimizee.Optimizee):
    '''A simple convex function that can be used to combine with other optimizees.'''
    def __init__(self, x_dim):
        optimizee.Optimizee.__init__(self)
        self.x_dim = x_dim

    def build(self):
        self.a = tf.placeholder(tf.float32, shape=[self.x_dim])
    def get_x_dim(self):
        return self.x_dim
    def get_initial_x(self):
        return np.zeros(shape=[self.x_dim])
    def next_internal_feed_dict(self):
        self.internal_feed_dict[self.a] = np.random.uniform(-1.0, 1.0, size=[self.x_dim])
        return self.internal_feed_dict
    def next_feed_dict(self, n_iterations):
        return {}
    def loss(self, i, x):
        return tf.reduce_mean(tf.clip_by_value(tf.square(x - self.a), 0, 10))
