import tensorflow as tf
import numpy as np
import optimizee
import math

def lstm_func(x, h, c, wx, wh, b):
    """
        x: (N, D)
        h: (N, H)
        c: (N, H)
        wx: (D, 4H)
        wh: (H, 4H)
        b: (4H, )
    """
    N, H = tf.shape(h)[0], tf.shape(h)[1]
    a = tf.reshape(tf.matmul(x, wx) + tf.matmul(h, wh) + b, (N, -1, H))
    i, f, o, g = a[:,0,:], a[:,1,:], a[:,2,:], a[:,3,:]
    i = tf.sigmoid(i)
    f = tf.sigmoid(f)
    o = tf.sigmoid(o)
    g = tf.tanh(g)
    next_c = f * c + i * g
    next_h = o * tf.tanh(next_c)
    return next_h, next_c

class SinLSTMModel(optimizee.Optimizee):
    '''A simple sequence prediction task implemented by LSTM.'''
    mnist = None

    def __init__(self, n_batches=128, n_h=20, n_l=10, n_lstm=1, noise_scale=0.1, initial_param_scale=0.1):
        optimizee.Optimizee.__init__(self)

        self.n_batches = n_batches
        self.n_h = n_h
        self.n_l = n_l
        self.n_lstm = n_lstm
        self.initial_param_scale = initial_param_scale
        self.noise_scale = noise_scale

        self.x_dim = 0
        self.x_dim += self.get_n([1, 4 * self.n_h])
        for i in range(self.n_lstm - 1):
            self.x_dim += self.get_n([self.n_h, 4 * self.n_h])
        for i in range(self.n_lstm):
            self.x_dim += self.get_n([self.n_h, 4 * self.n_h])
        for i in range(self.n_lstm):
            self.x_dim += self.get_n([4 * self.n_h])
        self.x_dim += self.get_n([self.n_h, 1])
        self.x_dim += self.get_n([1])

    def build(self):
        self.x = tf.placeholder(tf.float32, [None, None, None, 1])
        self.y_ = tf.placeholder(tf.float32, [None, None, 1])

    def get_x_dim(self):
        return self.x_dim
    def get_initial_x(self):
        return np.random.normal(size=[self.x_dim], scale=self.initial_param_scale)
    def next_internal_feed_dict(self):
        return {}
    def next_feed_dict(self, n_iterations):
        x_data = np.zeros([n_iterations, self.n_batches, self.n_l, 1])
        y_data = np.zeros([n_iterations, self.n_batches, 1])
        for i in range(n_iterations):
            for b in range(self.n_batches):
                phi = np.random.uniform(0.0, 2 * math.pi)
                omega = np.random.uniform(0.0, math.pi / 2)
                A = np.random.uniform(0.0, 10.0)
                for k in range(self.n_l):
                    x_data[i][b][k][0] = A * math.sin(k * omega + phi) + np.random.normal(scale=self.noise_scale)
                y_data[i][b][0] = A * math.sin(self.n_l * omega + phi)
        return { self.x: x_data, self.y_: y_data }
    def loss(self, i, x):
        self.start_get_weights(x)

        wx = [self.get_weights([1, 4 * self.n_h])] + [self.get_weights([self.n_h, 4 * self.n_h]) for j in range(self.n_lstm - 1)]
        wh = [self.get_weights([self.n_h, 4 * self.n_h]) for j in range(self.n_lstm)]
        b = [self.get_weights([4 * self.n_h]) for j in range(self.n_lstm)]

        wo = self.get_weights([self.n_h, 1])
        bo = self.get_weights([1])

        h = [tf.zeros([self.n_batches, self.n_h]) for j in range(self.n_lstm)]
        c = [tf.zeros([self.n_batches, self.n_h]) for j in range(self.n_lstm)]

        for k in range(self.n_l):
            last = self.x[i,:,k,:]
            for j in range(self.n_lstm):
                h[j], c[j] = lstm_func(last, h[j], c[j], wx[j], wh[j], b[j])
                last = h[j]

        return tf.reduce_mean(tf.square(tf.matmul(h[-1], wo) + bo - self.y_))
