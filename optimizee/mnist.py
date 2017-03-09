import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import optimizee

class MnistLinearModel(optimizee.Optimizee):
    mnist = None

    def __init__(self, activation='sigmoid', n_batches=128, n_h=20, n_l=1, initial_param_scale=0.1):
        optimizee.Optimizee.__init__(self)
        
        self.activation = activation
        self.n_batches = n_batches
        self.n_l = n_l
        self.n_h = n_h
        self.initial_param_scale = initial_param_scale
        
        if n_l == 0:
            self.x_dim = 784 * 10 + 10
        else:
            self.x_dim = 784 * n_h + n_h + (n_h * n_h + n_h) * (n_l - 1) + n_h * 10 + 10

    def build(self):
        if MnistLinearModel.mnist == None:
            MnistLinearModel.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.mnist = MnistLinearModel.mnist

        self.x = tf.placeholder(tf.float32, [None, None, 784])
        self.y_ = tf.placeholder(tf.float32, [None, None, 10])

    def get_x_dim(self):
        return self.x_dim
    def get_initial_x(self):
        return np.random.normal(size=[self.x_dim], scale=self.initial_param_scale)
    def next_internal_feed_dict(self):
        return {}
    def next_feed_dict(self, n_iterations):
        x_data = np.zeros([n_iterations, self.n_batches, 784])
        y_data = np.zeros([n_iterations, self.n_batches, 10])
        for i in range(n_iterations):
            x_data[i], y_data[i] = self.mnist.train.next_batch(self.n_batches)
        return { self.x: x_data, self.y_: y_data }
    def loss(self, i, x):
        self.start_get_weights(x)

        if self.n_l > 0:
            w1 = self.get_weights([784, self.n_h])
            b1 = self.get_weights([self.n_h])
            w2 = self.get_weights([self.n_h, 10])
            b2 = self.get_weights([10])

            wl = [self.get_weights([self.n_h, self.n_h]) for k in range(self.n_l - 1)]
            bl = [self.get_weights([self.n_h]) for k in range(self.n_l - 1)]

            def act(x):
                if self.activation == 'sigmoid':
                    return tf.sigmoid(x)
                elif self.activation == 'relu':
                    return tf.nn.relu(x)
                elif self.activation == 'elu':
                    return tf.nn.elu(x)
                elif self.activation == 'tanh':
                    return tf.tanh(x)

            last = act(tf.matmul(self.x[i], w1) + b1)

            for k in range(self.n_l - 1):
                last = act(tf.matmul(last, wl[k]) + bl[k])

            last = tf.matmul(last, w2) + b2
        else:
            w = self.get_weights([784, 10])
            b = self.get_weights([10])
            last = tf.matmul(self.x[i], w) + b

        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(last, self.y_[i]))
