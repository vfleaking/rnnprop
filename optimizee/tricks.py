import tensorflow as tf
import numpy as np
import optimizee

class Mixed(optimizee.Optimizee):
    def __init__(self, opt_dict):
        optimizee.Optimizee.__init__(self)
        self.opt_dict = opt_dict
        self.x_dim = sum(opt.get_x_dim() for opt in self.opt_dict)

    def build(self):
        for opt in self.opt_dict:
            opt.build()
    def get_x_dim(self):
        return self.x_dim
    def get_initial_x(self):
        return np.concatenate([opt.get_initial_x() for opt in self.opt_dict])
    def next_internal_feed_dict(self):
        feed_dict = {}
        for opt in self.opt_dict:
            feed_dict.update(opt.next_internal_feed_dict())
        return feed_dict
    def next_feed_dict(self, n_iterations):
        feed_dict = {}
        for opt in self.opt_dict:
            feed_dict.update(opt.next_feed_dict(n_iterations))
        return feed_dict
    def loss(self, i, x):
        self.start_get_weights(x)
        return sum(coe * opt.loss(i, self.get_weights([opt.get_x_dim()])) for opt, coe in self.opt_dict.items())

class ExponentiallyPointwiseRandomScaling(optimizee.Optimizee):
    def __init__(self, opt, r=3.0):
        optimizee.Optimizee.__init__(self)
        self.opt = opt
        self.r = r
        self.x_dim = self.opt.get_x_dim()

    def build(self):
        self.opt.build()
        self.coe = tf.placeholder(tf.float32, [self.x_dim])
    def get_x_dim(self):
        return self.x_dim
    def get_initial_x(self):
        return self.opt.get_initial_x() / self.coe_val
    def next_internal_feed_dict(self):
        self.coe_val = np.exp(np.random.uniform(-self.r, self.r, size=[self.x_dim]))
        feed_dict = { self.coe: self.coe_val }
        feed_dict.update(self.opt.next_internal_feed_dict())
        return feed_dict
    def next_feed_dict(self, n_iterations):
        return self.opt.next_feed_dict(n_iterations)
    def loss(self, i, x):
        return self.opt.loss(i, x * self.coe)
