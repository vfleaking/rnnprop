'''All optimizees used in the experiments.'''

import tensorflow as tf
import optimizee

tests = {
    'mnist-nn-sigmoid-100': {
        'frequency': 1,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='sigmoid'),
        'gd': lambda: tf.train.AdamOptimizer(0.0312500),
        'n_steps': 100
    },
    'mnist-nn-sigmoid-2000': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='sigmoid'),
        'gd': lambda: tf.train.AdamOptimizer(0.0156250),
        'n_steps': 2000
    },
    'mnist-nn-sigmoid-10000': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='sigmoid'),
        'gd': lambda: tf.train.AdamOptimizer(0.0078125),
        'n_steps': 10000
    },
    'mnist-nn-relu-100': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='relu'),
        'gd': lambda: tf.train.AdamOptimizer(0.0220971),
        'n_steps': 100
    },
    'mnist-nn-elu-100': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='elu'),
        'gd': lambda: tf.train.AdamOptimizer(0.0220971),
        'n_steps': 100
    },
    'mnist-nn-tanh-100': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='tanh'),
        'gd': lambda: tf.train.AdamOptimizer(0.0110485),
        'n_steps': 100
    },
    'mnist-nn-l2-sigmoid-100': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='sigmoid', n_l=2),
        'gd': lambda: tf.train.AdamOptimizer(0.0312500),
        'n_steps': 100
    },
    'mnist-nn-l3-sigmoid-100': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='sigmoid', n_l=3),
        'gd': lambda: tf.train.AdamOptimizer(0.0312500),
        'n_steps': 100
    },
    'mnist-nn-l4-sigmoid-100': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='sigmoid', n_l=4),
        'gd': lambda: tf.train.AdamOptimizer(0.0156250),
        'n_steps': 100
    },
    'mnist-nn-l5-sigmoid-100': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='sigmoid', n_l=5),
        'gd': lambda: tf.train.AdamOptimizer(0.0156250),
        'n_steps': 100
    },
    'mnist-nn-l6-sigmoid-100': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='sigmoid', n_l=6),
        'gd': lambda: tf.train.AdamOptimizer(0.0156250),
        'n_steps': 100
    },
    'mnist-nn-l7-sigmoid-100': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='sigmoid', n_l=7),
        'gd': lambda: tf.train.AdamOptimizer(0.0110485),
        'n_steps': 100
    },
    'mnist-nn-l8-sigmoid-100': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='sigmoid', n_l=8),
        'gd': lambda: tf.train.AdamOptimizer(0.0004883),
        'n_steps': 100
    },
    'mnist-nn-l9-sigmoid-100': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='sigmoid', n_l=9),
        'gd': lambda: tf.train.AdamOptimizer(0.0006905),
        'n_steps': 100
    },
    'mnist-nn-l10-sigmoid-100': {
        'frequency': 0,
        'optimizee': optimizee.mnist.MnistLinearModel(activation='sigmoid', n_l=10),
        'gd': lambda: tf.train.AdamOptimizer(0.0006905),
        'n_steps': 100
    },
    'vgg-mnist-fc1-conv2-pool1-100': {
        'frequency': 0,
        'optimizee': optimizee.vgg.VGGModel(input_data='mnist', n_batches=128, fc_num=1, conv_num=2, pool_num=1),
        'gd': lambda: tf.train.AdamOptimizer(0.0156250),
        'n_steps': 100
    },
    'vgg-cifar-fc1-conv2-pool1-100': {
        'frequency': 0,
        'optimizee': optimizee.vgg.VGGModel(input_data='cifar10', n_batches=128, fc_num=1, conv_num=2, pool_num=1),
        'gd': lambda: tf.train.AdamOptimizer(0.0078125),
        'n_steps': 100
    },
    'vgg-mnist-fc2-conv4-pool2-100': {
        'frequency': 0,
        'optimizee': optimizee.vgg.VGGModel(input_data='mnist', n_batches=128, fc_num=2, conv_num=4, pool_num=2),
        'gd': lambda: tf.train.AdamOptimizer(0.0055243),
        'n_steps': 100
    },
    'vgg-cifar-fc2-conv4-pool2-100': {
        'frequency': 0,
        'optimizee': optimizee.vgg.VGGModel(input_data='cifar10', n_batches=128, fc_num=2, conv_num=4, pool_num=2),
        'gd': lambda: tf.train.AdamOptimizer(0.0039062),
        'n_steps': 100
    },
    'sin_lstm': {
        'frequency': 0,
        'optimizee': optimizee.lstm.SinLSTMModel(),
        'gd': lambda: tf.train.AdagradOptimizer(0.5000000),
        'n_steps': 100
    },
    'sin_lstm-x2': {
        'frequency': 0,
        'optimizee': optimizee.lstm.SinLSTMModel(n_lstm=2),
        'gd': lambda: tf.train.AdamOptimizer(0.0220971),
        'n_steps': 100
    },
    'sin_lstm-no001': {
        'frequency': 0,
        'optimizee': optimizee.lstm.SinLSTMModel(noise_scale=0.01),
        'gd': lambda: tf.train.AdamOptimizer(0.0312500),
        'n_steps': 100
    }
}
