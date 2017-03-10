'''The model, tricks and optimizee used when training an RNN optimizer.'''

import nn_opt
import optimizee
import test_list

tasks = {
    'rnnprop': {
        'model': nn_opt.rnn.RNNpropModel,
        'optimizee': {
            'train': optimizee.tricks.Mixed({
                optimizee.tricks.ExponentiallyPointwiseRandomScaling(optimizee.mnist.MnistLinearModel()): 1.0,
                optimizee.tricks.ExponentiallyPointwiseRandomScaling(optimizee.trivial.Square(x_dim=20), 1.0): 1.0
            }),
            'tests': test_list.tests
        },
        'use_avg_loss': False,
    },
    'deepmind-lstm-avg': {
        'model': nn_opt.deepmind.LSTMOptModel,
        'optimizee': {
            'train': optimizee.mnist.MnistLinearModel(),
            'tests': test_list.tests
        },
        'use_avg_loss': True
    },
}  
