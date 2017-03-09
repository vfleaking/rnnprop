import tensorflow as tf
from tensorflow.python.ops import gradients
from tensorflow.python.ops.rnn_cell import LSTMCell, MultiRNNCell
import numpy as np
import nn_opt

class RNNpropModel(nn_opt.BasicNNOptModel):
    def _build_pre(self):
        self.dimA = 20
        self.cellA = MultiRNNCell([LSTMCell(self.dimA)] * 2)
        self.b1 = 0.95
        self.b2 = 0.95
        self.lr = 0.1
        self.eps = 1e-8
    def _build_input(self):
        self.x = self.ph([None])
        self.m = self.ph([None])
        self.v = self.ph([None])
        self.b1t = self.ph([])
        self.b2t = self.ph([])
        self.sid = self.ph([])
        self.cellA_state = tuple((self.ph([None, size.c]), self.ph([None, size.h])) for size in self.cellA.state_size)
        self.input_state = [self.sid, self.b1t, self.b2t, self.x, self.m, self.v, self.cellA_state]
    def _build_initial(self):
        x = self.x
        m = tf.zeros(shape=tf.shape(x))
        v = tf.zeros(shape=tf.shape(x))
        b1t = tf.ones([])
        b2t = tf.ones([])
        cellA_state = self.cellA.zero_state(tf.size(x), tf.float32)
        self.initial_state = [tf.zeros([]), b1t, b2t, x, m, v, cellA_state]
    
    # return state, fx
    def _iter(self, f, i, state):
        sid, b1t, b2t, x, m, v, cellA_state = state

        fx, grad = self._get_fx(f, i, x)
        grad = tf.stop_gradient(grad)

        m = self.b1 * m + (1 - self.b1) * grad
        v = self.b2 * v + (1 - self.b2) * (grad ** 2)

        b1t *= self.b1
        b2t *= self.b2

        sv = tf.sqrt(v / (1 - b2t)) + self.eps
        
        last = tf.pack([grad / sv, (m / (1 - b1t)) / sv], 1)
        last = tf.nn.elu(self.fc(last, 20))
        
        with tf.variable_scope("cellA"):
            lastA, cellA_state = self.cellA(last, cellA_state)
        with tf.variable_scope("fc_A"):
            a = self.fc(lastA, 1)[:,0]

        a = tf.tanh(a) * self.lr
        x -= a

        return [sid + 1, b1t, b2t, x, m, v, cellA_state], fx
