import tensorflow as tf
import numpy as np
import optimizee
from tensorflow.contrib.learn.python.learn.datasets.mnist import dense_to_one_hot
from tensorflow.examples.tutorials.mnist import input_data
import os
import re
import sys
import tarfile
import pickle
from six.moves import urllib

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

class VGGModel(optimizee.Optimizee):
    '''VGG-like CNNs on dataset MNIST or CIFAR10.'''
    mnist_dataset = None 
    cifar_dataset = None 
    def __init__(self, input_data='mnist', n_batches=128, fc_num=1, conv_num=2, pool_num=1, add_dropout=False, initial_param_scale=0.1):
        assert conv_num % pool_num == 0
        optimizee.Optimizee.__init__(self)
        self.n_batches = n_batches
        self.input_data = input_data
        self.add_dropout = add_dropout
        self.fc_num = fc_num
        self.conv_num = conv_num
        self.pool_num = pool_num
        self.initial_param_scale = initial_param_scale
        if self.input_data == 'cifar10':
            self.n_classes = 10
            self.input_size = 32
            self.input_channel = 3
        if self.input_data == 'mnist':
            self.n_classes = 10
            self.input_size = 28
            self.input_channel = 1
        assert self.input_size % (2**self.pool_num) == 0
        self.x_dim = 0

        for j in range(self.pool_num):
            self.x_dim += self.get_n([3, 3, self.input_channel if j == 0 else 2**(j+3), 2**(j+4)])
            self.x_dim += self.get_n([2**(j+4)])
            for k in range(self.conv_num / self.pool_num - 1):
                self.x_dim += self.get_n([3, 3, 2**(j+4), 2**(j+4)])
                self.x_dim += self.get_n([2**(j+4)])
        for j in range(self.fc_num):
            self.x_dim += self.get_n([(self.input_size * self.input_size * 2**3 / 2**self.pool_num) if j == 0 else 2**(self.pool_num+4), self.n_classes if j == (self.fc_num-1) else 2**(self.pool_num+4)])
            self.x_dim += self.get_n([self.n_classes if j == (self.fc_num-1) else 2**(self.pool_num+4)])

    def build(self):
        if self.input_data == 'mnist':
            if VGGModel.mnist_dataset == None:
                VGGModel.mnist_dataset = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False).train
            self.dataset = VGGModel.mnist_dataset
        if self.input_data == 'cifar10':
            if VGGModel.cifar_dataset == None:
                VGGModel.cifar_dataset = cifar("Cifar10_data/", one_hot=True)
            self.dataset = VGGModel.cifar_dataset
        self.x = tf.placeholder(tf.float32, [None, self.n_batches, self.input_size, self.input_size, self.input_channel])
        self.y_ = tf.placeholder(tf.float32, [None, self.n_batches, self.n_classes])

    def get_x_dim(self):
        return self.x_dim
    
    def get_initial_x(self):
        return np.random.normal(size=[self.x_dim], scale=self.initial_param_scale)
    
    def next_internal_feed_dict(self):
        return {}
    
    def next_feed_dict(self, n_iterations):
        x_data = np.zeros([n_iterations, self.n_batches, self.input_size, self.input_size, self.input_channel])
        y_data = np.zeros([n_iterations, self.n_batches, self.n_classes])
        for i in range(n_iterations):
            x_data[i], y_data[i] = self.dataset.next_batch(self.n_batches)
        return { self.x: x_data, self.y_: y_data }
    
    def loss(self, i, x):
        self.start_get_weights(x)
        
        conv_f = []
        conv_b = []
        for j in range(self.pool_num):
            f = [self.get_weights([3, 3, self.input_channel if j == 0 else 2**(j+3), 2**(j+4)])]
            b = [self.get_weights([2**(j+4)])]
            for k in range(self.conv_num / self.pool_num - 1):
                f.append(self.get_weights([3, 3, 2**(j+4), 2**(j+4)]))
                b.append(self.get_weights([2**(j+4)]))
            conv_f.append(f)
            conv_b.append(b)
        
        fc_w = []
        fc_b = []
        for j in range(self.fc_num):
            fc_w.append(self.get_weights([(self.input_size * self.input_size * 2**3 / 2**self.pool_num) if j == 0 else 2**(self.pool_num+4), self.n_classes if j == (self.fc_num-1) else 2**(self.pool_num+4)]))
            fc_b.append(self.get_weights([self.n_classes if j == (self.fc_num-1) else 2**(self.pool_num+4)]))
        
        last = self.x[i]
        #print last.get_shape
        for j in range(self.pool_num):
            for k in range(self.conv_num / self.pool_num):
                last = self.conv(last, conv_f[j][k], conv_b[j][k])
            last = self.max_pool(last)
        #print last.get_shape
        
        for j in range(self.fc_num - 1):
            last = self.fc(last, fc_w[j], fc_b[j])
            last = tf.nn.relu(last)
            if self.add_dropout:
                last = tf.nn.dropout(last, 0.5)
        last = self.fc(last, fc_w[self.fc_num - 1], fc_b[self.fc_num - 1])
        #print last.get_shape
        
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(last, self.y_[i]))
        return loss

    def conv(self, bottom, f, b):
        last = tf.nn.conv2d(bottom, f, [1, 1, 1, 1], padding='SAME')
        last = tf.nn.bias_add(last, b)
        last = tf.nn.relu(last)
        return last
            
    def fc(self, bottom, w, b):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])
        last = tf.nn.bias_add(tf.matmul(x, w), b)
        return last
            
    def max_pool(self, bottom):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        
class cifar(object):
    """
    A class to help to read data from cifar-10. 
    Only training data included.
    """
    def __init__(self, dirname, one_hot=True):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._one_hot = one_hot
        self._num_classes = 10
        
        self.maybe_download_and_extract(dirname)
        dirname = os.path.join(dirname, 'cifar-10-batches-py/')
        images = []
        labels = []
        for i in range(1, 6):
            fpath = os.path.join(dirname, 'data_batch_' + str(i))
            image, label = self.load_batch(fpath)
            if i == 1:
                images = np.array(image)
                labels = np.array(label)
            else:
                images = np.concatenate([images, image], axis=0)
                labels = np.concatenate([labels, label], axis=0)
        images = np.dstack((images[:, :1024], images[:, 1024:2048], images[:, 2048:]))
        images = np.reshape(images, [-1, 32, 32, 3])
        if self._one_hot:
            labels = dense_to_one_hot(labels, self._num_classes)
        
        print 'Cifar images size:', images.shape
        print 'Cifar labels size:', labels.shape
        self._images = images / 255.0 - 0.5
        self._labels = labels
        self._num_examples = images.shape[0]
        
    
    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
        
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]
    
    
    def maybe_download_and_extract(self, data_dir):
        """Download and extract the tarball from Alex's website."""
        dest_directory = data_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                   float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
        
    
    def load_batch(self, fpath):
        with open(fpath, 'rb') as f:
            if sys.version_info > (3, 0):
                # Python3
                d = pickle.load(f, encoding='latin1')
            else:
                # Python2
                d = pickle.load(f)
        data = d["data"]
        labels = d["labels"]
        return data, labels
