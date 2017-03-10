import tensorflow as tf
import random
import time
import numpy as np
import sys
import pprint
import optimizee
import nn_opt
import math
import os
import task_list
import test_list

def log(s, filename):
    print s
    with open(filename, 'a') as fp:
        print >>fp, s

def randid():
    return ''.join([chr(random.randint(0, 25) + ord('a')) for i in range(6)])

def task_id():
    flags = tf.app.flags.FLAGS
    return flags.task + "-" + flags.id

def train_optimizer(task):
    '''Train an RNN optimizer.
    
    Args:
        task: A dictionary in task_list.py which specifies the model of the optimizer and 
            the tricks and the optimizee to use when training the optimizer.
    '''
    flags = tf.app.flags.FLAGS
    session = tf.get_default_session()

    task['optimizee']['train'].build()

    n_steps = task['n_steps'] if 'n_steps' in task else flags.n_steps
    n_bptt_steps = task['n_bptt_steps'] if 'n_bptt_steps' in task else flags.n_bptt_steps
    use_avg_loss = task['use_avg_loss'] if 'use_avg_loss' in task else False
    options = task['options'] if 'options' in task else {}
    assert n_steps % n_bptt_steps == 0

    model = task['model'](
            name=task_id(),
            optimizee=task['optimizee']['train'],
            n_bptt_steps=n_bptt_steps,
            use_avg_loss=use_avg_loss,
            **options)
    model.prepare_train_optimizee(task['optimizee']['tests'])

    log_filename = model.name + "_data/log.txt" 

    session.run(tf.global_variables_initializer())
    if flags.eid != 0:
        model.restore(flags.eid)
        model.bid = flags.eid * flags.n_batches
    
    avg_loss_value = 0.0
    avg_final_loss_value = 0.0

    eid = flags.eid
    while flags.n_epochs == 0 or eid < flags.n_epochs:
        eid += 1

        loss_values = []
        for i in range(flags.n_batches):
            ret = model.train_one_iteration(n_steps)

            loss_value = ret['loss']

            loss_values.append(loss_value)

            sys.stdout.write("\r\033[K")
            msg = "iteration #%d" % i
            msg += ": loss = %.5f avg loss = %.5f" % (loss_value, np.mean(loss_values))
            sys.stdout.write(msg)
            sys.stdout.flush()

        sys.stdout.write("\r\033[K")
        msg = "epoch #%d" % eid
        msg += ": loss = %.5f" % np.mean(loss_values)
        log(msg, log_filename)
        log(str(loss_values), log_filename)

        if eid % 10 == 0:
            model.save(eid)

        test_loss_values = {}
        for i in range(flags.n_tests):
            for name, avg_loss_value, gd_avg_loss_value in model.test(eid):
                if name not in test_loss_values:
                    test_loss_values[name] = {'nn': [], 'gd': []}
                test_loss_values[name]['nn'].append(avg_loss_value)
                test_loss_values[name]['gd'].append(gd_avg_loss_value)
        for name in test_loss_values:
            log("epoch #%d test %s: loss = %.5f gd_loss = %.5f" % (eid, name,
                    np.mean(test_loss_values[name]['nn']), np.mean(test_loss_values[name]['gd'])), log_filename)

def train_optimizee(task):
    '''Use traditional optimization algorithm to train an optimizee and get more 
        information about the gradient each step and the final optimizee parameters.
    
    Args:
        task: A dictionary in test_list.py which specifies the optimizee to train, 
            the optimization algorithm to use and how many steps to train the optimizee.
    '''
    flags = tf.app.flags.FLAGS
    session = tf.get_default_session()
    if task['frequency'] == 0:
        task['frequency'] = 1

    opt = task['optimizee']

    opt.build()

    all_val_final_loss = []

    x = tf.Variable(np.zeros([opt.x_dim]), dtype=tf.float32)
    loss = opt.loss(0, x)
    gd = task['gd']()
    grad = gd.compute_gradients(loss)
    train_step = gd.apply_gradients(grad)

    for it in range(100):
        internal_feed_dict = opt.next_internal_feed_dict()

        session.run(tf.global_variables_initializer())
        session.run(x.assign(opt.get_initial_x()))
        
        eid = 0
        while flags.n_epochs == 0 or eid < flags.n_epochs:
            eid += 1

            data_dicts = []
            for i in range(task['n_steps']):
                data_dicts.append(opt.next_feed_dict(1))

            loss_values = []
            avg_grad_rms = 0
            for i in range(task['n_steps']):
                feed_dict = internal_feed_dict
                feed_dict.update(data_dicts[i])
                _, loss_value, grad_value = session.run([train_step, loss, grad[0][0]], feed_dict=feed_dict)

                loss_values.append(loss_value)

                sys.stdout.write("\r\033[K")
                sys.stdout.write("iteration #%d: loss = %.5f grad rms: %.5f grad mean: %.5f grad var: %.5f grad min: %.5f grad max: %.5f" % (i, loss_value, math.sqrt(np.mean(grad_value ** 2)), np.mean(grad_value), math.sqrt(np.var(grad_value)), np.min(grad_value), np.max(grad_value)))
                avg_grad_rms += math.sqrt(np.mean(grad_value ** 2))
                sys.stdout.flush()

            avg_grad_rms /= task['n_steps']

            sys.stdout.write("\r\033[K")

            val_final_loss = 0.0
            for i in range(task['n_steps']):
                feed_dict = internal_feed_dict
                feed_dict.update(data_dicts[i])
                val_final_loss += session.run(loss, feed_dict=feed_dict)
            val_final_loss /= task['n_steps']

            val_x = x.eval()
            print "epoch #%d: loss = %.5f" % (eid, val_final_loss)
            print "x mean: %.5f x std_var: %.5f x min: %.5f x max: %.5f grad rms: %.5f" % (np.mean(val_x), math.sqrt(np.var(val_x)), np.min(val_x), np.max(val_x), avg_grad_rms)

        all_val_final_loss.append(val_final_loss)
        print 'mean final loss = %.5f' % np.mean(all_val_final_loss)

def optimizer_train_optimizee(task):
    '''Test a trained RNN optimizer on different optimizees.
    
    Args:
        task: A dictionary in task_list.py which specifies the RNN optimizer.
    '''

    flags = tf.app.flags.FLAGS
    session = tf.get_default_session()

    test_names = [
            'mnist-nn-sigmoid-100',
            'mnist-nn-relu-100',
            'mnist-nn-sigmoid-2000',
            'mnist-nn-sigmoid-10000',
            'vgg-mnist-fc1-conv2-pool1-100',
            'vgg-cifar-fc1-conv2-pool1-100',
            'vgg-mnist-fc2-conv4-pool2-100',
            'vgg-cifar-fc2-conv4-pool2-100',
            'mnist-nn-elu-100',
            'mnist-nn-tanh-100',
            'mnist-nn-l2-sigmoid-100',
            'mnist-nn-l3-sigmoid-100',
            'mnist-nn-l4-sigmoid-100',
            'mnist-nn-l5-sigmoid-100',
            'mnist-nn-l6-sigmoid-100',
            'mnist-nn-l7-sigmoid-100',
            'mnist-nn-l8-sigmoid-100',
            'mnist-nn-l9-sigmoid-100',
            'mnist-nn-l10-sigmoid-100',
            'sin_lstm',
            'sin_lstm-x2',
            'sin_lstm-no001',
    ]
    tests = {}
    for name in test_names:
        tests[name] = task['optimizee']['tests'][name]
        tests[name]['frequency'] = 1

    task['optimizee']['train'].build()

    n_bptt_steps = task['n_bptt_steps'] if 'n_bptt_steps' in task else flags.n_bptt_steps
    use_avg_loss = task['use_avg_loss'] if 'use_avg_loss' in task else False
    options = task['options'] if 'options' in task else {}

    model = task['model'](
            name=task_id(),
            optimizee=task['optimizee']['train'],
            n_bptt_steps=n_bptt_steps,
            use_avg_loss=use_avg_loss,
            **options)
    model.prepare_train_optimizee(tests)

    log_filename = model.name + "_data/log_test.txt" 

    session.run(tf.global_variables_initializer())
    assert flags.eid != 0
    model.restore(flags.eid)
    log("model %s after epoch #%d" % (task_id(), flags.eid  ), log_filename)
    
    eid = 0
    test_loss_values = {}
    while flags.n_epochs == 0 or eid < flags.n_epochs:
        eid += 1
        for i in range(flags.n_tests):
            for name, avg_loss_value, gd_avg_loss_value in model.test(eid):
                if name not in test_loss_values:
                    test_loss_values[name] = {'nn': [], 'gd': []}
                test_loss_values[name]['nn'].append(avg_loss_value)
                test_loss_values[name]['gd'].append(gd_avg_loss_value)
        for name in sorted(test_loss_values):
            log("epoch #%d: test %s: loss = %.5f gd_loss = %.5f" % (eid, name,
                    np.mean(test_loss_values[name]['nn']), np.mean(test_loss_values[name]['gd'])), log_filename)

def main(argv):
    pprint.pprint(tf.app.flags.FLAGS.__flags)

    flags = tf.app.flags.FLAGS        

    graph = tf.Graph()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(flags.gpu)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22, allow_growth=True)
    with graph.as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),graph=graph) as session:
            all_tests = test_list.tests
            tasks = task_list.tasks

            if flags.train == 'optimizer':
                train_optimizer(tasks[flags.task])
            elif flags.train == 'optimizee':
                train_optimizee(all_tests[flags.task])
            elif flags.train == 'optimizer_train_optimizee':
                optimizer_train_optimizee(tasks[flags.task])
            elif flags.train == 'test':
                test(tasks[flags.task])


if __name__ == '__main__':
    tf.app.flags.DEFINE_string("task", "temp", "the name of the task")
    tf.app.flags.DEFINE_string("id", randid(), "id")
    tf.app.flags.DEFINE_integer("gpu", 0, "gpu id")
    tf.app.flags.DEFINE_string("train", "optimizer", "optimizer, optimizee, or optimizer_train_optimizee")
    tf.app.flags.DEFINE_integer("eid", 0, "the epoch id to continue training")
    tf.app.flags.DEFINE_integer("n_steps", 100, "the number of iterations in training RNN optimizer")
    tf.app.flags.DEFINE_integer("n_bptt_steps", 20, "the number of iterations in training RNN optimizer")
    tf.app.flags.DEFINE_integer("n_batches", 100, "#batches per epoch")
    tf.app.flags.DEFINE_integer("n_tests", 1, "#batches per epoch at the test stage")
    tf.app.flags.DEFINE_integer("n_epochs", 0, "#epochs")
    tf.app.run()
