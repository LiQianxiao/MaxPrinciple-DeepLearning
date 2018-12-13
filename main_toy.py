"""
File: main_toy.py
Project: MaxPrinciple-DeepLearning
File Created: Thursday, 13th December 2018 10:08:04 pm
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
-----
Copyright - 2018 Qianxiao Li, IHPC, A*STAR
License: MIT License
"""

import tensorflow as tf
import numpy as np
from msalib import layers
from msalib import network
from msalib import train
from msalib.utils import Dataset
import yaml


def target(x):
    """Target function

    Arguments:
        x {numpy array} -- input

    Returns:
        numpy array -- output
    """

    return np.sin(x)


def loss_func(output, label):
    """Loss function (l2)

    Arguments:
        output {tf tensor} -- output from network
        label {tf tensor} -- labels

    Returns:
        tf tensor -- loss
    """

    output = tf.reduce_sum(output, axis=1, keepdims=True)
    return tf.losses.mean_squared_error(
        predictions=output, labels=label)


def generate_datasets(config):
    """Generate synthetic dataset

    Arguments:
        config {dict} -- read from config.yml

    Returns:
        tuple of Dataset objects -- train/test sets
    """

    np.random.seed(config['seed'])
    num_train, num_test = config['num_train'], config['num_test']
    x_train = np.random.uniform(-np.pi, np.pi, (num_train, 1))
    x_test = np.random.uniform(-np.pi, np.pi, (num_test, 1))
    y_train, y_test = target(x_train), target(x_test)
    x_train = np.repeat(x_train, axis=1, repeats=config['num_nodes'])
    x_test = np.repeat(x_test, axis=1, repeats=config['num_nodes'])
    trainset = Dataset(data=x_train, target=y_train)
    testset = Dataset(data=x_test, target=y_test)
    return trainset, testset


def run_toy(config):
    """Run toy problem

    Arguments:
        config {dict} -- read from config.yml
    """

    print('='*100)
    print('Run toy model')
    print('='*100)
    print('Config:')
    for key, value in config.items():
        print('{:20} ({})'.format(key, value))
    print('='*100)

    tf.logging.set_verbosity(
        getattr(tf.logging, config['verbosity']))
    tf.set_random_seed(config['seed'])

    # Generate data
    trainset, testset = generate_datasets(config)

    # Build network
    input = tf.placeholder(
        tf.float32, [None, config['num_nodes']], name='input')
    output = tf.placeholder(tf.float32, [None, 1], name='output')
    net = network.Network(name='msa_net')

    for n in range(config['num_layers']):
        if n == 0:
            net.add(layers.ResidualDense(
                input_shape=(input.shape[1:]),
                units=config['num_nodes'], activation=config['activation'],
                msa_rho=config['rho'], msa_reg=config['reg'],
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=config['kernel_init']),
                bias_initializer=tf.constant_initializer(config['bias_init']),
                delta=config['delta'], name='residualdense_{}'.format(n)))
        else:
            net.add(layers.ResidualDense(
                units=config['num_nodes'], activation=config['activation'],
                msa_rho=config['rho'], msa_reg=config['reg'],
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=config['kernel_init']),
                bias_initializer=tf.constant_initializer(config['bias_init']),
                delta=config['delta'], name='residualdense_{}'.format(n)))

    net.msa_compute_x(input)
    net.msa_compute_p(output, loss_func)
    net.summary()
    sess = tf.Session()

    # MSA trainer
    msa_trainer = train.MSATrainer(
        network=net,
        name='MSA_trainer',
        maxiter=config['msa_maxiter'],
        perturb_init=config['msa_perturb_init'])
    msa_trainer.initialize(sess)
    msa_trainer.train(
        session=sess,
        trainset=trainset,
        testset=testset,
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        buffer_size=config['buffer_size'],
        print_step=config['print_step'])

    # SGD trainer
    sgd_trainer = train.BPTrainer(
        network=net, name='SGD_trainer',
        method='GradientDescentOptimizer',
        args={'learning_rate': config['sgd_lr']})
    sgd_trainer.initialize(sess)
    sgd_trainer.train(
        session=sess,
        trainset=trainset,
        testset=testset,
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        buffer_size=config['buffer_size'],
        print_step=config['print_step'])

    # Adagrad trainer
    adagrad_trainer = train.BPTrainer(
        network=net, name='Adagrad_trainer',
        method='AdagradOptimizer',
        args={'learning_rate': config['adagrad_lr']})
    adagrad_trainer.initialize(sess)
    adagrad_trainer.train(
        session=sess,
        trainset=trainset,
        testset=testset,
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        buffer_size=config['buffer_size'],
        print_step=config['print_step'])

    # Adam trainer
    Adam_trainer = train.BPTrainer(
        network=net, name='Adam_trainer',
        method='AdamOptimizer',
        args={'learning_rate': config['adam_lr']})
    Adam_trainer.initialize(sess)
    Adam_trainer.train(
        session=sess,
        trainset=trainset,
        testset=testset,
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        buffer_size=config['buffer_size'],
        print_step=config['print_step'])


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yml"))
    run_toy(config['toy'])
