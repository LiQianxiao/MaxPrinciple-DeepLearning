"""
File: main_mnist.py
Project: MaxPrinciple-DeepLearning
File Created: Thursday, 13th December 2018 10:08:04 pm
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
-----
Copyright - 2018 Qianxiao Li, IHPC, A*STAR
License: MIT License
"""

import tensorflow as tf
from msalib import layers
from msalib import network
from msalib import train
from msalib.utils import load_dataset
import yaml


def loss_func(output, label):
    """Loss function (softmax cross entropy)

    Arguments:
        output {tf tensor} -- output from network
        label {tf tensor} -- labels

    Returns:
        tf tensor -- loss
    """

    return tf.losses.softmax_cross_entropy(
        logits=output, onehot_labels=label)


def run_mnist(config):
    """Run mnist/fashion_mnist classification

    Arguments:
        config {dict} -- read from config.yml
    """

    print('='*100)
    print('Run CNN on MNIST')
    print('='*100)
    print('Config:')
    for key, value in config.items():
        print('{:20} ({})'.format(key, value))
    print('='*100)

    tf.logging.set_verbosity(
        getattr(tf.logging, config['verbosity']))
    tf.set_random_seed(config['seed'])

    # Generate data
    trainset, testset = load_dataset(
        name=config['dataset_name'],
        num_train=config['num_train'],
        num_test=config['num_test'],
        lift_dim=config['lift_dimension']
    )

    # Build network
    input = tf.placeholder(
        tf.float32, [None, 28, 28, config['lift_dimension']], name='input')
    output = tf.placeholder(tf.float32, [None, 10], name='output')
    net = network.Network(name='msa_net')

    net.add(layers.Conv2D(
        input_shape=input.shape[1:],
        filters=config['num_channels'],
        kernel_size=config['filter_size'],
        padding=config['padding'],
        activation=config['activation'],
        msa_rho=config['rho'],
        msa_reg=config['reg'],
        name='conv2d_0')
    )

    net.add(layers.AveragePooling2D(
        pool_size=2, strides=2, padding=config['padding'], name='avg_pool_0')
    )

    net.add(layers.Conv2D(
        input_shape=input.shape[1:],
        filters=config['num_channels'],
        kernel_size=config['filter_size'],
        padding=config['padding'],
        activation=config['activation'],
        msa_rho=config['rho'],
        msa_reg=config['reg'],
        name='conv2d_init')
    )

    net.add(layers.AveragePooling2D(
        pool_size=2, strides=2, padding=config['padding'], name='avg_pool_1')
    )

    for n in range(config['num_layers']):
        net.add(layers.ResidualConv2D(
            filters=config['num_channels'],
            kernel_size=config['filter_size'],
            padding=config['padding'],
            activation=config['activation'],
            msa_rho=config['rho'],
            msa_reg=config['reg'],
            delta=config['delta'],
            name='residualconv2d_{}'.format(n)))

    net.add(layers.Lower(name='lower'))
    net.add(layers.Flatten(name='flatten'))
    net.add(layers.Dense(
        units=10, msa_rho=config['rho'], msa_reg=config['reg'], name='dense'))

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
    run_mnist(config['mnist'])
