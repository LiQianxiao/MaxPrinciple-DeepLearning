"""
File: train.py
Project: msalib
File Created: Thursday, 13th December 2018 10:08:17 pm
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
-----
Copyright - 2018 Qianxiao Li, IHPC, A*STAR
License: MIT License
"""

import tensorflow as tf
import math
from msalib.scipyoptimizer import ScipyOptimizer


class Trainer(object):
    """Abstract trainer object
    """

    def __init__(self, network, name, **kwargs):
        """Abstract trainer object

        Arguments:
            network {Network object} -- network
            name {string} -- name
        """

        self.network = network
        self.name = name
        self.kwargs = kwargs
        self._add_loss()
        self._add_train_ops()

    def initialize(self, session):
        session.run(tf.global_variables_initializer())

    def _add_loss(self):
        self.loss = self.network.msa_terminal_loss + \
            tf.add_n([l.msa_regularizer() for l in self.network.layers])

    def add_train_ops(self):
        raise NotImplementedError

    def _compute_loss(self, session, dataset, buffer_size):
        # TODO: Iterators with tf.dataset API
        loss = 0
        num_steps = math.ceil(dataset.num_examples / buffer_size)
        for step in range(num_steps):
            input_batch, label_batch = dataset.next_batch(buffer_size)
            feed_dict = {
                self.network.input: input_batch,
                self.network.label: label_batch,
            }
            loss += session.run(self.loss, feed_dict)
        return loss / num_steps

    def _train_step(self, session, dataset, batch_size):
        raise NotImplementedError

    def _train_epoch(self, session, trainset, testset, batch_size,
                     buffer_size, print_step=False):
        num_steps = math.ceil(trainset.num_examples / batch_size)
        for step in range(num_steps):
            self._train_step(session, trainset, batch_size)
            if print_step and step % math.ceil(num_steps / print_step) == 0:
                train_loss = self._compute_loss(session, trainset, buffer_size)
                test_loss = self._compute_loss(session, trainset, buffer_size)
                print('Step {:5} of {:5}: '.format(step, num_steps))
                print('Train loss: {}'.format(train_loss))
                print('Test loss: {}'.format(test_loss))

    def train(self, session, trainset, testset, batch_size,
              num_epochs, buffer_size=500, print_step=False):
        """Train epoch

        Arguments:
            session {tf session} -- session
            trainset {Dataset object} -- training set
            testset {Dataset object} -- testing set
            batch_size {int} -- batch size
            num_epochs {int} -- number of epochs to train

        Keyword Arguments:
            buffer_size {int} -- buffer size for feed_dict (default: {500})
            print_step {bool} -- print step losses (default: {False})
        """

        print('='*100)
        print('Trainer: {} ({})'.format(self.__class__.__name__, self.name))
        print('Settings: {}'.format(getattr(self, 'kwargs', 'none')))
        print('='*100)
        print('Epoch: init')
        print('Train loss: {}'.format(
            self._compute_loss(session, trainset, buffer_size)))
        print('Test loss: {}'.format(
            self._compute_loss(session, testset, buffer_size)))

        for epoch in range(num_epochs):
            self._train_epoch(
                session, trainset, testset,
                batch_size, buffer_size, print_step)

            print('Epoch: {}'.format(epoch))
            print('Train loss: {}'.format(
                self._compute_loss(session, trainset, buffer_size)))
            print('Test loss: {}'.format(
                self._compute_loss(session, testset, buffer_size)))


class MSATrainer(Trainer):
    """Implementation of the E-MSA algorithm
    """

    def _get_placeholders(self, list_of_tensors, prefix):
        return [
            tf.placeholder(t.dtype, t.shape, prefix+'_ph_{}'.format(i))
            for i, t in enumerate(list_of_tensors)
        ]

    def _add_train_ops(self):
        self.ph_xs = self._get_placeholders(
            list_of_tensors=self.network.msa_xs[:-1],
            prefix='xs')
        self.ph_ys = self._get_placeholders(
            list_of_tensors=self.network.msa_xs[1:],
            prefix='ys')
        self.ph_ps = self._get_placeholders(
            list_of_tensors=self.network.msa_ps[1:],
            prefix='ps')
        self.ph_qs = self._get_placeholders(
            list_of_tensors=self.network.msa_ps[:-1],
            prefix='qs')

        self.objectives, self.optimizers = [], []
        for n, layer in enumerate(self.network.layers):
            if layer.msa_trainable:
                objective = layer.msa_minus_H_aug(
                        self.ph_xs[n], self.ph_ys[n],
                        self.ph_ps[n], self.ph_qs[n])
                optimizer = ScipyOptimizer(
                    objective, var_list=layer.variables,
                    perturb_init=self.kwargs['perturb_init'],
                    method='L-BFGS-B',
                    options={"maxiter": self.kwargs['maxiter']})
            else:
                objective = None
                optimizer = None
            self.objectives.append(objective)
            self.optimizers.append(optimizer)

    def _train_step(self, session, dataset, batch_size):
        # TODO: (1) Without PHs (2) Parallelize
        # TODO: (3) BFGS on GPU (4) Variable rho
        input_batch, label_batch = dataset.next_batch(batch_size)
        feed_dict = {
            self.network.input: input_batch,
            self.network.label: label_batch,
        }
        xval, pval = session.run(
            [self.network.msa_xs, self.network.msa_ps],
            feed_dict
        )
        for n, opt in enumerate(self.optimizers):
            if opt is not None:
                opt.minimize(
                    session, {
                        self.ph_xs[n]: xval[n],
                        self.ph_ys[n]: xval[n+1],
                        self.ph_ps[n]: pval[n+1],
                        self.ph_qs[n]: pval[n]
                    }
                )


class BPTrainer(Trainer):
    """Trainer for the usual GD-back-prop algorithms
    """

    def _add_train_ops(self):
        opt = getattr(tf.train, self.kwargs['method'])(**self.kwargs['args'])
        self.train_op = opt.minimize(self.loss)

    def _train_step(self, session, dataset, batch_size):
        input_batch, label_batch = dataset.next_batch(batch_size)
        feed_dict = {
            self.network.input: input_batch,
            self.network.label: label_batch,
        }
        session.run(self.train_op, feed_dict)
