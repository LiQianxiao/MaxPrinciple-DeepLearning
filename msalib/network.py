"""
File: network.py
Project: msalib
File Created: Thursday, 13th December 2018 10:08:17 pm
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
-----
Copyright - 2018 Qianxiao Li, IHPC, A*STAR
License: MIT License
-----
Notes:
    Wrappers over tf.keras.layers for MSA
    type of algorithms
"""

import tensorflow as tf


class Network(tf.keras.Sequential):
    """Network class combining MSA layers
    """

    def msa_compute_x(self, input):
        """Compute the states

        Arguments:
            input {tf tensor} -- initial state
        """

        super().apply(input)
        self.msa_xs = [l.input for l in self.layers] + [self.output, ]

    def _msa_add_terminal_loss(self, label, loss_func):
        self.label = label
        self.msa_terminal_loss = loss_func(
            self.output, self.label)

    def msa_compute_p(self, label, loss_func):
        """Compute the co-states

        Arguments:
            label {tf tensor} -- labels
            loss_func {function returning tf tensor} -- loss function
        """

        self._msa_add_terminal_loss(label, loss_func)
        p = - tf.gradients(self.msa_terminal_loss, self.output)[0]
        self.msa_ps = [p]
        for layer in reversed(self.layers):
            p = layer.msa_backward(layer.input, p)
            self.msa_ps.append(p)
        self.msa_ps.reverse()
