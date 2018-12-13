"""
File: layers.py
Project: msalib
File Created: Thursday, 13th December 2018 10:08:17 pm
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
-----
Copyright - 2018 Qianxiao Li, IHPC, A*STAR
License: MIT License
-----
Notes:
    Wrappers for Maximum Principles based algorithms
    ! tf.keras.layers doesnt not work with L-BFGS-B, which
    ! throws line search error, so we use tf.layers as base
"""

import tensorflow as tf
Base = tf.layers


class MSALayer(Base.Layer):
    """Parent class for Layers used for MSA-based algorithms
            All additional signatures and attributes will
            start with msa_ (i.e. msa_rho) to avoid any
            potential inheritance issues
    """
    def __init__(self, *args, msa_rho=0.1, msa_reg=0.1,
                 msa_trainable=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.msa_rho = tf.placeholder_with_default(
            msa_rho, [], 'msa_rho')
        tf.add_to_collection('msa_rho', self.msa_rho)
        self.msa_reg = msa_reg
        self.msa_trainable = msa_trainable

    def msa_regularizer(self):
        if self.variables:
            return self.msa_reg * tf.add_n([
                tf.nn.l2_loss(v) for v in self.variables])
        else:
            return 0.0

    def msa_hamiltonian(self, x, p):
        return tf.reduce_sum(p * self.apply(x))

    def msa_backward(self, x, p):
        """Compute p_{n}

        Arguments:
            x {tf tensor} -- x_{n}
            p {tf tensor} -- p_{n+1}

        Returns:
            tf tensor -- p_{n}
        """

        x = tf.stop_gradient(x)
        p = tf.stop_gradient(p)
        H = self.msa_hamiltonian(x, p)
        return tf.gradients(H, x)[0]

    def msa_minus_H_aug(self, x, y, p, q):
        """Computes minus of augmented Hamiltonian
            Note that since p,q are reversed in order,
            (p-q)/dt ~ - dp/dt. This causes some sign reversions

        Arguments:
            x {tf tensor} -- x_{n}
            y {tf tensor} -- x_{n+1}
            p {tf tensor} -- p_{n+1}
            q {tf tensor} -- p_{n}

        Returns:
            tf tensor -- minus of augmented H
        """

        x, y, p, q = [tf.stop_gradient(t) for t in [x, y, p, q]]
        dHdp = self.apply(x)
        H = tf.reduce_sum(p * dHdp) - self.msa_regularizer()
        dHdx = tf.gradients(H, x)[0]
        x_feasibility = self.msa_rho * tf.nn.l2_loss(y - dHdp)
        p_feasibility = self.msa_rho * tf.nn.l2_loss(q - dHdx)
        return - H + x_feasibility + p_feasibility


class Dense(Base.Dense, MSALayer):
    pass


class ResidualDense(Dense):
    """Residual connection dense layer
    """

    def __init__(self, *args, delta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.msa_delta = delta

    def call(self, inputs):
        return inputs + self.msa_delta * super().call(inputs)


class Conv2D(Base.Conv2D, MSALayer):
    pass


class ResidualConv2D(Conv2D):
    """Residual connection conv2d layer
        This is slightly different from the usual ResNet
        in that there is only 1 conv layer, instead of 2
    """

    def __init__(self, *args, delta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.msa_delta = delta

    def call(self, inputs):
        return inputs + self.msa_delta * super().call(inputs)


class Lower(MSALayer):
    """Lower dimension layer
    """

    def __init__(self, *args, lower_axis=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.lower_axis = lower_axis
        self.msa_trainable = False

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.lower_axis, keepdims=True)


class Flatten(Base.Flatten, MSALayer):
    """Flatten layer
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msa_trainable = False


class AveragePooling2D(Base.AveragePooling2D, MSALayer):
    """Average pooling layer
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msa_trainable = False
