"""
File: scipyoptimizer.py
Project: msalib
File Created: Thursday, 13th December 2018 10:08:17 pm
Author: Qianxiao Li (liqix@ihpc.a-star.edu.sg)
-----
Copyright - 2018 Qianxiao Li, IHPC, A*STAR
License: MIT License
"""

import tensorflow as tf
import numpy as np


Base = tf.contrib.opt.ScipyOptimizerInterface


class ScipyOptimizer(Base):
    """Scipy optimizer interface with option to
       randomize initial condition
    """

    def __init__(self, *args, perturb_init=None, **kwargs):
        self.perturb_init = perturb_init
        super().__init__(*args, **kwargs)

    def minimize(self, session=None, feed_dict=None, fetches=None,
                 step_callback=None, loss_callback=None, **run_kwargs):
        """Minimize a scalar `Tensor`.
        Variables subject to optimization are updated in-place at the end of
        optimization.
        Note that this method does *not* just return a minimization `Op`, unlike
        `Optimizer.minimize()`; instead it actually performs minimization by
        executing commands to control a `Session`.
        Args:
        session: A `Session` instance.
        feed_dict: A feed dict to be passed to calls to `session.run`.
        fetches: A list of `Tensor`s to fetch and supply to `loss_callback`
            as positional arguments.
        step_callback: A function to be called at each optimization step;
            arguments are the current values of all optimization variables
            flattened into a single vector.
        loss_callback: A function to be called every time the loss and gradients
            are computed, with evaluated fetches supplied as positional arguments.
        **run_kwargs: kwargs to pass to `session.run`.
        """
        session = session or tf.get_default_session()
        feed_dict = feed_dict or {}
        fetches = fetches or []

        loss_callback = loss_callback or (lambda *fetches: None)
        step_callback = step_callback or (lambda xk: None)

        # Construct loss function and associated gradient.
        loss_grad_func = self._make_eval_func(
            [self._loss, self._packed_loss_grad],
            session, feed_dict, fetches, loss_callback)

        # Construct equality constraint functions and associated gradients.
        equality_funcs = self._make_eval_funcs(
            self._equalities, session, feed_dict, fetches)
        equality_grad_funcs = self._make_eval_funcs(
            self._packed_equality_grads, session, feed_dict, fetches)

        # Construct inequality constraint functions and associated gradients.
        inequality_funcs = self._make_eval_funcs(
            self._inequalities, session, feed_dict, fetches)
        inequality_grad_funcs = self._make_eval_funcs(
            self._packed_inequality_grads, session, feed_dict, fetches)

        # Get initial value from TF session.
        initial_packed_var_val = session.run(self._packed_var)

        # Apply a perturbation to initial value fed into L-BFGS-B
        if self.perturb_init:
            initial_packed_var_val += self.perturb_init * np.random.uniform(
                low=-1.0, high=1.0, size=initial_packed_var_val.shape)

        # Perform minimization.
        packed_var_val = self._minimize(
            initial_val=initial_packed_var_val,
            loss_grad_func=loss_grad_func,
            equality_funcs=equality_funcs,
            equality_grad_funcs=equality_grad_funcs,
            inequality_funcs=inequality_funcs,
            inequality_grad_funcs=inequality_grad_funcs,
            packed_bounds=self._packed_bounds,
            step_callback=step_callback,
            optimizer_kwargs=self.optimizer_kwargs)
        var_vals = [
            packed_var_val[packing_slice] for packing_slice
            in self._packing_slices
        ]

        # Set optimization variables to their new values.
        session.run(
            self._var_updates,
            feed_dict=dict(zip(self._update_placeholders, var_vals)),
            **run_kwargs)
