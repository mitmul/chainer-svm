#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

import chainer
from chainer import cuda


class Optimizer(chainer.Optimizer):

    def l1_regularization(self, c=0.01):
        """Applies L1 regularization to the parameter/gradient pairs.

        Args:
            c (float): Coefficient of L1 norm term

        """
        for p, g, _ in self.tuples[:1]:
            if isinstance(p, cuda.ndarray):
                with cuda.get_device(p):
                    g = cuda.elementwise(
                        'T p, float32 c', 'T g',
                        'if (p > 0) g += c; else if (p < 0) g -= c;',
                        'l1_regularization')(p, c, g)
            else:
                g[numpy.where(p > 0)] += c
                g[numpy.where(p < 0)] -= c


class SGD(Optimizer):

    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one_cpu(self, param, grad, _):
        param -= self.lr * grad

    def update_one_gpu(self, param, grad, _):
        cuda.elementwise('T grad, T lr', 'T param',
                         'param -= lr * grad',
                         'sgd')(grad, self.lr, param)
