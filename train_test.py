#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')
import argparse
import chainer
import numpy as np
import matplotlib.pyplot as plt
from chainer import cuda
from chainer import optimizers
from svm import SVM
from optimizer import SGD
from sklearn.datasets.samples_generator import make_blobs


def get_model_optimizer(args):
    model = SVM(c=args.c, penalty=args.penalty)
    if args.gpu >= 0:
        model.to_gpu()
    if args.penalty == 'l2':
        optimizer = optimizers.SGD(lr=args.lr)
    elif args.penalty == 'l1':
        optimizer = SGD(lr=args.lr)
    optimizer.setup(model)

    return model, optimizer


def get_data():
    X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
    Y[np.where(Y == 0)] = -1

    return X, Y


def train(args, X, Y, model, optimizer):
    N = len(Y)
    losses = []
    xp = cuda.cupy if args.gpu >= 0 and cuda.available else np
    for epoch in range(args.epoch):
        perm = np.random.permutation(N)
        sum_loss = 0
        for i in range(0, N, args.batchsize):
            x = xp.asarray(X[perm[i:i + args.batchsize]], dtype=np.float32)
            t = xp.asarray(Y[perm[i:i + args.batchsize]], dtype=np.int32)

            optimizer.zero_grads()
            loss = model.forward(x, t)
            loss.backward()
            if args.penalty == 'l1':
                optimizer.l1_regularization(c=args.c)
            elif args.penalty == 'l2':
                optimizer.weight_decay(decay=args.c)
            optimizer.update()
            sum_loss += float(loss.data) * len(t)

        print('Epoch:{}\tloss:{}'.format(epoch, sum_loss / N))
        losses.append(sum_loss)

    return losses


def visualize(args, X, Y, model, losses, init):
    W, b = model.fc.W.reshape((2,)), model.fc.b
    W_init, b_init = init
    if args.gpu >= 0:
        W, b = map(cuda.cupy.asnumpy, [W, b])
        W_init, b_init = map(cuda.cupy.asnumpy, [W_init[0], b_init])

    plt.plot(losses)
    plt.savefig('loss.png')
    plt.clf()

    x = np.arange(X[:, 0].min(), X[:, 0].max() + 1e-3, 1e-3)
    y = - W[0] / W[1] * x - b / W[1]
    plt.plot(x, y, label='result')

    x = np.arange(X[:, 0].min(), X[:, 0].max() + 1e-3, 1e-3)
    y = - W_init[0] / W_init[1] * x - b_init / W_init[1]
    plt.plot(x, y, label='initial')

    x1 = X[np.where(Y == -1)]
    x2 = X[np.where(Y == 1)]
    plt.scatter(x1[:, 0], x1[:, 1], c='b')
    plt.scatter(x2[:, 0], x2[:, 1], c='r')
    plt.legend()
    plt.savefig('data.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=-1)
    parser.add_argument('--c', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--penalty', type=str, default='l1')
    args = parser.parse_args()

    X, Y = get_data()
    model, optimizer = get_model_optimizer(args)
    init = [model.fc.W.copy(), model.fc.b.copy()]
    losses = train(args, X, Y, model, optimizer)
    visualize(args, X, Y, model, losses, init)
