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
    if args.penalty == 'L2':
        optimizer = optimizers.SGD(lr=args.lr)
    elif args.penalty == 'L1':
        optimizer = SGD(lr=args.lr)
    optimizer.setup(model)

    return model, optimizer


def get_data():
    X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.5)

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
            if args.penalty == 'L1':
                optimizer.l1_regularization(c=args.c)
            elif args.penalty == 'L2':
                optimizer.weight_decay(decay=args.c)
            optimizer.update()
            sum_loss += float(loss.data) * len(t)

        print('Epoch:{}\tloss:{}'.format(epoch, sum_loss / N))
        losses.append(sum_loss)

    return losses


def get_z(args, model, X, Y, i):
    W, b = model.fc.W[i, :], model.fc.b[i]
    if args.gpu >= 0:
        W, b = map(cuda.cupy.asnumpy, [W, b])
        X, Y = map(cuda.cupy.asnumpy, [X, Y])

    delta = 0.02
    x = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), delta)
    y = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), delta)
    x, y = np.meshgrid(x, y)
    xx, yy = map(np.ravel, [x, y])
    z = (W[0] * xx + W[1] * yy + b).reshape(x.shape)
    z[np.where(z > 0)], z[np.where(z <= 0)] = 1, -1

    plt.clf()
    plt.xlim([np.min(X[:, 0]) + delta, np.max(X[:, 0]) - delta])
    plt.ylim([np.min(X[:, 1]) + delta, np.max(X[:, 1]) - delta])
    plt.contourf(x, y, z, cmap=plt.cm.Paired, alpha=0.8)

    x0 = X[np.where(Y == 0)]
    x1 = X[np.where(Y == 1)]
    plt.scatter(x0[:, 0], x0[:, 1], c='b')
    plt.scatter(x1[:, 0], x1[:, 1], c='r')
    plt.savefig('data_{}.png'.format(i))


def visualize(args, X, Y, model, losses):
    plt.plot(losses)
    plt.savefig('loss.png')
    plt.clf()

    get_z(args, model, X, Y, 0)
    get_z(args, model, X, Y, 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=-1)
    parser.add_argument('--c', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--penalty', type=str, default='L1')
    args = parser.parse_args()

    X, Y = get_data()
    model, optimizer = get_model_optimizer(args)
    losses = train(args, X, Y, model, optimizer)
    visualize(args, X, Y, model, losses)
