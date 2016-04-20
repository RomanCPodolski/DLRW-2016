#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 romancpodolski <romancpodolski@Romans-MacBook-Pro.local>
#
# Distributed under terms of the MIT license.

from __future__ import print_function

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

sys.path.append(os.path.join(os.path.split(__file__)[0], '..', 'data'))

from data import load_data

import numpy as np

import theano
import theano.tensor as T

import climin as cli
import climin.initialize as init
import climin.util
import itertools

import matplotlib.pyplot as plt

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=600, optimizer='gd'):

    datasets = load_data(dataset, shared = False)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    tmpl = [(28 * 28, 10), 10]
    flat, (Weights, bias) = climin.util.empty_with_views(tmpl)

    cli.initialize.randomize_normal(flat, 0, 0.1) # initialize the parameters with random numbers

    if batch_size is None:
        args = itertools.repeat(([train_set_x, train_set_y], {}))
        batches_per_pass = 1
    else:
        args = cli.util.iter_minibatches([train_set_x, train_set_y], batch_size, [0, 0])
        args = ((i, {}) for i in args)
        batches_per_pass = train_set_x.shape[0] / batch_size

    print('... building the model')

    W = theano.shared(
            value = np.zeros((28 * 28, 10), dtype = theano.config.floatX),
            name = 'W',
            borrow = True
            ) # weights dimension 28 * 28, 10

    b = theano.shared(
            value = np.zeros((10,), dtype = theano.config.floatX),
            name = 'b',
            borrow = True
            ) # biases

    x = T.matrix('x') # data, represented as rasterized images dimension 28 * 28
    y = T.ivector('y') # labes, represented as 1D vector of [int] labels dimension 10

    p_y_given_x =  T.nnet.softmax(T.dot(x, W) + b)
    y_pred      =  T.argmax(p_y_given_x, axis=1)

    NLL = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y]) # negative log likelihood
    
    cost = theano.function(inputs = [ x, y ], outputs = NLL, allow_input_downcast = True)
    
    g_W = theano.function(inputs = [ x, y ], outputs = T.grad(NLL, W), allow_input_downcast = True)
    g_b = theano.function(inputs = [ x, y ], outputs = T.grad(NLL, b), allow_input_downcast = True)

    def loss(parameters, inputs, targets):
        Weights, bias = climin.util.shaped_from_flat(parameters, tmpl)

        W.set_value(Weights)
        b.set_value(bias)

        return cost(inputs, targets)

    def d_loss_wrt_pars(parameters, inputs, targets):
        Weights, bias = climin.util.shaped_from_flat(parameters, tmpl)

        W.set_value(Weights)
        b.set_value(bias)

        return np.concatenate([g_W(inputs, targets).flatten(), g_b(inputs, targets)])

    if optimizer == 'gd':
        print('... using gradient descent')
        opt = cli.GradientDescent(flat, d_loss_wrt_pars, step_rate=0.1, momentum=.95, args=args)
    elif optimizer == 'lbfgs':
        print('... using using quasi-newton L-BFGS')
        opt = cli.Lbfgs(flat, loss, d_loss_wrt_pars, args=args)
    elif optimizer == 'nlcg':
        print('... using using non linear conjugate gradient')
        opt = cli.NonlinearConjugateGradient(flat, loss, d_loss_wrt_pars, args=args)
    elif optimizer == 'rmsprop':
        print('... using rmsprop')
        opt = cli.RmsProp(flat, d_loss_wrt_pars, step_rate=1e-4, decay=0.9, args=args)
    elif optimizer == 'rprop':
        print('... using resilient propagation')
        opt = cli.Rprop(flat, d_loss_wrt_pars, args=args)
    else:
        print('unknown optimizer')
        return 1

    print('... training the model')

    valid_losses = []
    train_losses = []
    test_losses = []

    done_looping = False
    epoch = 0

    start_time = timeit.default_timer()
    for info in opt:

        train_loss = cost(train_set_x, train_set_y)
        train_losses.append(train_loss)

        valid_loss = cost(valid_set_x, valid_set_y)
        valid_losses.append(valid_loss)

        test_loss = cost(test_set_x, test_set_y)
        test_losses.append(test_loss)

        print('Loss', valid_loss)
        if info['n_iter'] >= n_epochs and (not done_looping):
            break

    end_time = timeit.default_timer()

    print('Done Training ', end_time - start_time)

    return train_losses, valid_losses, test_losses

def predict():
    """
    An example of how to load a trained model and use it
    to predict labels.
    """

    # loads the saved model
    classifier = pickle.load(open('best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(
            inputs = [classifier.input],
            outputs = classifier.y_pred)

    # We can test it on some examples from test set
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print("Predicted values for the first 10 examples in test set:")
    print(predicted_values)

if __name__ == "__main__":
    f, (gd_plt, lbfgs_plt, nlcg_plt, rms_plt, rprop_plt) = plt.subplots(5)

    gd_train_loss, gd_valid_loss, gd_test_loss = sgd_optimization_mnist(optimizer='gd')
    gd_plt.plot(gd_train_loss, '-', linewidth=1, label='train loss')
    gd_plt.plot(gd_valid_loss, '-', linewidth=1, label='vaidation loss')
    gd_plt.plot(gd_test_loss, '-', linewidth=1, label='test loss')

    lbfgs_train_loss, lbfgs_valid_loss, lbfgs_test_loss = sgd_optimization_mnist(optimizer='lbfgs')
    lbfgs_plt.plot(lbfgs_train_loss, '-', linewidth=1, label='train loss')
    lbfgs_plt.plot(lbfgs_valid_loss, '-', linewidth=1, label='vaidation loss')
    lbfgs_plt.plot(lbfgs_test_loss, '-', linewidth=1, label='test loss')

    nlcg_train_loss, nlcg_valid_loss, nlcg_test_loss = sgd_optimization_mnist(optimizer='nlcg')
    nlcg_plt.plot(nlcg_train_loss, '-', linewidth=1, label='train loss')
    nlcg_plt.plot(nlcg_valid_loss, '-', linewidth=1, label='vaidation loss')
    nlcg_plt.plot(nlcg_test_loss, '-', linewidth=1, label='test loss')

    rms_train_loss, rms_valid_loss, rms_test_loss = sgd_optimization_mnist(optimizer='rmsprop')
    rms_plt.plot(rms_train_loss, '-', linewidth=1, label='train loss')
    rms_plt.plot(rms_valid_loss, '-', linewidth=1, label='vaidation loss')
    rms_plt.plot(rms_test_loss, '-', linewidth=1, label='test loss')

    rprop_train_loss, rprop_valid_loss, rprop_test_loss = sgd_optimization_mnist(optimizer='rprop')
    rprop_plt.plot(rprop_train_loss, '-', linewidth=1, label='train loss')
    rprop_plt.plot(rprop_valid_loss, '-', linewidth=1, label='vaidation loss')
    rprop_plt.plot(rprop_test_loss, '-', linewidth=1, label='test loss')

    plt.savefig('errors.png')
