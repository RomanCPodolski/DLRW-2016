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
    
    loss = theano.function(inputs = [ x, y ], outputs = NLL, allow_input_downcast = True)
    
    g_W = theano.function(inputs = [ x, y ], outputs = T.grad(NLL, W), allow_input_downcast = True)
    g_b = theano.function(inputs = [ x, y ], outputs = T.grad(NLL, b), allow_input_downcast = True)

    def d_loss_wrt_pars(parameters, inputs, targets):
        Weights, bias = climin.util.shaped_from_flat(parameters, tmpl)

        W.set_value(Weights)
        b.set_value(bias)

        return np.concatenate([g_W(inputs, targets).flatten(), g_b(inputs, targets)])

    if optimizer == 'gd':
        opt = cli.GradientDescent(flat, d_loss_wrt_pars, step_rate=0.1, momentum=.95, args=args)
    elif optimizer == 'lbfgs':
        opt = cli.Lbfgs(flat, loss, d_loss_wrt_pars, args=args)
    elif optimizer == 'ncg':
        opt = cli.NonlinearConjugateGradient(flat, loss, d_loss_wrt_pars, args=args)
    elif optimizer == 'rmsprop':
        opt = cli.RmsProp(flat, d_loss_wrt_pars, step_rate=1e-4, decay=0.9, args=args)
    elif optimizer == 'rprop':
        opt = cli.Rprop(flat, d_loss_wrt_pars, args=args)
    else:
        print('unknown optimizer')
        return 1

    #################
    #  TRAIN MODEL  #
    #################
    print('... training the model')

    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    for info in opt:
        # print(info['loss'], loss(valid_set_x, valid_set_y))
        print('Loss', loss(valid_set_x, valid_set_y))
        if info['n_iter'] >= n_epochs and (not done_looping):
            break

    end_time = timeit.default_time()

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
    sys.exit(sgd_optimization_mnist(optimizer='rprop'))
