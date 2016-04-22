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

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

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

    x = T.matrix('x') # data, represented as rasterized images dimension 28 * 28
    y = T.ivector('y') # labes, represented as 1D vector of [int] labels dimension 10

    parameters = T.vector('params') # flat vector storing the parameters, required from climin

    classifier = LogisticRegression(input = x, n_in = 28 * 28, n_out = 10)

    cost = theano.function(
            inputs = [ x, y ],
            outputs = classifier.negative_log_likelihood(y),
            allow_input_downcast = True
            )

    gradients = theano.function(
            inputs = [x, y],
            outputs = [
                T.grad(classifier.negative_log_likelihood(y), classifier.W),
                T.grad(classifier.negative_log_likelihood(y), classifier.b)
                ],
            # updates = [
                # (classifier.W, u_W),
                # (classifier.b, u_b),
                # ],
            allow_input_downcast = True
            )
    # loss = theano.function(
            # inputs = [parameters, x, y ],
            # outputs = NLL,
            # updates = [
                # (W, climin.util.shaped_from_flat(parameters, tmpl)[0]),
                # (b, climin.util.shaped_from_flat(parameters, tmpl)[1]),
                # ],
            # allow_input_downcast = True
            # )

    # d_loss_wrt_pars = theano.function(
            # inputs = [parametrs, x, y ],
            # outputs = [
                # T.grad(NLL, W),
                # T.grad(NLL, b)
                # ],
            # updates = [
                # (W, climin.util.shaped_from_flat(parameters, tmpl)[0]),
                # (b, climin.util.shaped_from_flat(parameters, tmpl)[1]),
                # ],
            # allow_input_downcast = True
            # )

    def loss(parameters, input, target):
        Weights, bias = climin.util.shaped_from_flat(parameters, tmpl)

        classifier.W.set_value(Weights)
        classifier.b.set_value(bias)

        return cost(input, target)

    def d_loss_wrt_pars(parameters, inputs, targets):
        Weights, bias = climin.util.shaped_from_flat(parameters, tmpl)

        classifier.W.set_value(Weights)
        classifier.b.set_value(bias)

        g_W, g_b = gradients(inputs, targets)

        return np.concatenate([g_W.flatten(), g_b])

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

    losses = (train_losses, valid_losses, test_losses)

    return classifier, losses 

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

    gd_classifier, gd_losses = sgd_optimization_mnist(optimizer='gd')
    gd_train_loss, gd_valid_loss, gd_test_loss = gd_losses

    lbfgs_classifier, lbfgs_losses = sgd_optimization_mnist(optimizer='lbfgs')
    lbfgs_train_loss, lbfgs_valid_loss, lbfgs_test_loss = lbfgs_losses

    nlcg_classifier, nlcg_losses = sgd_optimization_mnist(optimizer='nlcg')
    nlcg_train_loss, nlcg_valid_loss, nlcg_test_loss = nlcg_losses

    rms_classifier, rms_losses = sgd_optimization_mnist(optimizer='rmsprop')
    rms_train_loss, rms_valid_loss, rms_test_loss = rms_losses

    rprop_classifier, rprop_losses = sgd_optimization_mnist(optimizer='rprop')
    rprop_train_loss, rprop_valid_loss, rprop_test_loss = rprop_losses

    f_errors, (gd_plt, lbfgs_plt, nlcg_plt, rms_plt, rprop_plt) = plt.subplots(5, sharex = True, sharey = True)

    gd_plt.plot(gd_train_loss, '-', linewidth=1, label='train loss')
    gd_plt.plot(gd_valid_loss, '-', linewidth=1, label='vaidation loss')
    gd_plt.plot(gd_test_loss, '-', linewidth=1, label='test loss')

    lbfgs_plt.plot(lbfgs_train_loss, '-', linewidth=1, label='train loss')
    lbfgs_plt.plot(lbfgs_valid_loss, '-', linewidth=1, label='vaidation loss')
    lbfgs_plt.plot(lbfgs_test_loss, '-', linewidth=1, label='test loss')

    nlcg_plt.plot(nlcg_train_loss, '-', linewidth=1, label='train loss')
    nlcg_plt.plot(nlcg_valid_loss, '-', linewidth=1, label='vaidation loss')
    nlcg_plt.plot(nlcg_test_loss, '-', linewidth=1, label='test loss')

    rms_plt.plot(rms_train_loss, '-', linewidth=1, label='train loss')
    rms_plt.plot(rms_valid_loss, '-', linewidth=1, label='vaidation loss')
    rms_plt.plot(rms_test_loss, '-', linewidth=1, label='test loss')

    rprop_plt.plot(rprop_train_loss, '-', linewidth=1, label='train loss')
    rprop_plt.plot(rprop_valid_loss, '-', linewidth=1, label='vaidation loss')
    rprop_plt.plot(rprop_test_loss, '-', linewidth=1, label='test loss')

    plt.savefig('errors.png')

    f_repfields, (gd_plt, lbfgs_plt, nlcg_plt, rms_plt, rprop_plt) = plt.subplots(1,5)

    gd_plt.imshow(gd_classifier.W.get_value(), cmap = 'Greys_r')
    lbfgs_plt.imshow(lbfgs_classifier.W.get_value(), cmap = 'Greys_r')
    nlcg_plt.imshow(nlcg_classifier.W.get_value(), cmap = 'Greys_r')
    rms_plt.imshow(rms_classifier.W.get_value(), cmap = 'Greys_r')
    rprop_plt.imshow(rprop_classifier.W.get_value(), cmap = 'Greys_r')
    plt.savefig('repfields.png')

