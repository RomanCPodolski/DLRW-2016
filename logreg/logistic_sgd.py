#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 romancpodolski <romancpodolski@Romans-MacBook-Pro.local>
#
# Distributed under terms of the MIT license.

"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

    .. math::

      P(Y = i | x, W, b) &= softmax_i(W x + b) \\
                         &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}

The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i | x).

   .. math::

      y_{pred} = argmax_i P(Y = i|x,W,b)

This tutorial presents a stochastic gradient descent optimisation method
suitable for large datasets.

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import os
import sys
import timeit

sys.path.append(os.path.join( os.path.split(__file__)[0], '..', 'data'))

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
        :param n_in: number of the input units, the dimension of the space in
                     which the datapoints lie
        
        :type n_out: int
        :param n_out: number of output units, the dimensions of the space in 
                      which the labels lie
        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
                value = np.zeros(
                    (n_in, n_out),
                    dtype = theano.config.floatX
                    ),
                name = 'W', # weights
                borrow = True
                )

        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
                value = np.zeros(
                    (n_out,),
                    dtype = theano.config.floatX
                    ),
                name = 'b', # bias
                borrow = True
                )

        # <latex>
        # P(Y = i | x, W, b) = softmax_i(Wx + b)
        #</latex>
        # sybolic expression for computing the matrix of class-membership probabilities
        # Where:
        # W is a matrix where column-k represents the seperation hyperplane for class-k
        # x is a matrix where row-j represents input traning sample-j
        # b is a vector where element-k represents the free parameter of hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as a class whose probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parametes of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution

        .. math::
          \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
          \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
            \log(P(Y=y^{(i)}|x^{(i)}, W, b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the correct label

        Note: we use the mean instead of the sum so that 
              the learning rate is less dependent on the batch size

        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n)
        # T.arrange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,...,n-1] T.log(self.p_y_given_x) is a matrix of 
        # Log-Probabilities (call it LP) whith one row per example and 
        # one column per class LP[T.arrange(y.shape[0]), y] is a vector 
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2, y[2]], ...,
        # LP[n-1, y[n-1]]] and T.mean(LP[T.arrange(y.shape[0]), y]) is 
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelyhood across the minibatch
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has the same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            return TypeError(
                    'y should have the same shape as self.y_pred',
                    ('y', y.type, 'y_pred', y_pred.type)
                    )
        # check if y is one of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistace in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
        dataset='mnist.pkl.gz',
        batch_size=600, optimizer='gd'):
    """Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    :type optimizer: string
    :param optimizer: the climin optimizer to use. One can choose gradient decent ('gd'), RmsProp ('rmsprop')
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ########################
    #  BUILD ACTUAL MODEL  #
    ########################
    print('... building the model')

    # allocate symbotic variables for the data
    index = T.lscalar()

    # generate the symbolic variables for input (x and y represent a
    # minibatch)

    x = T.matrix('x') # data, represented as rasterized images
    y = T.ivector('y') # labes, represented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MINST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out = 10)

    # the cost we minimize during training is the negative log likelihood
    # the model is in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # combiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    # d_loss_wrt_pars = theano.function(
            # inputs = [parameters, x, y],
            # outputs = [T.grad(cost=cost, wrt=classifier.W).flatten(), T.grad(cost=cost, wrt=classifier.b)]
            # )

    test_model = theano.function(
            inputs = [index],
            outputs = classifier.errors(y),
            givens = {
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )

    validate_model = theano.function(
            inputs = [index],
            outputs = classifier.errors(y),
            givens = {
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )

    # train_error_model = theano.function(
            # inputs = [index],
            # outputs = classifier.errors(y),
            # givens = {
                # x: train_set_x[index * batch_size: (index + 1) * batch_size],
                # y: train_set_y[index * batch_size: (index + 1) * batch_size]
                # }
            # )

    # compute the gradient of cost with respect to theta=(W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`

    # to make this model work with climin, make the loss function take an array of data and return a loss function
    # train_model = theano.function(
            # inputs = [classifier.params, x, y],
            # outputs = cost,
            # updates = updates,
            # givens = {
                # x: x,
                # y: y
                # }
            # )
    # train_model = theano.function(
            # inputs = [index],
            # outputs = cost,
            # updates = updates,
            # givens = {
                # x: train_set_x[index * batch_size: (index + 1) * batch_size],
                # y: train_set_y[index * batch_size: (index + 1) * batch_size]
                # }
            # )

    flat = np.empty(7850)
    cli.initialize.randomize_normal(flat, 0, 1)

    # if batch_size is None:
    args = itertools.repeat(([train_set_x, train_set_y], {}))
    batches_per_pass = 1
    # else:
        # args = cli.util.iter_minibatches([train_set_x, train_set_y], batch_size, [0, 0])
        # args = ((i, {}) for i in args)
        # batches_per_pass = train_set_x.shape[0] / batch_size

    if optimizer == 'gd':
        opt = cli.GradientDescent(classifier.params, classifier.negative_log_likelihood, step_rate=0.1, momentum=.95, args=args)
    elif optimizer == 'lbfgs':
        opt = cli.Lbfgs(flat, loss, d_loss_wrt_pars, args=args)
    elif optimizer == 'ncg':
        opt = cli.NonlinearConjugateGradient(flat, loss, d_loss_wrt_pars,
                                                args=args)
    elif optimizer == 'rmsprop':
        opt = cli.RmsProp(flat, d_loss_wrt_pars, step_rate=1e-4, decay=0.9,
                             args=args)
    elif optimizer == 'rprop':
        opt = cli.Rprop(flat, d_loss_wrt_pars, args=args)
    else:
        print('unknown optimizer')
        return 1

    #################
    #  TRAIN MODEL  #
    #################
    print('... training the model')
    # early stopping parameters
    patience = 5000 # look at this many examples regardless
    patience_increase = 2 # wait this mutch longer when a new best is found
    improvement_threshold = 0.995 # for a relative improvement of this much is
                                  # considered significant 
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go trough this many
                                  # minibatches before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    train_errors = []
    valid_errors = []
    test_errors = []

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    for info in opt:
        if epoch < n_epochs and (not done_looping):
            break
    # while (epoch < n_epochs) and (not done_looping):
        # epoch = epoch + 1
        # for minibatch_index in range(n_train_batches):
            # minibatch_avg_cost = train_model(minibatch_index)
            # iter = (epoch - 1) * n_train_batches + minibatch_index
        
            # # train_errors.append(np.mean([train_model(i) for i in range(n_train_batches)]))
            # # train_errors.append(minibatch_avg_cost * 100)

            # if (iter + 1) % validation_frequency == 0:
                # # compute zero-one loss on validation set
                # validation_losses = [validate_model(i)
                        # for i in range(n_valid_batches)]
                # this_validation_loss = np.mean(validation_losses)

                # print(
                        # 'epoch %i, minibatch, %i/%i, validation error %f %%' %
                        # (
                            # epoch,
                            # minibatch_index + 1,
                            # n_train_batches,
                            # this_validation_loss * 100
                            # )
                        # )

                # # valid_errors.append(this_validation_loss * 100)
                # # train_errors.append(np.mean([train_error_model(i) for i in range(n_train_batches)]) * 100)

                # # if we got the best validation score until now
                # if this_validation_loss < best_validation_loss:
                    # # improve patience if loss improvement is good enough
                    # if this_validation_loss < best_validation_loss * \
                            # improvement_threshold:
                        # patience = max(patience, iter * patience_increase)

                    # best_validation_loss = this_validation_loss
                    # # test it on the set

                    # test_losses = [test_model(i)
                            # for i in range(n_test_batches)]
                    # test_score = np.mean(test_losses)

                    # print(
                            # (
                            # '   epoch %i, minibatch %i/%i, test error of'
                            # ' best model %f %%'
                        # )
                             # % (
                                    # epoch,
                                    # minibatch_index + 1,
                                    # n_train_batches,
                                    # test_score * 100.
                                    # )
                             # )

                    # test_errors.append(test_score * 100)
                    # valid_errors.append(this_validation_loss * 100)
                    # train_errors.append(np.mean([train_error_model(i) for i in range(n_train_batches)]) * 100)

                    # # save the best model
                    # with open('best_model.pkl', 'wb') as f:
                        # pickle.dump(classifier, f)

            # if patience <= iter:
                # done_looping = True
                # break

    # end_time = timeit.default_timer()
    # print(
            # (
                # 'Optimization complete whith best validation score of %f %%,'
                # 'with test performance %f %%'
                # )
            # % ( best_validation_loss * 100., test_score * 100. )
            # )
    # print('The code run for %d epochs, with %f epochs/sec' % (
        # epoch, 1. * epoch / (end_time - start_time)))
    # print(('The code for file ' + 
        # os.path.split(__file__)[1] + 
        # ' ran for %.1fs' % (( end_time - start_time))), file=sys.stderr)

    # train_error_plt = plt.plot(train_errors, '-', linewidth=1, label='train loss')
    # test_error_plt = plt.plot(test_errors, '-',   linewidth=1, label='test loss')
    # valid_error_plt = plt.plot(valid_errors, '-', linewidth=1, label='valid loss')

    # plt.legend()

    # plt.xlabel('epoch')
    # plt.ylabel('error in %')

    # plt.savefig('errors.png')

    # plt.show()

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
    sys.exit(sgd_optimization_mnist())
