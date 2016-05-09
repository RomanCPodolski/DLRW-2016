#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 romancpodolski <romancpodolski@Romans-MacBook-Pro.local>
#
# Distributed under terms of the MIT license.

"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid). One can use many such
hidden layers making the architecture deep. This tutorial will also tackle
the problem of MNIST digit classification.

.. math:

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Patternrecognition and Machine Learning" -
                 Chistopher M. Bishop, section 5
"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import os
import sys
import timeit

sys.path.append(os.path.join(os.path.split(__file__)[0], '..', 'logreg'))
from logistic_sgd import LogisticRegression

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

class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, W = None, b = None, activation=T.tanh):
        """Typical hidden Layer of a MLP: units are fully-connected and have
        sigmodial activation function. Weight matrix W is of shape (n_in, n_out)
        and the bias vector is of shape (n_out,)

        NOTE: The nonlinearity used her is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_example, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for than activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function is used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                    rng.uniform(
                        low = -np.sqrt(6. / (n_in + n_out)),
                        high = np.sqrt(6. / (n_in + n_out)),
                        size = (n_in, n_out)
                        ),
                    dtype = theano.config.floatX
                    )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value = W_values, name = 'W', borrow = True)

        if b is None:
            b_values = np.zeros((n_out,), dtype = theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)

        self.W = W
        self.b = b

        self.params = [self.W, self.b]

        lin_output = T.dot(input, self.W) + self.b
        self.output = ( lin_output if activation is None else activation(lin_output))

class MLP(object):

    """Multi-Layer Perceptron Class
    
    A multilayer perceptron is a feedforward artificial neural network model
    that is one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defines here by ``HiddenLayer`` class) while the
    top layer a softmax layer (defined here by a ``LogisticRegression``
    class)."""

    def __init__(self, rng, input, n_in, n_hidden, n_out, activation_h = T.tanh, W_hidden = None, b_hidden = None, W_log = None, b_log = None):
        """Initialize the parameters for the multilayer perceptron
        
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialze weights
        
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)
        
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie
        
        :type n_hidden: int
        :param n_hidden: number of hidden units
        
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
                rng = rng,
                input = input,
                n_in = n_in,
                n_out = n_hidden,
                activation = activation_h,
                W = W_hidden,
                b = b_hidden,
                )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
                input = self.hiddenLayer.output,
                n_in = n_hidden,
                n_out = n_out,
                W = W_log,
                b = b_log
                )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = ( abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = ( (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum())

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
                self.logRegressionLayer.negative_log_likelihood
                )

        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # keep track of model input
        self.input = input

def test_mlp(learning_rate = 0.01, L1_reg = 0.00, L2_reg = 0.0001, n_epochs = 1000,
        dataset = 'mnist.pkl.gz', batch_size = 20, n_hidden = 300, optimizer = 'gd', activation = T.tanh):
    """
    Demonstrate stochastic gradient decent optimisation for a multilayer
    percepron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient)

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2 reg: float
    :param L2 reg: L2-norm's weight when added to the cost (see
    regularization)
    
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    tmpl = [(28 * 28, n_hidden), n_hidden, (n_hidden, 10), 10]
    flat, (Weights_hidden, bias_hidden, Weights_log, bias_log) = climin.util.empty_with_views(tmpl)

    # cli.initialize.randomize_normal(flat, -np.sqrt(6. / (28*28 + n_hidden)), np.sqrt(6. / (28*28 + n_hidden))) # initialize the parameters with random numbers
    cli.initialize.randomize_normal(flat, 0, 0.01) # initialize the parameters with random numbers

    if batch_size is None:
        args = itertools.repeat(([train_set_x, train_set_y], {}))
        batches_per_pass = 1
    else:
        args = cli.util.iter_minibatches([train_set_x, train_set_y], batch_size, [0, 0])
        args = ((i, {}) for i in args)
        n_train_batches = train_set_x.shape[0] // batch_size
        n_valid_batches = valid_set_x.shape[0] // batch_size
        n_test_batches  = test_set_x.shape[0] // batch_size

    ########################
    #  BUILD ACTUAL MODEL  #
    ########################
    print('... building the model')

    # allocate symbolic variables for the data
    # index = T.lscalar() # index to a [mini]batch
    x = T.matrix('x') # the data is represented as rasterized images
    y = T.ivector('y') # the labels are presented as 1D of [int] labels
    parameters = T.vector('parameters')

    rng = np.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
            rng = rng,
            input = x,
            n_in = 28 * 28,
            n_hidden = n_hidden,
            n_out = 10,
            activation_h = activation,
            W_hidden = theano.shared(value = Weights_hidden, name = 'W_h', borrow = True),
            b_hidden = theano.shared(value = bias_hidden, name = 'b_h', borrow = True),
            W_log    = theano.shared(value = Weights_log, name = 'W_l', borrow = True),
            b_log    = theano.shared(value = bias_log, name = 'b_l', borrow = True)
            )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost_t = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    cost = theano.function(
            inputs = [x, y],
            outputs = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr,
            allow_input_downcast = True
            )

    # compiling a theano function, that computes the mistakes that are made
    # by the model on a minibatch
    zero_one_loss = theano.function(
            inputs = [x, y],
            outputs = classifier.errors(y),
            allow_input_downcast = True
            )

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradient will be stored in a list gparams
    gparams = [T.grad(cost_t, param) for param in classifier.params]

    gradients = theano.function(
            inputs = [x, y],
            outputs = gparams,
            allow_input_downcast = True
            )

    def loss(paramters, inputs, targets):
        # Weights_log, bias_log, Weights_hidden, bias_hidden = climin.util.shaped_from_flat(parameters, tmpl)

        return cost(inputs, targets)

    def d_loss_wrt_pars(parameters, inputs, targets):
        # Weights_hidden, bias_hidden, Weights_log, bias_log = climin.util.shaped_from_flat(parameters, tmpl)

        g_W_h, g_b_h, g_W_l, g_b_l = gradients(inputs, targets)

        return np.concatenate([g_W_h.flatten(), g_b_h, g_W_l.flatten(), g_b_l])

    if optimizer == 'gd':
        print('... using gradient descent')
        opt = cli.GradientDescent(flat, d_loss_wrt_pars, step_rate = learning_rate, momentum = .95, args=args)
    elif optimizer == 'bfgs':
        print('... using using quasi-newton BFGS')
        opt = cli.Bfgs(flat, loss, d_loss_wrt_pars, args=args)
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
    elif optimizer == 'adam':
        print('... using adaptive momentum estimation optimizer')
        opt = cli.Adam(flat, d_loss_wrt_pars, step_rate = 0.0002, decay = 0.99999999, decay_mom1 = 0.1, decay_mom2 = 0.001, momentum = 0, offset = 1e-08, args=args)
    elif optimizer == 'adadelta':
        print('... using adadelta')
        opt = cli.Adadelta(flat, d_loss_wrt_pars, step_rate=1, decay = 0.9, momentum = .95, offset = 0.0001, args=args)
    else:
        print('unknown optimizer')
        return 1

    #################
    #  TRAIN MODEL  #
    #################
    print('... training')

    # early-stopping parameters
    patience = 10000 # look at this many examples regardless
    patience_increase = 2 # wait this much longer when a new best is found
    improvement_threshold = 0.995 # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience // 2) # go trough this many minibatches before checking the network on the validation set; in this case we check every epoch
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    valid_losses = []
    train_losses = []
    test_losses = []

    for info in opt:
        iter = info['n_iter'] - 1
        epoch = iter // n_train_batches + 1
        minibatch_index = iter % n_train_batches
        minibatch_x, minibatch_y = info['args']
        
        # minibatch_average_cost = cost(minibatch_x, minibatch_y)

        if (iter + 1) % validation_frequency == 0:
            validation_loss = zero_one_loss(valid_set_x, valid_set_y)
            valid_losses.append([epoch, validation_loss])

            print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        validation_loss * 100.
                        )
                    )
            # if we got the best validation score until now
            if validation_loss < best_validation_loss:
                # improve patience if loss improvement is good enough
                if (validation_loss < best_validation_loss * improvement_threshold):
                    patience = max(patience, iter * patience_increase)

                best_validation_loss = validation_loss
                best_iter = iter
                test_score = zero_one_loss(test_set_x, test_set_y)

                # test it on the test set
                test_losses.append([epoch, test_score])

                print(('   epoch %i, minibatch index %i/%i, test error of '
                    ' best model %f %%') %
                    (epoch, minibatch_index + 1, n_train_batches, test_score * 100.)) 

                # with open('best_model_mlp.pkl', 'wb') as f:
                    # pickle.dump(classifier, f)

        if patience <= iter or epoch >= n_epochs:
            break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best varidation score of %f %%'
        'obtained at iteration %i, with test performance %f %%') %
        (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
        os.path.split(__file__)[1] +
        ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    losses = (np.asarray(train_losses), np.asarray(valid_losses), np.asarray(test_losses))
    methadata = (best_validation_loss * 100., test_score * 100.,(end_time - start_time) / 60.)

    return classifier, losses, methadata

if __name__ == "__main__":
    # f = plt.figure()
    # classifier, losses, methadata = test_mlp(activation = T.tanh, optimizer = 'rmsprop')
    # train_loss, valid_loss, test_loss = losses
    # best_validation_loss, best_test_loss, time_trained = methadata

    # # plt.plot(gd_train_loss, '-', linewidth = 1, label = 'train loss')
    # plt.plot(valid_loss[:,0], valid_loss[:,1], '-', linewidth = 1, label = 'validation loss')
    # plt.plot(test_loss[:,0], test_loss[:,1], '-', linewidth = 1, label = 'test loss')

    # plt.legend()

    # plt.title('Error activation tanh with best validation score of %f %%,\n test performance %f %%, after %.1fm ' % (best_validation_loss, best_test_loss, time_trained))
    # plt.savefig('errors_tanh.png')

    # f_repfields, axes  = plt.subplots(15, 20, subplot_kw = {'xticks': [], 'yticks': []})
    # repfield = []

    # for i in range(300):
        # repfield.append(np.reshape(np.array(classifier.hiddenLayer.W.get_value())[:,i], (28, 28)))

    # for ax, rep in zip(axes.flat, repfield):
        # ax.imshow(rep, cmap=plt.cm.gray, interpolation = 'none')

    # f_repfields.suptitle('Receptive Fields for a two layer neural net with 300 tanh neurons on MNIST')
    # plt.savefig('repfields_tanh.png')

    f = plt.figure()
    classifier, losses, methadata = test_mlp(activation = T.nnet.sigmoid, optimizer = 'gd')
    train_loss, valid_loss, test_loss = losses
    best_validation_loss, best_test_loss, time_trained = methadata

    # plt.plot(gd_train_loss, '-', linewidth = 1, label = 'train loss')
    plt.plot(valid_loss[:,0], valid_loss[:,1], '-', linewidth = 1, label = 'validation loss')
    plt.plot(test_loss[:,0], test_loss[:,1], '-', linewidth = 1, label = 'test loss')

    plt.legend()

    plt.title('Error activation sigmoid with best validation score of %f %%,\n test performance %f %%, after %.1fm ' % (best_validation_loss, best_test_loss, time_trained))
    plt.savefig('errors_sigmoid.png')

    f_repfields, axes  = plt.subplots(15, 20, subplot_kw = {'xticks': [], 'yticks': []})
    repfield = []

    for i in range(300):
        repfield.append(np.reshape(np.array(classifier.hiddenLayer.W.get_value())[:,i], (28, 28)))

    for ax, rep in zip(axes.flat, repfield):
        ax.imshow(rep, cmap=plt.cm.gray, interpolation = 'none')

    f_repfields.suptitle('Receptive Fields for a two layer neural net with 300 sigmoid neurons on MNIST')
    plt.savefig('repfields_sigmoid.png')

    f = plt.figure()
    classifier, losses, methadata = test_mlp(activation = T.nnet.relu, optimizer = 'gd')
    train_loss, valid_loss, test_loss = losses
    best_validation_loss, best_test_loss, time_trained = methadata

    # plt.plot(gd_train_loss, '-', linewidth = 1, label = 'train loss')
    plt.plot(valid_loss[:,0], valid_loss[:,1], '-', linewidth = 1, label = 'validation loss')
    plt.plot(test_loss[:,0], test_loss[:,1], '-', linewidth = 1, label = 'test loss')

    plt.legend()

    plt.title('Error activation ReLU with best validation score of %f %%,\n test performance %f %%, after %.1fm ' % (best_validation_loss, best_test_loss, time_trained))
    plt.savefig('errors_relu.png')

    f_repfields, axes  = plt.subplots(15, 20, subplot_kw = {'xticks': [], 'yticks': []})
    repfield = []

    for i in range(300):
        repfield.append(np.reshape(np.array(classifier.hiddenLayer.W.get_value())[:,i], (28, 28)))

    for ax, rep in zip(axes.flat, repfield):
        ax.imshow(rep, cmap=plt.cm.gray, interpolation = 'none')

    f_repfields.suptitle('Receptive Fields for a two layer neural net with 300 ReLU Neurons on MNIST')
    plt.savefig('repfields_relu.png')
