#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 romancpodolski <romancpodolski@dhcp-10-177-9-71.dynamic.eduroam.mwn.de>
#
# Distributed under terms of the MIT license.

"""
"""

from __future__ import print_function

import sys

from logistic_sgd import LogisticRegression, load_data
from data import load_data

sys.path.append('../data/')
__docformat__ = 'restructedtext en'

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
        dataset='mnist.pkl.gz',
        batch_size=600):
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

    # allocate symbotic variables for the dataj
    index = T.lscalar()

    # generate the symbolic variables for input (x and y represent a
    # minibatch)

    x = T.matrix('x') # data, represented as rasterized images
    y = T.ivector('y') #labes, represented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MINST image has size 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out = 10)

    # the cost we minimize during training is the negative log likelihood
    # the model is in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # combiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
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
    train_model = theano.function(
            inputs = [index],
            outputs = cost,
            updates = updates,
            givens = {
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )

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

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
        
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                        for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                        'epoch %i, minibatch, %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100
                            )
                        )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the set

                    test_losses = [test_model(i)
                            for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                            (
                            '   epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        )
                             % (
                                    epoch,
                                    minibatch_index + 1,
                                    n_train_batches,
                                    test_score * 100.
                                    )
                             )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
            (
                'Optimization complete whith best validation score of %f %%,'
                'with test performance %f %%'
                )
            % ( best_validation_loss * 100., test_score * 100. )
            )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' + 
        os.path.split(__file__)[1] + 
        ' ran for %.1fs' % (( end_time - start_time))), file=sys.stderr)

if __name__ == "__main__":
    sgd_optimization_mnist()
