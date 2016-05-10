#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 romancpodolski <roman.podolski@tum.de>
#
# Distributed under terms of the MIT license.

"""
TODO
References:

    - papers: "Learning Feature Represantations with K-means" -
              Adam Coates and Andreq Y. Ng
"""

from __future__ import print_function
import theano
import theano.tensor as T

import numpy as np

import matplotlib.pyplot as plt

import sys
import os
import six.moves.cPickle as pickle

sys.path.append(os.path.join(os.path.split(__file__)[0], '..', 'data'))
from data import load_data

class K_Means(object):

    def __init__(self, n_dim, n_samples, k = 500, input = None):
        """Instance of k-Means object
        """
        self.k = k 
        self.input = input

        self.D = theano.shared(
                name = 'D',
                value = np.asarray(
                    np.random.normal(0, size = (n_dim, k)),
                    dtype = theano.config.floatX
                    ),
                borrow = True
                )

        self.S = theano.shared(
                name = 'S',
                value = np.zeros(
                    (k, n_samples),
                    dtype = theano.config.floatX
                    ),
                borrow = True
                )

        max_position  = T.argmax(abs(T.dot(self.D.T, self.input)), axis = 0)
        max_values    = T.dot(self.D.T, self.input)[max_position, T.arange(max_position.shape[0])]
        zero_sub      = T.zeros_like(self.S)[max_position, T.arange(max_position.shape[0])]
        self.S_update = T.set_subtensor(zero_sub, max_values)

        D_update = T.dot(self.input, self.S.T) + self.D
        self.D_norm = D_update / D_update.norm(2)

        self.cost = T.sum(T.sqrt(T.sum(T.sqr(T.dot(self.D, self.S) - self.input), axis = 0)))

    def normalize(self, eta = 10):
        """ 
        .. math::
            x^{(i)} := \frac{x^{(i)} - mean(x^{(i)})}{\sqrt{var(x^{(i)})}+\eta_{norm}}, \forall i
        """
        results, updates = theano.scan(
                fn = lambda x_i: (x_i - T.mean(x_i)) / T.sqrt(T.var(x_i) + eta),
                sequences = [self.input]
                )
        return results

    def whiten_inputs(self, S, eps_zca = 0.01):
        """
        .. math::
            \[V, D\] := eig(cov(x))
            x^{(i)} := V(D + \eta_{zca}I)^{-1/2}V^Tx^{(i)}, \forall i
        """

        # S = T.dmatrix('S')
        # x_mean = T.mean(self.input, axis = 0)
        # results, updates = theano.scan(
                # fn = lambda S, x_i: S + 1./self.input.shape[0] * T.outer((x_i - x_mean), (x_i - x_mean)),
                # outputs_info = T.zeros_like(T.outer(x_mean, x_mean))
                # sequences = [self.input]
                # )
        # S = results[-1]
        eigenvalues, eigenvectors = T.nlinalg.eig(S)
        V = eigenvectors
        D = T.nlinalg.diag(eigenvalues)
        I = T.identity_like(D)
        inverted_squared = T.nlinalg.matrix_inverse(T.sqrt(D + eps_zca * I))
        results, updates = theano.scan(
                fn = lambda x_i: T.dot(T.dot(T.dot(V, inverted_squared), V.T), x_i),
                sequences = [self.input]
                )
        return results

def train(dataset = 'cifar-10-python.tar.gz', n_classes = 500, max_iter = 10):

    datasets = load_data(dataset)
    train_set_x = datasets[0][0]

    # TODO: rescale to 12 x 12 pixel images

    print('... building the model')

    X = T.dmatrix('X')
    classifier = K_Means(n_dim = 32 * 32, n_samples = 30000, k = n_classes, input = X)

    S = T.dmatrix('S')

    normalizer = theano.function(
            inputs = [X],
            outputs = classifier.normalize()
            )

    whitener = theano.function(
            inputs = [X, S],
            outputs = classifier.whiten_inputs(S)
            )

    train_model = theano.function(
            inputs = [X],
            outputs = classifier.cost,
            updates = [
                (classifier.S, classifier.S_update),
                (classifier.D, classifier.D_norm)
                ]
            )

    print('... normalize input')
    normalized_data = normalizer(train_set_x)

    print('... whiten input')
    whitend_data = whitener(normalized_data, np.cov(normalized_data.T)).T

    print('... training the model')
    for iter in range(max_iter):
        cur_cost = train_model(whitend_data)
        print('iteration %d/%d, cost %.2f ' % (iter + 1, max_iter, cur_cost))

    with open(os.path.join(os.path.split(__file__)[0], 'best_model.pkl'), 'wb') as f:
        pickle.dump(classifier, f)

def plot():
    print('... ploting the receptive fields')
    classifier = pickle.load(open(os.path.join(os.path.split(__file__)[0], 'best_model.pkl')))
    f_pca, axes  = plt.subplots(20, 25, subplot_kw = {'xticks': [], 'yticks': []})
    centroids = []
    for i in range(classifier.k):
        centroids.append(np.reshape(np.array(classifier.D.get_value())[:,i], (32, 32)))

    for ax, centroid in zip(axes.flat, centroids):
        ax.imshow(centroid, cmap=plt.cm.gray, interpolation = 'none')

    plt.savefig(os.path.join(os.path.split(__file__)[0], 'repflds.png'), format = 'png')

def main(argv):

    if len(argv) < 1:
        print("please call with at least 1 argument")
        return -1

    command = argv[0]

    if command == 'train':
        return train()

    elif command == 'plot':
        return plot()
    else: 
        print('unknown command: %' % command) 
        print("either use 'train' or 'plot'") 
        return -1

if __name__ == "__main__": 
    sys.exit(main(sys.argv[1:]))
