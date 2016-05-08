#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 romancpodolski <romancpodolski@Romans-MacBook-Pro-2.local>
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

    def __init__(self, k = 500, input = None):
        """Instance of k-Means object
        """
        self.k = k 
        self.input = input

    def normalize(self, eta):
        """ 
        .. math::
            x^{(i)} := \frac{x^{(i)} - mean(x^{(i)})}{\sqrt{var(x^{(i)})}+\eta_{norm}}, \forall i
        """
        results, updates = theano.scan(
                fn = lambda x_i: (x_i - T.mean(x_i)) / T.sqrt(T.var(x_i) + eta),
                sequences = [self.input]
                )
        return results

    def whiten_inputs(self, S, eta = 0.01):
        """
        .. math::
            \[V, D\] := eig(cov(x))
            x^{(i)} := V(D + \eta_{zca}I)^{-1/2}V^Tx^{(i)}, \forall i
        """
        eigenvalues, eigenvectors = T.nlinalg.eig(S)
        V = eigenvectors
        D = T.nlinalg.diag(eigenvalues)
        results, updates = theano.scan(
                fn = lambda x_i: T.dot(T.dot(T.dot(V, D + eta_zca*T.identity_like(D))^(-0.5), V.T), x_i),
                sequences = [self.input]
                )
        return results


def train(dataset = 'cifar-10-python.tar.gz', n_classes = 500):

    datasets = load_data(dataset)
    train_set_x = datasets[0][0]

    print('... building the model')

    X = T.dmatrix('X')
    classifier = K_Means(k = n_classes, input = X)

    S = T.dmatrix('S')

    normalizer = theano.function(
            inputs = [X],
            outputs = classifier.normalize(10)
            )

    whitener = theano.function(
            inputs = [X, S],
            outputs = classifier.whiten_inputs(S)
            )

    print('... normalize input')
    normalized_data = normalizer(train_set_x)

    print('... whiten input')
    whitend_data = whitener(normalized_data, np.cov(normalized_data.T))

    print('... training the model')
    with open(os.path.join(os.path.split(__file__)[0], 'best_model.pkl'), 'wb') as f:
        pickle.dump(classifier, f)

def plot():
    classifier = pickle.load(open(os.path.join(os.path.split(__file__)[0], 'best_model.pkl')))

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
