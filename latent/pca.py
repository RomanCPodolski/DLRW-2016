#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 romancpodolski <romancpodolski@Romans-MBP-2.home>
#
# Distributed under terms of the MIT license.

"""
Principal Component Analysis

References:
    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 12.1

                 "Machine Learning A Propabilistic Perspective" -
                 Kevin P. Murpy, 28.3.2, 28.3.3, 28.4.2, 28.4.3

"""
import theano
import theano.tensor as T

import numpy as np

import sys
import os

sys.path.append(os.path.join(os.path.split(__file__)[0], '..', 'data'))
from data import load_data

import matplotlib.pyplot as plt


class PCA(object):
    """Principal Component Analysis Class

    """

    def __init__(self,input, n_components):
        """Initialize the parameters for the principal component analysis

        :type n_components: int
        :param n_components: dimensionality of feature space after PCA
        """

        self.n_components = n_components

        self.mean_of_x = T.mean(input, axis = 0) # calculate the mean along all axis
        
        self.cov_of_x = 1. / input.shape[0] * T.sum(T.dot(( input - mean ).T, input - mean), axis = 1)

        self.eig = T.nlinalg.eig(self.cov_of_x)

        print("PCA class created with %d n_components" % self.n_components)

    def transform(data):
        """Transorm A dataset with dimensionality M > n_components.

        :type data: numpy.array
        :param data: the dataset to transform
        
        """

def test_pca(dataset='mnist.pkl.gz'):
    
    datasets = load_data(dataset, shared = False)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    pca = PCA(n_components = 2)

if __name__ == "__main__":
    test_pca(dataset='mnist.pkl.gz')
    test_pca(dataset='cifar-10-python.tar.gz')
