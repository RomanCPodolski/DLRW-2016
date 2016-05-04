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
from __future__ import print_function
import theano
import theano.tensor as T

import numpy as np
from itertools import product

import sys
import os

sys.path.append(os.path.join(os.path.split(__file__)[0], '..', 'data'))
from data import load_data
from data import load_cifar

import matplotlib.pyplot as plt

def test_pca(dataset='mnist.pkl.gz'):
    
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]

    s = T.matrix('S')
    x = T.matrix('x')

    eigenvalues, eigenvectors = T.nlinalg.eig(s)
    projection = T.dot(x, eigenvectors[:, :2])
    pca = theano.function(
            inputs = [x, s],
            outputs = projection
            )

    f_pca, axes  = plt.subplots(10, 10)
    f_pca.set_size_inches(10, 10)
    plt.prism()

    for i, j in product(xrange(10), repeat = 2):
        if i > j:
            continue

        X_ = train_set_x[(train_set_y == i) + (train_set_y == j)]
        y_ = train_set_y[(train_set_y == i) + (train_set_y == j)]

        cov = np.cov(X_.T)
        projected_set = pca(X_, cov)

        axes[i,j].scatter(projected_set[:,0], projected_set[:,1], s = 1, c = y_, edgecolors = 'none')
        axes[i,j].set_xticks(())
        axes[i,j].set_yticks(())

        axes[j,i].scatter(projected_set[:,0], projected_set[:,1], s = 1, c = y_, edgecolors = 'none')
        axes[j,i].set_xticks(())
        axes[j,i].set_yticks(())

        if i == 0:
            axes[i,j].set_title(j)
            axes[j,i].set_ylabel(j)

    plt.tight_layout()
    if dataset == 'mnist.pkl.gz':
        plt.savefig('scatterplotMNIST.png', format = 'png', dpi = 500)
    elif(dataset == 'cifar-10-python.tar.gz'):
        plt.savefig('scatterplotCIFAR.png', format = 'png', dpi = 5000)

if __name__ == "__main__":
    test_pca(dataset = 'mnist.pkl.gz')
    test_pca(dataset = 'cifar-10-python.tar.gz')
