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
import scipy.misc

import matplotlib.pyplot as plt

import sys
import os
import six.moves.cPickle as pickle

sys.path.append(os.path.join(os.path.split(__file__)[0], '..', 'data'))
from data import load_data
from data import shared_dataset
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

class K_Means(object):

    def __init__(self, n_dim, n_samples, k = 500, input = None):
        """Instance of k-Means object

        :type n_dim: int
        :param n_dim: TODO

        :type n_samples: int
        :param n_samples: TODO

        :type k: int
        :param k: number of centroids used, columns in the Dictionary D
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
        # self.D_norm = D_update / D_update.norm(2)
        self.D_norm = D_update / T.sqrt(T.sum(T.sqr(D_update), axis = 0))

        self.cost = T.sum(T.sqrt(T.sum(T.sqr(T.dot(self.D, self.S) - self.input), axis = 0)))

def train(dataset = 'cifar-10-python.tar.gz', n_classes = 500, max_iter = 10, batch_size = 600):

    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    n_train_batches = train_set_x.shape[0] // batch_size

    print(batch_size)

    print('... resizing the input from 32x32 pixels to 12x12 to pixels')
    resized = []
    for i in xrange(train_set_x.shape[0]):
        resized.append(scipy.misc.imresize(np.reshape(train_set_x[i], (32, 32)), (12, 12)).flatten() )
    train_set_x = np.asarray(resized)

    print('... normalize input')
    epsilon = 10
    train_set_x = (train_set_x - np.mean(train_set_x, axis = 1)[:, np.newaxis]) / np.sqrt(np.var(train_set_x, axis = 1) + epsilon)[:, np.newaxis]

    print('... whiten input')
    d, V = np.linalg.eig(np.cov(train_set_x.T))
    D = np.diag(d)
    I = np.eye(d.shape[0])
    epsilon = 0.01
    train_set_x = np.dot(np.dot(np.dot(V, np.linalg.inv(np.sqrt(D + epsilon * I))), V.T), train_set_x.T)

    train_set = (train_set_x.T, train_set_y)

    train_set = shared_dataset(train_set)

    train_set_x = train_set[0]

    print('... building the model')

    x = T.dmatrix('X')

    classifier = K_Means(n_dim = 12 * 12, n_samples = batch_size, k = n_classes, input = x)
    
    updates = [(classifier.S, classifier.S_update),
               (classifier.D, classifier.D_norm)]

    index = T.lscalar('index')

    train_model = theano.function(
            inputs = [index],
            outputs = classifier.cost,
            updates = updates,
            givens = {
                x: train_set_x[index * batch_size: (index + 1) * batch_size].T,
                }
            )


    print('... training the model')
    for epoch in range(max_iter):
        cost = []
        for minibatch_index in range(n_train_batches):
            cost = train_model(minibatch_index)
            print('epoch %i, minibatch %i, %i, cost %.2f ' % (epoch + 1, minibatch_index + 1, n_train_batches, cost))


    with open(os.path.join(os.path.split(__file__)[0], 'best_model.pkl'), 'wb') as f:
        pickle.dump(classifier, f)

def plot():
    import scipy.ndimage
    classifier = pickle.load(open(os.path.join(os.path.split(__file__)[0], 'best_model.pkl')))

    print('... magnifying the receptive fields using bilinear interpolation to 36x36 pixels')
    resized = []
    for i in xrange(classifier.D.get_value(borrow = True).T.shape[0]):
        resized.append(scipy.ndimage.zoom(np.reshape(classifier.D.get_value(borrow = True).T[i], (12, 12)), 3, order = 1).flatten() )
    repflds = np.asarray(resized)
    
    print('... ploting the receptive fields with 36x36 pixels')
    image = Image.fromarray(tile_raster_images(X = repflds, img_shape = (36, 36), tile_shape = (10, 50), tile_spacing=(1, 1)))
    print('... saving image to %s' % os.path.join(os.path.split(__file__)[0], 'repflds.png'))
    image.save(os.path.join(os.path.split(__file__)[0], 'repflds.png'))

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
