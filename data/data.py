#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 romancpodolski <romancpodolski@Romans-MacBook-Pro.local>
#
# Distributed under terms of the MIT license.
import six.moves.cPickle as pickle
import gzip
import tarfile
import os
import sys

import theano
import theano.tensor as T
import numpy

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    it is needed (the default behaviour if the data is not a shared
    variable) would lead to a large decrease in performance
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
        dtype=theano.config.floatX), borrow=borrow)

    shared_y = theano.shared(numpy.asarray(data_y,
        dtype=theano.config.floatX), borrow=borrow)

    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_x`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesnt make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    # return shared_x, T.cast(shared_y, 'int32')
    return shared_x, shared_y

def load_data(dataset, shared = False):
    if dataset == 'cifar-10-python.tar.gz':
        rv = load_cifar(dataset, shared)
    elif dataset == 'mnist.pkl.gz':
        rv = load_mnist(dataset, shared)

    return rv

def load_cifar(dataset, shared = False):
    """Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here cifar)
    """

    ###############
    #  LOAD DATA  #
    ###############
    
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
                os.path.split(__file__)[0],
                dataset
                )
        if os.path.isfile(new_path) or data_file == 'cifar-10-python.tar.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'cifar-10-python.tar.gz':
        from six.moves import urllib
        origin = (
                'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
                )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading cifar-10 grayscale')

    with tarfile.open(dataset, 'r') as f:
        try:
            batch_1 = pickle.load(f.extractfile(f.getmember('cifar-10-batches-py/data_batch_1')))
            batch_2 = pickle.load(f.extractfile(f.getmember('cifar-10-batches-py/data_batch_2')))
            batch_3 = pickle.load(f.extractfile(f.getmember('cifar-10-batches-py/data_batch_3')))
            batch_4 = pickle.load(f.extractfile(f.getmember('cifar-10-batches-py/data_batch_4')))
            batch_5 = pickle.load(f.extractfile(f.getmember('cifar-10-batches-py/data_batch_5')))
            batch_test = pickle.load(f.extractfile(f.getmember('cifar-10-batches-py/data_batch_5')))
            train_set = [numpy.concatenate((batch_1['data'][:,:1024], batch_2['data'][:,:1024], batch_3['data'][:,:1024])), numpy.asarray(batch_1['labels'] + batch_2['labels'] + batch_3['labels'])]
            valid_set = [numpy.concatenate((batch_4['data'][:,:1024], batch_5['data'][:,:1024])), numpy.asarray(batch_4['labels'] + batch_5['labels'])]
            test_set = [batch_test['data'][:,:1024], numpy.asarray(batch_test['labels'])] 
        except:
            train_set, valid_set, test_set = pickle.load(f)

    if shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
    else:
        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval

def load_mnist(dataset, shared = False):
    """Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MINST)
    """

    ###############
    #  LOAD DATA  #
    ###############
    
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
                os.path.split(__file__)[0],
                dataset
                )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
                'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
                )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading mnist')

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target 
    # to the example whith the same index in the input.

    if shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
    else:
        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set


    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval
