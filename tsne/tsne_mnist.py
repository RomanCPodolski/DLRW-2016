#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 romancpodolski <romancpodolski@Romans-MBP-2.fritz.box>
#
# Distributed under terms of the MIT license.

"""

"""

from __future__ import print_function

import six.moves.cPickle as pickle
import numpy as np
import timeit
import os
import sys

sys.path.append(os.path.join(os.path.split(__file__)[0], 'bh_tsne'))

from bhtsne import bh_tsne

sys.path.append(os.path.join(os.path.split(__file__)[0],'..' , 'data'))

from data import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

def scale_to_unit_interval(ndar, eps = 1e-8):
    ndar = ndar.copy()
    ndar = ndar.min()
    ndar *= 1. / (ndar.max() + eps)
    return ndar

def train(dataset = 'mnist.pkl.gz'):
    dataset = load_data(dataset)
    data = dataset[0][0].astype('float64')

    start_time = timeit.default_timer()

    results = np.zeros((data.shape[0], 2))
    print('... training barnes-Hut tsne')
    for res, save in zip(bh_tsne(np.copy(data), theta = 0.5), results):
        save[...] = res

    end_time = timeit.default_timer()
    print(('The code for file  ' + os.path.split(__file__)[1] + ' ran for %.2fs' % (end_time - start_time)), file = sys.stderr)

    with open(os.path.join(os.path.split(__file__)[0], 'data.pkl'), 'wb') as f:
        pickle.dump(results, f)

    results = results - np.min(results, axis = 0)
    results = results / np.max(results, axis = 0)

def plot(dataset = 'mnist.pkl.gz'):
    print('... plotting the results')
    dataset = load_data(dataset)
    data = dataset[0][0].astype('float64')
    results = pickle.load(open(os.path.join(os.path.split(__file__)[0], 'data.pkl')))

    results = results - np.min(results, axis = 0)
    results = results / np.max(results, axis = 0)

    out = np.zeros((8000, 8000), dtype = 'uint8')
    out[...] = 255
    
    for i in xrange(data.shape[0]):
        xpos = int(results[i][0] * (8000 - 1000) + 500)
        ypos = int(results[i][1] * (8000 - 1000) + 500)
        pic = scale_to_unit_interval(data[i].reshape((28, 28)))
        out[xpos:xpos + 28, ypos: ypos + 28] = pic * 255

    print('... saving to file ' + os.path.join(os.path.split(__file__)[0], 'tsne_mnist.png'))

    image = Image.fromarray(out)
    image.save(os.path.join(os.path.split(__file__)[0], 'tsne_mnist.png'))

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

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
