"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

from __future__ import print_function

import six.moves.cPickle as pickle
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

sys.path.append(os.path.join(os.path.split(__file__)[0], '..', 'data'))
from data import load_data
from utils import tile_raster_images

# from utils import tile_raster_images

import climin as cli
import climin.initialize as init
import climin.util
import itertools

try:
    import PIL.Image as Image
except ImportError:
    import Image


class dA(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None,
        corruption_level = 0.,
        sparsity_lambda = 0,
        learning_rate = 0.13
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

        self.tilde_x = self.get_corrupted_input(self.x, corruption_level)
        self.y = self.get_hidden_values(self.tilde_x)
        self.z = self.get_reconstructed_input(self.y)
        self.L = - T.sum(self.x * T.log(self.z) + (1 - self.x) * T.log(1 - self.z), axis=1)
        self.L1_hidden_penalty = sparsity_lambda * self.y.mean(axis = 0).sum()
        self.cost = T.mean(self.L) + self.L1_hidden_penalty

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        self.gparams = T.grad(self.cost, self.params)

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level, dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    # def get_cost_updates(self, corruption_level,sparsity_lambda, learning_rate):
        # """ This function computes the cost and the updates for one trainng
        # step of the dA """

        # tilde_x = self.get_corrupted_input(self.x, corruption_level)
        # y = self.get_hidden_values(tilde_x)
        # z = self.get_reconstructed_input(y)
        # # note : we sum over the size of a datapoint; if we are using
        # #        minibatches, L will be a vector, with one entry per
        # #        example in minibatch
        # L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # # note : L is now a vector, where each element is the
        # #        cross-entropy cost of the reconstruction of the
        # #        corresponding example of the minibatch. We need to
        # #        compute the average of all these to get the cost of
        # #        the minibatch
        # L1_hidden_penalty = sparsity_lambda * y.mean(axis = 0).sum()
        # cost = T.mean(L) + L1_hidden_penalty

        # # compute the gradients of the cost of the `dA` with respect
        # # to its parameters
        # gparams = T.grad(cost, self.params)
        # # generate the list of updates
        # # updates = [
            # # (param, param - learning_rate * gparam)
            # # for param, gparam in zip(self.params, gparams)
        # # ]

        # return (cost, gparams)


def train(learning_rate=0.1, training_epochs=15, dataset='mnist.pkl.gz', batch_size=600, n_hidden = 800, optimizer = 'gd', sparsity = 0.08, corruption = 0.3):
    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y   = datasets[2]

    # create climin templates
    tmpl = [(28 * 28, n_hidden), n_hidden, 28 * 28]
    flat, (Weights, bias_hidden, bias_visible) = climin.util.empty_with_views(tmpl)
    climin.initialize.randomize_normal(flat, 0, 0.1)

    print('... building the model')

    # start-snippet-2
    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    # end-snippet-2

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible = 28 * 28,
        n_hidden = n_hidden,
        W = theano.shared(value = Weights, name = 'W', borrow = True),
        bvis = theano.shared(value = bias_visible, name = "b'", borrow = True),
        bhid = theano.shared(value = bias_hidden, name = 'b', borrow = True),
        corruption_level = corruption,
        sparsity_lambda = sparsity
    )

    loss = theano.function([x], da.cost)
    g_params_da = theano.function([x], da.gparams)

    def d_loss_wrt_pars(parameters, inputs, targets):
        g_W, g_b_hidden, g_b_vis,  = g_params_da(inputs)
        return np.concatenate([g_W.flatten(), g_b_hidden, g_b_vis])

    if batch_size is None:
        print('... training on the full train set')
        args = itertools.repeat(([train_set_x, train_set_y], {}))
        batches_per_pass = 1
    else:
        n_train_batches = train_set_x.shape[0] // batch_size
        print('... training on %d minibatches of size %d each' % (n_train_batches, batch_size))
        args = cli.util.iter_minibatches([train_set_x, train_set_y], batch_size, [0, 0])
        args = ((i, {}) for i in args)

    if optimizer == 'gd':
        print('... using gradient descent')
        opt = cli.GradientDescent(flat, d_loss_wrt_pars, step_rate = learning_rate, momentum = .95, args=args)
    elif optimizer == 'rmsprop':
        print('... using rmsprop')
        opt = cli.RmsProp(flat, d_loss_wrt_pars, step_rate = learning_rate, decay = 0.9, args=args)
    else:
        print('unknown optimizer')
        return -1

    ############
    # TRAINING #
    ############
    start_time = timeit.default_timer()
    # go through training epochs
    for info in opt:
        iter = info['n_iter'] - 1
        epoch = iter // n_train_batches + 1

        if (iter + 1) % n_train_batches == 0:
            print('training epoch %d, cost %f ...' % (epoch, loss(train_set_x)))

        if epoch >= training_epochs:
            break

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print(('The no corruption code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)

    with open(os.path.join(os.path.split(__file__)[0], 'autoencoder.pkl'), 'wb') as f:
        pickle.dump(da, f)
    return 1

def plot(element, dataset = 'mnist.pkl.gz'):
    autoencoder = pickle.load(open(os.path.join(os.path.split(__file__)[0], 'autoencoder.pkl')))
    if element == 'reconstructions':
        print('... plot reconstructions')

        datasets = load_data(dataset)
        test_set_x, test_set_y   = datasets[2]

        rec = theano.function([autoencoder.x], T.nnet.sigmoid(T.dot(autoencoder.y, autoencoder.W_prime) + autoencoder.b_prime))
        image = Image.fromarray(tile_raster_images(X = rec(test_set_x[:100]), img_shape = (28, 28), tile_shape= (5, 20), tile_spacing=(1, 1)))
        image.save(os.path.join(os.path.split(__file__)[0], 'autoencoderrec.png'))
    elif element == 'repflds':
        print('... plot receptive fields')
        image = Image.fromarray(tile_raster_images(X=autoencoder.W.get_value(borrow = True).T, img_shape = (28, 28), tile_shape = (5, 20), tile_spacing=(1, 1)))
        image.save(os.path.join(os.path.split(__file__)[0],'autoencoderfilter.png'))
    else:
        print("dot't know how to plot %" % element) 
        print("either use 'reconstructions' or 'repflds'") 
        return -1

def main(argv):
    if len(argv) < 1:
        print("please call with at least 1 argument")
        return -1

    command = argv[0]

    if command == 'train':
        return train()

    elif command == 'plot':
        if len(argv) < 2:
            print('please define what element to plot')
            return -1

        return plot(argv[1])
    else: 
        print('unknown command: %' % command) 
        print("either use 'train' or 'plot'") 
        return -1

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
