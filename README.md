# 2016-DLRW-Podolski
Individual student repository for the 2016 Summer semester course Deep Learning in the Real World.

## Running the code

This repository uses [Rake](https://github.com/ruby/rake) (a ruby implementation of Make) as a build tool.
To run the code and build the plots execute

```
$ rake
```

Warning: This could take a while ...

Or you could just execute the scripts manually using python.

## Multiclass logistic regression

To run the code, do
```
$ rake logreg
```
or

```
$ python logreg/logistic_regression.py train gd # or some other optimizer
$ python logreg/logistic_regression.py plot repflds
$ python logreg/logistic_regression.py plot error
```

**Problem 8:** Implement multiclass logistic regression in Python using Theano. Use standart gradient descent with minibatches as the initial optimization method. You may follow the tutorial, or, better, write your own implementation from ground up.

The code for multiclass logisitc regression on mnist with minibaches and gradient descent can be found under [logreg/logistic_regression.py](logreg/logistic_regression.py). I did follow the tutorial for a mayor part of development. This implementation already solves several other problems and uses climin. To get an insight in the development process, please revere to `git log`.

**Problem 9:** Evaluate your implementation to classify handwritten digits from the MNIST dataset.

After training logisitc regression, one can use

```
$ rake logreg:predict
```
or
```
$ python logreg/logistic_regression.py predict
```

to predict some digits from the mnist dataset.

**Problem 10:** Try improving your results by using better optimization methods. You can use readily available optimizer from the Climin library.

[logreg/logistic_regression.py](logreg/logistic_regression.py) implements
* Gradient descent (`gd`)
* Non-Linear Conjugate Gradient (`nlcg`)
* RMSPROP (`rmsprop`)
* Resilient Propagation (`rprop`)
* Adam (`adam`)
* Adadelta (`adadelta`)

Use them by calling `logreg/logistic_regression.py train ` with the optimizer as argument, i.e.

```
$ python logreg/logistic_regression.py train gd # or some other optimizer
```

**Problem 11:** Visualize the receptive fields of each class (i.e., digit) and write them to file [repflds.png](logreg/repflds.png). In logistic regression the receptive fields are the weight matrices for each class.

![receptive fields of logistic regression mnist](logreg/repflds.png)

**Problem 12:** Plot the error curves (i.e., error over iteration) for the training, evaluation and test set into file [error.png](logreg/error.png). When do you stop training?

![error](logreg/error.png)

I plotted only the validation and test loss, since omitting calculating the training loss resulted in a very significant reduction in time needed to train the model. The train loss can easily added to the plot, if required.

Training of the model is stopped if the improvement of the errors loss over a number validation steps is not considered significant. The number of validations to wait until stoppage is increased, based on the iteration, in which a significant improvement appeared, times a freely-chosen factor. Regardless of improvement of error loss look at a minimum number of training data regardless.

**Problem 13:** Fine tune your implementation until you achieve an error rate of about 7% on the test set. One approach might be to augment the training set by including transformed (slightly rotated, elastically deformed) versions of the digits from the vanilla training set. Do not spend too much time on this problem, since we have not verified that this error rate is achievable using logistic regression.

| Optimizer                     | Runtime        | Best validation loss | Best test loss |
| :---------------------------- | :------------- | :------------------- | :------------- |
| Gradient descent              | 34.3 s         | 7.07 %               | 7.48 %         |
| RMSPROP                       | 250.5 s        | 6.75 %               | 7.46 %         |
| Resilient Propagation         | 35.6 s         | 12.15 %              | 12.55 %        |
| Adam                          | 152.0          | 6.81 %               | 7.14 %         |
| Adadelta                      | 32.8 s         | 7.66 %               | 7.85 %         |


**Bonus Question:** Why is the last problem statement actually very bad scientific practice?

> Acquiring enough labeled data to train such models is difficult[...]. In simple settings, such as hand-written character recognition, it is possible to generate lots of labeled data by making modified copies of a small manually labeled training set [...], but it seems unlikely that this approach will scale to complex scenes.
>
> Chapter 28.2 Machine Learning A probabilistic perspective - Kevin P. Murphy

Further, including modified data in the training set, can propagate unnatural characteristics in the dataset. If this feature is significant enough in a class, it is possible that the weigh of this feature is strengthen over a reasonable amount. A better approach would be to use a generative model. 

## Two-layer neural network
To run the code, do
```
$ rake nn
```
or

```
$ python logreg/logistic_regression.py tanh train
$ python logreg/logistic_regression.py tanh plot repflds
$ python logreg/logistic_regression.py tanh plot error
$ python logreg/logistic_regression.py sigmoid train
$ python logreg/logistic_regression.py sigmoid plot repflds
$ python logreg/logistic_regression.py sigmoid plot error
$ python logreg/logistic_regression.py relu train
$ python logreg/logistic_regression.py relu plot repflds
$ python logreg/logistic_regression.py relu plot error
```

**Problem 14:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Bonus Question:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Problem 15:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Problem 16:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Problem 17:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

![error](nn/error_tanh.png)
![error](nn/error_sigmoid.png)
![error](nn/error_relu.png)
**Problem 18:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

![receptiva fields](nn/repflds_tanh.png)
![receptiva fields](nn/repflds_sigmoid.png)
![receptiva fields](nn/repflds_relu.png)
**Problem 19:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

### PCA and sparse autoencoder

To run the code, do
```
$ rake latent
```
or

```
$ python latent/pca.py
$ python latent/dA.py train
$ python latent/dA.py plot
```

**Problem 20:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Problem 21:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Problem 22:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Problem 23:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Problem 24:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Problem 25:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Problem 26:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Bonus Question:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

![pca on MNIST](latent/scatterplotMNIST.png)
![pca on CIFAR-10](latent/scatterplotCIFAR.png)
![receptive fields sparse autoencoder](latent/autoencoderfilter.png)
![receptive fields sparse autoencoder](latent/autoencoderrec.png)
## t-SNE

To run the code, do
```
$ rake tsne
```
or

```
$ python tsne/tsne_mnist.py train
$ python tsne/tsne_mnist.py plot
```

**Problem 27:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Problem 28:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

![embeddings](tsne/embeddings.png)
**Problem 29:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

![tsne on mnist](tsne/tsne_mnist.png)
## k-Means

To run the code, do
```
$ rake kmeans
```
or

```
$ python kmeans/kmeans.py train
$ python kmeans/kmeans.py plot
```
**Problem 30:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Problem 31:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Bonus Question:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Bonus Question:** Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

![repfields](kmeans/repflds.png)
