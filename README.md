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

The code for multiclass logisitc regression on mnist with minibaches and gradient descent can be found under [logreg/logistic_regression.py](logreg/logistic_regression.py). I did follow the tutorial for a mayor part of development. This implementation already solves several other problems and uses climin. To get an insight in the development process, please refer to `git log`.

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
| Adam                          | 152.0 s        | 6.81 %               | 7.14 %         |
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
$ python nn/neural_net.py tanh train gd # or other activation (tanh/sigmoid/relu) function / optimizer (gd/rmsprop)
$ python nn/neural_net.py tanh plot repflds
$ python nn/neural_net.py tanh plot error
```

**Problem 14:** Implement a neural network with one hidden layer. We suggest that you implement it in a way that it works with different kinds of optimization algorithms. Use stochastic gradient descent with mini-batches and rmsprop as the initial optimization method. Implement early stopping. You may follow the tutorial or, better, write your own implementation from ground up.

The code for a multilayer perceptron on mnist with minibaches and gradient descent / rmsprop can be found under [nn/neural_net.py](nn/neural_net.py). I did follow the tutorial for a mayor part of development. To get an insight in the development process, please refer to `git log`.

**Problem 15:** Evaluate your implementation on MNIST. Initially use 300 hidden units with tanh activation functions.

**Problem 16:** Try different nonlinear activation functions for the hidden units. Evaluate logistic sigmoid, tanh and rectified linear neurons in the hidden layer. Think about how the different activation functions look like and how they behave. Does—and if it does, how does—this influence, e.g., weight initialization or data preprocessing? Implement and test your reasoning in your code to see if the results support your conclusions.

**Problem 17:** Plot the error curves for the training, evaluation and test set for each of the activation functions evaluated in the previous problem into file error.png. That is, either provide one file with three subplots (one per activation function) and three error curves each, or provide three different files ([error_tanh.png](nn/error_tanh.png), [error_sigmoid.png](nn/error_sigmoid.png), and [error_relu.png](nn/error_relu.png)).

![error tanh](nn/error_tanh.png)
![error sigmoid](nn/error_sigmoid.png)
![error relu](nn/error_relu.png)

**Problem 18:** Visualize the receptive fields *of the hidden* layer and write them to file [repflds.png](nn/repflds_tanh.png). As in the previous problem, either provide one file with three subplots, or three distinct files([repflds_tanh.png)](nn/repflds_tanh.png), [repflds_sigmoid.png)](nn/repflds_sigmoid.png), [repflds_relu.png)](nn/repflds_relu.png)).

![receptiva fields](nn/repflds_tanh.png)
![receptiva fields](nn/repflds_sigmoid.png)
![receptiva fields](nn/repflds_relu.png)


**Problem 19:**  Fine tune your implementation until you achieve an error rate of about 2%. Optionally try augmenting the training set as described in section [2](Multiclass Logistic Regression). Do not spend too much time on this problem.

| Activation function           | Runtime        | Best validation loss | Best test loss |
| :---------------------------- | :------------- | :------------------- | :------------- |
| Hyperbolic Tangens            | 445.2 m        | 1.67 %               | 1.81 %         |
| Logistic Sigmoid              | 46.3 m         | 2.01 %               | 2.05 %         |
| Rectified Linear Neurons      | 6.2 m          | 1.77 %               | 1.76 %         |

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

**Problem 20:** Implement PCA in Python using Theano. Write your own implementation from ground up.

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

**Problem 21:** Produce a PCA scatterplot (see http://peekaboo-vision.blogspot.de/2012/12/another-look-at-mnist.html) on MNIST, also do this on CIFAR-10. Write them to file [scatterplotMNIST.png](latent/scatterplotMNIST.png) and [scatterplotCIFAR.png](latent/scatterplotCIFAR.png) respectively.

![pca on MNIST](latent/scatterplotMNIST.png)
![pca on CIFAR-10](latent/scatterplotCIFAR.png)

**Problem 22:** Implement an autoencoder in Python using Theano. Train the network using the squared error loss L(x) = ||f (x) − x||2 where x is a data sample and f (x) is the output of the autoencoder. You may follow a part of the tutorial or, better, write your own implementation from ground up. If training is difficult using gradient descent, try using the RMSprop optimizer.



**Problem 23:** Increase the number of hidden units, but add a sparsity constraint on the hidden units. This means that the network should be encouraged to have most hidden units close to zero for a sample from the data set. This can be done by adding an L1 penalty (see literature section for reasons behind this) to the loss function, for example Lsparse(x) = L(x) + λ|h(x)|1 where h(x) denotes the hidden layer values for sample x and |z|1 =  i |zi| is the L1-norm of z. λ > 0 is a new hyperparameter that determines the trade-off between sparsity and reconstruction error.

**Problem 24:** Train the sparse autoencoder on MNIST. Write the reconstructions (i.e. outputs of the autoencoder) of the first 100 samples from the test set of MNIST into file [autoencoderrec.png](latent/autoencoderrec.png). Adjust λ and see how it affects the reconstructions.

![receptive fields sparse autoencoder](latent/autoencoderrec.png)

**Problem 25:** Visualise the learnt receptive fields (weights of the first layer). Write them to file [autoencoderfilter.png](latent/autoencoderfilter.png). Adjust λ and see how it affects the receptive fields.

![receptive fields sparse autoencoder](latent/autoencoderfilter.png)


**Problem 26:** Explain the meaning of a sparse encoding of MNIST.

**Bonus problem:** Replace the sparsity-inducing *L_1* penalty by a KL-divergence penalty between the data- induced distribution of the hidden units and Bernoulli random variables with a low (p < 0.05) success probability. This is described in http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity.

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
**Problem 30:** Implement k-Means in Python using Theano. Follow the above paper by Adam Coates and implement the steps on page 5 (skip everything from section 4 on).

**Problem 31:** Train your model on the CIFAR-10 dataset and visualise your receptive fields. Save them to file [repflds.png](kmeans/repflds.png). Make sure to rescale the images in the dataset from 32 × 32 to 12 × 12 pixels and choose no more than 500 centres.

![repfields](kmeans/repflds.png)

**Bonus problem:** Implement minibatch k-Means.

**Bonus problem:** Implement the k-Means version from the paper B. Kulis, M. Jordan: Revisiting k- means: New Algorithms via Bayesian Nonparametrics (http://arxiv.org/pdf/1111.0352.pdf). Apply it on a (rather large) set of of colour images and try to determine a good colour palette.
