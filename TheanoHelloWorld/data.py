# coding: utf-8
import gzip
import cPickle
import numpy as np
import theano
import theano.tensor as T


def load_data(data_set_path='mnist.pkl.gz', share=True):
    """Loads the dataset
    This code is downloaded from
    http://deeplearning.net/tutorial/code/logistic_sgd.py
    :type data_set_path: string
    :param data_set_path: the path to the dataset (here MNIST)

    MNIST: a handwritten number(0-9) picture data set: 50,000(train) + 10,000(valid) + 10,000(test)
        each of picture is  32*32
    """

    #############
    # LOAD DATA http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    #############
    # Load the data_set_path
    with gzip.open(data_set_path, 'rb') as f:
            train_set, valid_set, test_set = cPickle.load(f)
    print('data loaded')

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_data(data_xy, borrow=True):
        """ Function that loads the data_set_path into shared variables
        The reason we store our data_set_path in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = (shared_data(test_set) if share else test_set)
    valid_set_x, valid_set_y = (shared_data(valid_set) if share else valid_set)
    train_set_x, train_set_y = (shared_data(train_set) if share else train_set)
    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]

