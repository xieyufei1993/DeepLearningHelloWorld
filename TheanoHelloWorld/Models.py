# coding: utf-8
import theano
import theano.tensor as T
import numpy as np
import cPickle
from Layers import MLP
from updates import sgd


class MLPModel(object):
    def __init__(self, rng, n_in=28*28, n_hidden=500, n_out=10, params=None):
        print('... building the model')
        # allocate symbolic variables for the data
        x = T.matrix('x')    # the data is presented as rasterized images
        y = T.ivector('y')   # the labels are presented as 1D vector of [int] labels

        # construct the MLP class
        self.nn_classifier = MLP(rng=rng, input=x, n_in=n_in, n_hidden=n_hidden, n_out=n_out,  params=params)

        # the cost we minimize during training is the negative log likelihood of
        # the model.
        # We take the mean of the cost over each minibatch.
        cost = self.nn_classifier.logRegressionLayer.negative_log_likelihood(y)

        # L2 rate
        # l2_rate = np.float32(1e-5)
        # for param in nn_classifier.params:
        #    cost += T.sum(l2_rate*(param*param), dtype='float32')

        # compute the gradient of cost with respect to theta (stored in params)
        # the resulting gradients will be stored in a list gparams
        # and use different optimization algorithm ("sgd", "adagrad", "rmsprop", "adadelta", "adam")
        # gparams = [T.grad(cost, param) for param in nn_classifier.params]
        updates = sgd(cost, self.nn_classifier.params, 0.8)

        # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch
        self.train_model = theano.function(
            inputs=[x, y],
            updates=updates,
            outputs=cost
        )

        self.validate_model = theano.function(
            inputs=[x, y],
            outputs=self.nn_classifier.logRegressionLayer.errors(y)
        )

        self.test_model = theano.function(
            inputs=[x, y],
            outputs=self.nn_classifier.logRegressionLayer.errors(y)
        )

        self.predict_model = theano.function(
            inputs=[x],
            outputs=self.nn_classifier.logRegressionLayer.predict()
        )

    def fit(self, x, y):
        return self.train_model(x, y)

    def validate_test(self, x, y):
        return self.validate_model(x, y)

    def predict(self, x):
        return self.predict_model(x)

    def test(self, x, y):
        return self.test_model(x, y)

    def evaluate(self, x, y, n_x, batch_size):
        error_cnt_list = [self.test(
            x[i * batch_size: (i + 1) * batch_size],
            y[i * batch_size: (i + 1) * batch_size]) for i in range(n_x//batch_size)]
        error_rate = np.sum(error_cnt_list, dtype='float32') / n_x
        return error_rate

    @staticmethod
    def save_params(file_path):
        obj = dict()
        for layer in MLP.layers:
            obj[layer.name] = layer.params

        with open(file_path, 'wb') as fw:
            cPickle.dump(obj, fw, protocol=cPickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_params(file_path):
        with open(file_path, 'rb') as fr:
            params = cPickle.load(fr)
        return params
