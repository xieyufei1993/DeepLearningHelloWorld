# coding: utf-8
import numpy as np
import theano
import theano.tensor as T

from data import load_data
from Layers import MLP
from optimization import sgd


class MLPModel(object):
    def __init__(self, rng, dataset, batch_size=64, n_in=28*28, n_hidden=500, n_out=10):
        print('... building the model')
        self.batch_size = batch_size
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')    # the data is presented as rasterized images
        y = T.ivector('y')   # the labels are presented as 1D vector of [int] labels

        # construct the MLP class
        nn_classifier = MLP(rng=rng, input=x, n_in=n_in, n_hidden=n_hidden, n_out=n_out)

        # the cost we minimize during training is the negative log likelihood of
        # the model.
        # We take the mean of the cost over each minibatch.
        cost = nn_classifier.logRegressionLayer.negative_log_likelihood(y)

        # compute the gradient of cost with respect to theta (stored in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in nn_classifier.params]
        updates = sgd(nn_classifier.params, gparams, eta=0.9)
        # updates = ada_updates(nn_classifier.params, gparams)

        # compiling a Theano function that computes the mistakes that are made by
        # the model on a minibatch
        train_set_x, train_set_y = dataset[0]
        valid_set_x, valid_set_y = dataset[1]
        test_set_x, test_set_y = dataset[2]
        self.train_model = theano.function(
            inputs=[index],
            updates=updates,
            outputs=cost,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )

        self.validate_model = theano.function(
            inputs=[index],
            outputs=nn_classifier.logRegressionLayer.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.test_model = theano.function(
            inputs=[index],
            outputs=nn_classifier.logRegressionLayer.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.predict_model = theano.function(
            inputs=[index],
            outputs=nn_classifier.logRegressionLayer.y_pred,
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )

    def fit(self, index):
        return self.train_model(index)

    def validate_test(self, index):
        return self.validate_model(index)

    def predict(self, index):
        return self.predict_model(index)

    def test(self, n_test):
        cnt = 0.0
        for i in range(n_test//self.batch_size):
            cnt += self.test_model(i)
        print 'test error rate: {}'.format(cnt / n_test)


def train(rng=None, batch_size=64, n_epochs=10, n_in=28*28, n_hidden=500, n_out=10):
    dataset = load_data(data_set_path='mnist.pkl.gz', share=True)
    # compute number of mini batches for training, validation and testing
    n_train = dataset[0][0].get_value(borrow=True).shape[0]
    n_valid = dataset[1][0].get_value(borrow=True).shape[0]
    n_test = dataset[2][0].get_value(borrow=True).shape[0]
    n_train_batches = n_train // batch_size
    n_valid_batches = n_valid // batch_size
    print('data loaded ok.')

    mlp = MLPModel(rng, dataset, batch_size=batch_size, n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    print('model build ok.')

    validation_frequency = 30
    iter_num = 0
    best_validation_loss_rate = 1.0
    for epoch in xrange(n_epochs):
        mini_batch_avg_cost_list = []
        print 'epoch # : {}'.format(epoch)
        for mini_batch_index in range(n_train_batches):
            iter_num += 1
            mini_batch_avg_cost = mlp.fit(mini_batch_index)
            mini_batch_avg_cost_list.append(mini_batch_avg_cost)
            print 'cost: {}'.format(mini_batch_avg_cost)

            if iter_num % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [mlp.validate_test(i) for i in range(n_valid_batches)]
                this_validation_loss_rate = np.sum(validation_losses, dtype='float32') / n_valid
                if best_validation_loss_rate > this_validation_loss_rate:
                    best_validation_loss_rate = this_validation_loss_rate
                    print 'epoch {} get new best validation loss: {}'.format(epoch, best_validation_loss_rate)
                    if best_validation_loss_rate < 0.05:
                        mlp.test(n_test=n_test)
                        return


if __name__ == '__main__':
    rng = np.random.RandomState(1234)
    train(rng=rng, batch_size=64, n_epochs=10, n_in=28*28, n_hidden=500, n_out=10)
