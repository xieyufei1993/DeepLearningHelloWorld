# coding: utf-8
import numpy as np
import cPickle
from data import load_data
from Models import MLPModel


def train(rng=None, batch_size=64, n_epochs=50, n_in=28*28, n_hidden=500, n_out=10):
    dataset = load_data(data_set_path='mnist.pkl.gz', share=False)
    train_set_x, train_set_y = dataset[0]
    train_set_y = np.array(train_set_y, dtype='int32')
    valid_set_x, valid_set_y = dataset[1]
    valid_set_y = np.array(valid_set_y, dtype='int32')
    test_set_x, test_set_y = dataset[2]
    test_set_y = np.array(test_set_y, dtype='int32')
    # compute number of mini batches for training, validation and testing
    n_train = train_set_x.shape[0]
    n_valid = valid_set_x.shape[0]
    n_test = test_set_x.shape[0]
    n_train_batches = n_train // batch_size
    n_valid_batches = n_valid // batch_size
    n_test_batches = n_test // batch_size
    print('data loaded ok.')
    # model
    mlp = MLPModel(rng, n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    print('model build ok.')
    print('begin to train...')
    # train detail
    validation_frequency = 30
    valid_iter_num = 0
    best_validation_loss_rate = 1.0
    for epoch in range(n_epochs):
        mini_batch_avg_cost_list = []
        print ('epoch # : ' ,epoch)
        for index in range(n_train_batches):
            valid_iter_num += 1
            mini_batch_avg_cost = mlp.fit(
                train_set_x[index*batch_size: (index+1)*batch_size],
                train_set_y[index*batch_size: (index+1)*batch_size]
            )
            mini_batch_avg_cost_list.append(mini_batch_avg_cost)
            print ('cost: ', mini_batch_avg_cost)

            if valid_iter_num % validation_frequency == 0:
                # compute zero-one loss on validation set
                this_validation_loss_rate = mlp.evaluate(valid_set_x, valid_set_y, n_valid, batch_size)
                if best_validation_loss_rate >= this_validation_loss_rate:
                    best_validation_loss_rate = this_validation_loss_rate
                    print 'epoch {} get new best validation error rate: {}'.format(epoch, best_validation_loss_rate)
                    if best_validation_loss_rate < 0.05:
                        # this_test_loss_rate = mlp.evaluate(test_set_x, test_set_y, n_test, batch_size)
                        # print('test error rate: ', this_test_loss_rate)
                        print('save params...')
                        mlp.nn_classifier.save_params('params.pkl')
                        print('end end end')
                        return


def test(rng=None, batch_size=64, n_in=28*28, n_hidden=500, n_out=10):
    dataset = load_data(data_set_path='mnist.pkl.gz', share=False)

    test_set_x, test_set_y = dataset[2]
    test_set_y = np.array(test_set_y, dtype='int32')
    n_test = test_set_x.shape[0]
    print('data loaded ok.')
    # model

    mlp = MLPModel(rng, n_in=n_in, n_hidden=n_hidden, n_out=n_out, params=MLPModel.load_params('params.pkl'))
    print('model build ok.')
    this_test_loss_rate = mlp.evaluate(test_set_x, test_set_y, n_test, batch_size)
    print ('test error rate:', this_test_loss_rate)


if __name__ == '__main__':
    rng = np.random.RandomState(1234)

    train(rng=rng, batch_size=64, n_epochs=50, n_in=28*28, n_hidden=500, n_out=10)

    test(rng=None, batch_size=64, n_in=28 * 28, n_hidden=500, n_out=10)


