from copy import deepcopy
from random import randrange

from scipy.io import loadmat

from LogisticRegression import sigmoid
from NeuralNetwork import NeuralNetwork, sigmoid_gradient
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from Statistics import Statistic


class NeuralNetworkPlus(NeuralNetwork):

    def __init__(self, my_lambda=0, file_name="dump.txt", layers=[25, 3]):
        super().__init__(my_lambda, file_name)
        statistic = Statistic(output=file_name)
        x, y, self.m = statistic.get_training_set()
        self.layers = layers

        self.X = x[:, :]
        self.Y = y[:]

        self.m = len(self.Y)
        self.output_layer_size = 3

        # reading the data
        # data = loadmat('ex4data1.mat')
        # self.X = data['X'][0:500, :]
        # self.Y = data['y'][0:500, :]
        # self.m = len(self.Y)
        # self.output_layer_size = 10
        # self.layers = layers

        self.my_lambda = my_lambda
        self.input_layer_size = self.X.shape[1]
        self.layers.insert(0, self.input_layer_size)
        self.thetas = list()
        self.j = []
        self.weights = None
        # self._randomize_thetas()

    def _randomize_thetas(self):
        for i in range(len(self.layers) - 1):
            self.thetas.append(self.randomize_weights(self.layers[i+1], self.layers[i]))

    def ravel(self, thetas):
        ret = []
        for theta in thetas:
            ret.append(theta.ravel())
        return ret

    def optimize_thetas(self, initial_thetas, *args):
        x, y, my_lambda, theta_shapes = args
        thetas = list()
        t = list()
        _start_position = 0
        for shape in theta_shapes:
            thetas.append(deepcopy(initial_thetas[_start_position: _start_position + shape[0] * shape[1]].reshape(shape)))
            t.append(deepcopy(initial_thetas[_start_position: _start_position + shape[0] * shape[1]].reshape(shape)))
            _start_position = shape[0] * shape[1]
        n = x.shape[0]
        ones = np.ones((n, 1))
        a = [x]
        z = list()
        for i in range(len(thetas)):
            a[i] = np.hstack((ones,  a[i]))
            z.append(a[i] @ thetas[i].T)
            a.append(sigmoid(z[i]))

        Y = self._set_y(n, self.output_layer_size, y)

        d = [a[-1] - Y]
        for i in range(len(thetas) - 1):
            _d = np.multiply(d[-1] @ thetas[-i - 1][:, 1:], sigmoid_gradient(z[-i - 2]))
            d.append(_d)
        d.reverse()

        ret = []
        for i in range(len(thetas)):
            thetas[i] = (thetas[i] + (d[i].T @ a[i])) / self.m
            t[i][:, 0] = np.zeros(t[i].shape[0])
            t[i] = thetas[i] + (my_lambda * t[i])
            ret.append(t[i].ravel())

        return np.hstack(ret)

    def cost_function(self, initial_thetas, *args):
        x, y, my_lambda, theta_shapes = args
        thetas = list()
        _start_position = 0
        for shape in theta_shapes:
            thetas.append(deepcopy(initial_thetas[_start_position: _start_position + shape[0] * shape[1]].reshape(shape)))
            _start_position = shape[0] * shape[1]
        n = x.shape[0]
        ones = np.ones((n, 1))
        a = [x]
        z = list()
        for i in range(len(thetas)):
            a[i] = np.hstack((ones,  a[i]))
            z.append(a[i] @ thetas[i].T)
            a.append(sigmoid(z[i]))

        Y = self._set_y(n, self.output_layer_size, y)

        reg = 0
        for theta in thetas:
            reg += np.sum(np.sum(np.square(theta[:, 1:]), axis=1))

        J = np.sum(np.log(a[-1]) * (-Y) - np.log(1 - a[-1]) * (1 - Y)) / self.m
        reg = my_lambda * reg / (2 * self.m)

        self.j.append(J + reg)
        return J + reg

    def optimize(self):
        initial_thetas = self.ravel(self.thetas)
        result = op.minimize(fun=self.cost_function,
                             x0=np.hstack(initial_thetas),
                             jac=self.optimize_thetas,
                             args=(self.X, self.Y, self.my_lambda, [theta.shape for theta in self.thetas]),
                             method='TNC')

        _start_position = 0
        self.weights = []
        for shape in [theta.shape for theta in self.thetas]:
            self.weights.append(deepcopy(result.x[_start_position: _start_position + shape[0] * shape[1]].reshape(shape)))
            _start_position = shape[0] * shape[1]

        out = self.predict_raw(self.X)
        rate = np.mean(np.argmax(out, axis=1) == self.Y.flatten()) * 100
        print("Status - %s" % result['success'], "; Message - %s" % result['message'], "; Status - %s" % result['status'], "; Rate = %f" % rate)
        print(result['message'], result['success'])

        return [weight for weight in self.weights]

    def predict(self, X):
        if isinstance(X, list):
            X = np.array([X])
        out = self.predict_raw(X)
        print(out[-1], np.argmax(out[-1]), np.max(out[-1]))
        return np.argmax(out[-1])

    def predict_raw(self, X):
        a = [X]
        z = []

        for i in range(len(self.weights)):
            a[i] = np.append(np.ones((a[i].shape[0], 1)), a[i], axis=1)
            z.append(a[i] @ self.weights[i].T)
            a.append(sigmoid(z[i]))

        return a[-1]


def main():
    # nn = NeuralNetwork(file_name="dump_nn.txt", my_lambda=1)
    # nn._randomize_thetas()
    # nnp = NeuralNetworkPlus(file_name="dump_nn.txt", my_lambda=1, layers=[25, 3])
    # nnp.j = []
    # nnp.thetas[0] = deepcopy(nn.theta1)
    # nnp.thetas[1] = deepcopy(nn.theta2)
    #
    # nn.optimize()
    # nnp.optimize()
    #
    # # j = nn.cost_function(np.hstack((nn.theta1.ravel(), nn.theta2.ravel())), nn.X, nn.Y, nn.my_lambda, (nn.theta1.shape, nn.theta2.shape))
    # # jp = nnp.cost_function(np.hstack((nnp.thetas[0].ravel(), nnp.thetas[1].ravel())), nnp.X, nnp.Y, nnp.my_lambda,
    # #                      (nnp.thetas[0].shape, nnp.thetas[1].shape))
    # print (nn.j == nnp.j, nn.j, nnp.j)
    #
    # plt.plot(nn.j, 'g')
    # plt.plot(nnp.j, 'r')
    # plt.show()


    layers = [randrange(5) for _ in range(1)]
    layers.append(3)
    nnp = NeuralNetworkPlus(file_name="dump_ot.txt", my_lambda=1, layers=[10, 3])
    nnp._randomize_thetas()
    nnp.optimize()
    plt.plot(nnp.j, 'r')
    plt.show()


if __name__ == "__main__":
    main()