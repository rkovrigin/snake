from copy import deepcopy
import numpy as np
from scipy.io import loadmat

from Statistics import Statistic
from LogisticRegression import sigmoid
import scipy.optimize as op
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid_gradient(np_array):
    sg = sigmoid(np_array)
    g = sg * (1 - sg)
    return g


class NeuralNetwork(object):

    def __init__(self, my_lambda=0, file_name="dump.txt"):
        statistic = Statistic(output=file_name)
        x, y, self.m = statistic.get_training_set()

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

        self.my_lambda = my_lambda
        self.input_layer_size = self.X.shape[1]
        self.internal_layer_size = 25

        self.theta1 = None
        self.theta2 = None
        self.w1 = None
        self.w2 = None
        self.weights = None
        self.j = []
        # self._randomize_thetas()

    def _randomize_thetas(self):
        self.theta1 = self.randomize_weights(self.internal_layer_size, self.input_layer_size)
        self.theta2 = self.randomize_weights(self.output_layer_size, self.internal_layer_size)

    def randomize_weights(self, l_in, l_out):
        epsilon = 0.12
        return np.random.rand(l_in, l_out + 1) * 2 * epsilon - epsilon

    def _set_y(self, input_size, output_size, y):
        Y = np.zeros((input_size, output_size))
        for i in range(input_size):
            z = np.zeros(output_size)
            z[y[i]] = 1
            Y[i, :] = z
        return Y

    def cost_function(self, initial_thetas, *args):
        """
        J = sum(sum(log(h).*(-Y) - log(1-h).*(1-Y)))/m; % .* because of y(kth) and h(kth); every vector h[1,2,..,n] is one training output and y[0,0,...1,..,n]
        reg1 = sum(sum((Theta1.^2)(:, 2:end))); % from 2 because at 1th we have bias!
        reg2 = sum(sum((Theta2.^2)(:, 2:end))); % from 2 because at 1th we have bias!
        reg = lambda*(reg1 + reg2)/(2*m)
        J = J + reg;
        """
        x, y, my_lambda, shapes = args
        t1_shape, t2_shape = shapes
        theta1 = deepcopy(initial_thetas[0:t1_shape[0] * t1_shape[1]].reshape(t1_shape))
        theta2 = deepcopy(initial_thetas[t1_shape[0] * t1_shape[1]:].reshape(t2_shape))
        n = x.shape[0]

        ones = np.ones((n, 1))
        a1 = np.hstack((ones, x))
        z2 = a1 @ theta1.T
        a2 = sigmoid(z2)

        a2 = (np.hstack((ones, a2)))
        z3 = a2 @ theta2.T
        a3 = sigmoid(z3)
        h = a3

        Y = self._set_y(n, self.output_layer_size, y)

        J = np.sum(np.log(h) * (-Y) - np.log(1 - h) * (1 - Y)) / self.m
        reg1 = np.sum(np.sum(np.square(theta1[:, 1:]), axis=1))
        reg2 = np.sum(np.sum(np.square(theta2[:, 1:]), axis=1))
        reg = my_lambda * (reg1 + reg2) / (2 * self.m)

        self.j.append(J + reg)
        return J + reg

    def optimize_thetas(self, initial_thetas, *args):
        x, y, my_lambda, shapes = args
        t1_shape, t2_shape = shapes
        n = x.shape[0]
        theta1 = deepcopy(initial_thetas[:t1_shape[0] * t1_shape[1]].reshape(t1_shape))
        theta2 = deepcopy(initial_thetas[t1_shape[0] * t1_shape[1]:].reshape(t2_shape))
        t1 = deepcopy(theta1)
        t2 = deepcopy(theta2)
        Y = self._set_y(n, self.output_layer_size, y)

        ones = np.ones((n, 1))
        a1 = np.hstack((ones, x))
        z2 = a1 @ theta1.T
        a2 = sigmoid(z2)

        a2 = (np.hstack((ones, a2)))
        z3 = a2 @ theta2.T
        a3 = sigmoid(z3)

        d3 = a3 - Y
        d2 = np.multiply(d3 @ theta2[:, 1:], sigmoid_gradient(z2))

        theta1 = theta1 + (d2.T @ a1)
        theta2 = theta2 + (d3.T @ a2)

        theta1 = theta1 / self.m
        theta2 = theta2 / self.m

        """
        Theta1_grad = Theta1_grad./m;
        Theta2_grad = Theta2_grad./m;

        % PART 4

        Theta1_grad = Theta1_grad + (lambda * [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)])/m; % 0 as a 1st column, because we do not update BIAS!
        Theta2_grad = Theta2_grad + (lambda * [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)])/m;
        """

        t1[:, 0] = np.zeros(t1.shape[0])
        theta1_grad = theta1 + (my_lambda * t1) / self.m

        t2[:, 0] = np.zeros(t2.shape[0])
        theta2_grad = theta2 + (my_lambda * t2) / self.m

        return np.hstack((theta1_grad.ravel(), theta2_grad.ravel()))

    def _calc_inner_layers(self, x, theta1, theta2):
        if type(x) is list:
            x = np.array(x)
        n = x.shape[0]
        a1 = np.hstack((np.ones(1), x)).reshape(1, n + 1)
        z2 = np.matmul(a1, np.transpose(theta1))
        a2 = sigmoid(z2)
        a2 = np.transpose(np.vstack(np.hstack((np.ones(1), a2[0]))))
        z3 = np.matmul(a2, np.transpose(theta2))
        a3 = sigmoid(z3)
        return a1, a2, a3, z2, z3

    def optimize(self):
        # result = op.fmin_cg(f=self.cost_function,
        #                     # x0=np.hstack((self.theta1.ravel(order='F'), self.theta2.ravel(order='F'))),
        #                     x0=np.hstack((self.theta1.ravel(), self.theta2.ravel())),
        #                     fprime=self.optimize_thetas,
        #                     args=(self.X, self.Y, self.my_lambda, self.theta1.shape, self.theta2.shape),
        #                     # maxiter=150
        #                     )
        # result = op.minimize(fun=self._iterativ_cost_function,
        #                      x0=np.hstack((self.theta1.ravel(), self.theta2.ravel())),
        #                      jac=self._iterativ_optimize_thetas,
        #                      args=(self.X, self.Y, self.my_lambda, (self.theta1.shape, self.theta2.shape)),
        #                      method='TNC')
        result = op.minimize(fun=self.cost_function,
                             x0=np.hstack((self.theta1.ravel(), self.theta2.ravel())),
                             jac=self.optimize_thetas,
                             args=(self.X, self.Y, self.my_lambda, (self.theta1.shape, self.theta2.shape)),
                             method='TNC')

        self.w1 = result.x[0:self.theta1.shape[0] * self.theta1.shape[1]].reshape(self.theta1.shape)
        self.w2 = result.x[self.theta1.shape[0] * self.theta1.shape[1]:].reshape(self.theta2.shape)
        self.weights = [self.w1, self.w2]

        out = self.predict_raw(self.X)
        rate = np.mean(np.argmax(out, axis=1) == self.Y.flatten()) * 100
        print("Status - %s" % result['success'], "; Message - %s" % result['message'], "; Status - %s" % result['status'],  "; Rate = %f" % rate)

        return self.w1, self.w2

    def predict_raw(self, X):
        theta1, theta2 = self.w1, self.w2

        a1 = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        z2 = a1 @ theta1.T
        a2 = sigmoid(z2)
        a2 = np.append(np.ones((a2.shape[0], 1)), a2, axis=1)
        z3 = a2 @ theta2.T
        a3 = sigmoid(z3)

        return a3

    def predict(self, X):
        if isinstance(X, list):
            X = np.array([X])
        out = self.predict_raw(X)
        print(out[-1], np.argmax(out[-1]))
        return np.argmax(out[-1])


def main():
    nn = NeuralNetwork(file_name="dump_ot.txt", my_lambda=1)
    nn._randomize_thetas()
    res = nn.optimize()
    plt.plot(nn.j)
    nn.j = []


if __name__ == "__main__":
    main()
