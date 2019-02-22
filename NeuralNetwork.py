from copy import deepcopy
import numpy as np
from scipy.io import loadmat

from Statistics import Statistic
from LogisticRegression import sigmoid
import scipy.optimize as op
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid_gradient(np_array):
    # g = np.zeros(np_array.shape)
    sg = sigmoid(np_array)
    g = sg * (1 - sg)
    return g


class NeuralNetwork(object):

    def __init__(self, output_layer_size=None, internal_layers=1, my_lambda=0, file_name="dump.txt"):
        statistic = Statistic(output=file_name)
        x, y, self.m = statistic.get_training_set()

        self.X = x[:, :]
        self.Y = y[:]

        self.m = len(self.Y)
        self.output_layer_size = 3

        # reading the data
        # data = loadmat('ex4data1.mat')
        # self.X = data['X'][0000:2000, :]
        # self.Y = data['y'][0000:2000, :]
        # self.m = len(self.Y)
        # self.output_layer_size = 10

        self.my_lambda = my_lambda

        self.input_layer_size = self.X.shape[1]
        self.internal_layer_size = 25

        if not internal_layers:
            self.internal_layers = [self.internal_layer_size]
        else:
            self.internal_layers = internal_layers

        self.theta1 = self.randomize_weights(self.internal_layer_size, self.input_layer_size)
        self.theta2 = self.randomize_weights(self.output_layer_size, self.internal_layer_size)
        self.w1 = None
        self.w2 = None
        self.j = []

    def randomize_weights(self, l_in, l_out):
        epsilon = 0.12
        return np.random.rand(l_in, l_out + 1) * 2 * epsilon - epsilon

    def cost_function(self, initial_thetas, *args):
        """
        J = sum(sum(log(h).*(-Y) - log(1-h).*(1-Y)))/m; % .* because of y(kth) and h(kth); every vector h[1,2,..,n] is one training output and y[0,0,...1,..,n]
        reg1 = sum(sum((Theta1.^2)(:, 2:end))); % from 2 because at 1th we have bias!
        reg2 = sum(sum((Theta2.^2)(:, 2:end))); % from 2 because at 1th we have bias!
        reg = lambda*(reg1 + reg2)/(2*m)
        J = J + reg;
        """
        x, y, my_lambda, t1_shape, t2_shape = args
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

        Y = np.zeros((n, len(set(y.flatten()))))
        Y = np.zeros((n, self.output_layer_size))
        for i in range(n):
            z = np.zeros(len(set(y.flatten())))
            z = np.zeros(self.output_layer_size)
            z[y[i]] = 1
            Y[i, :] = z

        J = np.sum(np.log(h) * (-Y) - np.log(1 - h) * (1 - Y)) / self.m
        reg1 = np.sum(np.sum(np.square(theta1[:, 1:]), axis=1))
        reg2 = np.sum(np.sum(np.square(theta2[:, 1:]), axis=1))
        reg = my_lambda * (reg1 + reg2) / (2 * self.m)

        self.j.append(J + reg)
        return J + reg

    def optimize_thetas(self, initial_thetas, *args):
        x, y, my_lambda, t1_shape, t2_shape = args
        n = x.shape[0]
        theta1 = deepcopy(initial_thetas[:t1_shape[0] * t1_shape[1]].reshape(t1_shape))
        theta2 = deepcopy(initial_thetas[t1_shape[0] * t1_shape[1]:].reshape(t2_shape))
        t1 = deepcopy(theta1)
        t2 = deepcopy(theta2)
        Y = np.zeros((n, len(set(y.flatten()))))
        Y = np.zeros((n, self.output_layer_size))
        for i in range(n):
            z = np.zeros(len(set(y.flatten())))
            z = np.zeros(self.output_layer_size)
            z[y[i]] = 1
            Y[i, :] = z

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
        result = op.minimize(fun=self.cost_function,
                             x0=np.hstack((self.theta1.ravel(), self.theta2.ravel())),
                             jac=self.optimize_thetas,
                             args=(self.X, self.Y, self.my_lambda, self.theta1.shape, self.theta2.shape),
                             method='TNC')

        self.w1 = result.x[0:self.theta1.shape[0] * self.theta1.shape[1]].reshape(self.theta1.shape)
        self.w2 = result.x[self.theta1.shape[0] * self.theta1.shape[1]:].reshape(self.theta2.shape)

        print("w1 = ", self.w1)
        print("w2 = ", self.w2)

        out = self.predict(self.w1, self.w2, self.X, self.Y)
        print(self.Y)
        print(out)
        print(np.mean(out == self.Y.flatten()) * 100)
        if 1 in out or 2 in out:
            print("CONGRATES!!!", self.my_lambda)
        print("Status - %s" % result['success'], "; Message - %s" % result['message'], "; Status - %s" % result['status'])

        return self.w1, self.w2

    def predict_(self, layer_1_input, theta1, theta2):
        a1, a2, a3, z2, z3 = self._calc_inner_layers(x=layer_1_input, theta1=theta1, theta2=theta2)
        a3 = a3.tolist()[0]
        # print(a3, a3.index(max(a3)))
        return a3.index(max(a3))

    def predict(self, theta1, theta2, X, y):
        m = len(y)
        ones = np.ones((m, 1))

        a1 = np.hstack((ones, X))
        z2 = a1 @ theta1.T
        a2 = sigmoid(z2)
        a2 = (np.hstack((ones, a2)))
        z3 = a2 @ theta2.T
        a3 = sigmoid(z3)

        return np.argmax(a3, axis=1)


def main():
    nn = NeuralNetwork(file_name="dump_nn.txt", my_lambda=90)
    res = nn.optimize()
    plt.plot(nn.j)
    nn.j = []

    # print(nn.w1, nn.w2)


if __name__ == "__main__":
    main()
