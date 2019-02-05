import os
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt


def sigmoid(matrix):
    return 1.0 / (1.0 + np.exp(-matrix))


class LogisticRegression(object):

    def __init__(self, file_name="", my_lambda=0):
        self.X = None
        self.Y = None
        self.theta = None
        self.classes = 3
        self.my_lambda = my_lambda
        self.cost_data = [[], [], []]

        if os.path.exists(file_name):
            x_data = list()
            y_data = list()
            f = open(file_name, "r+")
            last = None
            for line in f:
                x_data.append([int(i) for i in line.split(",")][:-2])
                y_data.append(int(line.split(",")[-1:][0]))

            self.X = np.asarray(x_data)
            self.Y = np.asarray(y_data)
            self.theta = np.zeros([self.classes, len(x_data[0])])
            self.m = len(y_data)

            # print(self.X.shape)
            # print(self.Y.shape)
            # print(self.theta.shape)

    def cost_function(self, initial_theta, X, y, i):
        zero_theta = initial_theta.copy()
        zero_theta[0] = 0
        reg = (self.my_lambda * np.power(zero_theta, 2).sum())/2

        #s = sigmoid(X*theta);

        sig = sigmoid(np.matmul(X, initial_theta))

        # J = (-1*y'*log(s) - (1-y)'*log(1-s) + t)/m;
        cost = (-np.matmul(np.transpose(y), np.log(sig)) - np.matmul(np.transpose(1 - y), np.log(1 - sig)) + reg) / self.m

        # grad = (X'*(sigmoid(X*theta) - y) + lambda*t_theta)/m;
        # grad = (np.matmul(np.transpose(X), np.subtract(sig, y)) + self.my_lambda * zero_theta) / self.m
        self.cost_data[i].append(cost)

        return cost

    def gradient(self, initial_theta, X, y, i):
        zero_theta = initial_theta.copy()
        zero_theta[0] = 0
        sig = sigmoid(np.matmul(X, initial_theta))
        reg = self.my_lambda * zero_theta
        # reg = 0
        grad = (np.matmul(np.transpose(X), sig - y) + reg) / self.m
        # grad = (np.matmul(np.transpose(X), np.subtract(sig, y)) + self.my_lambda * zero_theta) / self.m
        # grad = (np.matmul(np.transpose(X), np.subtract(sig, y))) / self.m

        return grad

    def optimize(self, index):
        initial_theta = np.zeros(len(self.theta[index]))
        # y = self.Y == index
        # y = y.astype(int)
        # y = self.Y
        result = op.minimize(fun=self.cost_function,
                             x0=initial_theta,
                            args=(self.X, self.Y, index),
                            method='TNC',
                            jac=self.gradient)
        print(result)
        return result.x

    def optimize_thetas(self):
        t = list()
        for i in range(self.classes):
            t.append(self.optimize(i))

        self.theta = np.asarray(t)
        return self.theta

    def predict(self, x):
        w = np.matmul(self.theta, x).tolist()
        print(w, w.index(max(w)))
        return w.index(max(w))


# lr = LogisticRegression("learning3_copy.txt")
# print(lr.optimize_thetas())
#
# plt.plot(lr.cost_data[0])
# plt.plot(lr.cost_data[1])
# plt.plot(lr.cost_data[2])
# plt.ylabel('some numbers')
# plt.show()
