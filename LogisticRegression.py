import os
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

from Statistics import Statistic


def sigmoid(matrix):
    return 1.0 / (1.0 + np.exp(-matrix))


class LogisticRegression(object):

    def __init__(self, class_range=3, file_name="", my_lambda=1):
        self.X = None
        self.Y = None
        self.theta = None
        self.class_range = class_range
        self.my_lambda = my_lambda
        self.cost_data = [[], [], []]
        self.stat = Statistic(output=file_name)
        self.X, self.Y, self.m = self.stat.get_training_set()
        self.theta = np.zeros([self.class_range, self.X.shape[1]])

    def cost_function(self, initial_theta, X, y, i):
        zero_theta = initial_theta.copy()
        zero_theta[0] = 0
        reg = (self.my_lambda * np.power(zero_theta, 2).sum())/2
        sig = sigmoid(np.matmul(X, initial_theta))
        cost = (-np.matmul(np.transpose(y), np.log(sig)) - np.matmul(np.transpose(1 - y), np.log(1 - sig)) + reg) / self.m
        self.cost_data[i].append(cost)

        return cost

    def gradient(self, initial_theta, X, y, i):
        zero_theta = initial_theta.copy()
        zero_theta[0] = 0
        sig = sigmoid(np.matmul(X, initial_theta))
        reg = self.my_lambda * zero_theta
        grad = (np.matmul(np.transpose(X), sig - y) + reg) / self.m
        return grad

    def optimize(self, index):
        initial_theta = np.zeros(len(self.theta[index]))
        y = self.Y == index
        y = y.astype(int)
        result = op.minimize(fun=self.cost_function,
                             x0=initial_theta,
                            args=(self.X, y, index),
                            method='TNC',
                            jac=self.gradient)
        print(result)
        return result.x

    def optimize_thetas(self):
        t = list()
        for i in range(self.class_range):
            t.append(self.optimize(i))

        self.theta = np.asarray(t)
        return self.theta

    def predict(self, x):
        w = np.matmul(self.theta, x).tolist()
        print(w, w.index(max(w)))
        return w.index(max(w))


def main():
    lr = LogisticRegression(file_name="dump_ot.txt", my_lambda=0)
    # print(lr.optimize_thetas())
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    ls = [0]
    for l, color in zip(ls, colors):
        lr.my_lambda = l
        lr.theta = np.zeros([lr.class_range, lr.X.shape[1]])
        lr.cost_data[0].clear()
        lr.cost_data[1].clear()
        lr.cost_data[2].clear()
        lr.optimize_thetas()
        plt.plot(lr.cost_data[0], "k")
        plt.plot(lr.cost_data[1], "b")
        plt.plot(lr.cost_data[2], "r")

    plt.show()


if __name__ == "__main__":
    main()