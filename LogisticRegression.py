import os
import numpy as np


def sigmoid(matrix):
    return 1.0 / (1.0 + np.exp(matrix))


class LogisticRegression(object):

    def __init__(self, file_name=""):
        self.X = None
        self.Y = None
        self.theta = None
        self.classes = 4
        self.my_lambda = 0

        if os.path.exists(file_name):
            x_data = list()
            y_data = list()
            f = open(file_name, "r+")
            for line in f:
                x_data.append([int(i) for i in line.split(", ")][:-4])
                y_data.append([int(i) for i in line.split(", ")][-4:].index(1) + 1)

            self.X = np.asarray(x_data)
            self.Y = np.asarray(y_data)
            self.theta = np.zeros([self.classes, len(x_data[0])])
            self.m = len(y_data)

            # print(self.X.shape)
            # print(self.Y.shape)
            # print(self.theta.shape)

    def cost_function(self, index):
        initial_theta = self.theta[index]
        y = self.Y == index + 1
        y = y.astype(int)
        zero_theta = initial_theta.copy()
        zero_theta[0] = 0
        reg = (self.my_lambda * np.power(zero_theta, 2).sum())/2

        #s = sigmoid(X*theta);
        sig = sigmoid(np.matmul(self.X, initial_theta))

        # J = (-1*y'*log(s) - (1-y)'*log(1-s) + t)/m;
        cost = (-1 * np.matmul(np.transpose(y), np.log(sig)) - np.matmul(np.transpose(1 - y), np.log(1 - sig)) + reg) / self.m

        # grad = (X'*(sigmoid(X*theta) - y) + lambda*t_theta)/m;
        grad = (np.matmul(np.transpose(self.X), np.subtract(sig, y)) + self.my_lambda * zero_theta) / self.m

        return [cost, grad]


lr = LogisticRegression("learning.txt")

print(lr.cost_function(3))