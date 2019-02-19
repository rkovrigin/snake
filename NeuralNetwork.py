from copy import deepcopy
import numpy as np
from Statistics import Statistic
from LogisticRegression import sigmoid
import scipy.optimize as op
import matplotlib.pyplot as plt


def sigmoid_gradient(np_array):
    # g = np.zeros(np_array.shape)
    sg = sigmoid(np_array)
    g = sg * (1 - sg)
    return g


class NeuralNetwork(object):

    def __init__(self, output_layer_size=None, internal_layers=1, my_lambda=0, file_name="dump.txt"):
        statistic = Statistic(output=file_name)
        self.X, self.Y, self.m = statistic.get_training_set()
        self.my_lambda = my_lambda

        self.input_layer_size = self.X.shape[1]
        if not output_layer_size:
            self.output_layer_size = len(set(self.Y))
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
        J = sum(sum(log(h).*(-Y) - log(1-h).*(1-Y)))/m; % .* because of y(kth) and h(kth); evely vector h[1,2,..,n] is one training output and y[0,0,...1,..,n]
        reg1 = sum(sum((Theta1.^2)(:, 2:end))); % from 2 because at 1th we have bias!
        reg2 = sum(sum((Theta2.^2)(:, 2:end))); % from 2 because at 1th we have bias!
        reg = lambda*(reg1 + reg2)/(2*m)
        J = J + reg;
        """
        x, y, my_lambda, t1_shape, t2_shape = args
        theta1 = initial_thetas[0:t1_shape[0] * t1_shape[1]].reshape(t1_shape)
        theta2 = initial_thetas[t1_shape[0] * t1_shape[1]:].reshape(t2_shape)
        n = x.shape[0]
        a1 = np.hstack((np.ones((n, 1)), x))
        z2 = np.matmul(a1, np.transpose(theta1))
        a2 = sigmoid(z2)

        a2 = (np.hstack((np.ones((a2.shape[0], 1)), a2)))
        z3 = np.matmul(a2, np.transpose(theta2))
        a3 = sigmoid(z3)
        h = a3

        Y = np.zeros((n, len(set(y))))
        for i in range(n):
            z = np.zeros(len(set(y)))
            z[y[i]] = 1
            Y[i, :] = z

        J = np.sum(np.log(h) * (-Y) - np.log(1 - h) * (1 - Y)) / self.m
        reg1 = np.sum(np.square(theta1[:, 1:]))
        reg2 = np.sum(np.square(theta2[:, 1:]))
        reg = my_lambda * (reg1 + reg2) / (2 * self.m)

        self.j.append(J + reg)
        return J + reg

    def optimize_thetas(self, initial_thetas, *args):
        """
        for k=1:m
            a1 = [1, X(k, :)];
            z2 = a1*Theta1';
            a2 = sigmoid(z2);
            a2 = [1, a2];
            z3 = a2*Theta2';
            a3 = sigmoid(z3);

            d3 = a3 .- Y(k, :);
            d2 = d3 * Theta2(:, 2:end) .* sigmoidGradient(z2); % 1x25

            Theta1_grad = Theta1_grad + d2'*a1; % 25x401
            Theta2_grad = Theta2_grad + d3'*a2; % 10x26
        end
        Theta1_grad = Theta1_grad./m;
        Theta2_grad = Theta2_grad./m;

        % PART 4

        Theta1_grad = Theta1_grad + (lambda * [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)])/m; % 0 as a 1st column, because we do not update BIAS!
        Theta2_grad = Theta2_grad + (lambda * [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)])/m;
        """
        x, y, my_lambda, t1_shape, t2_shape = args
        theta1_grad = initial_thetas[:t1_shape[0] * t1_shape[1]].reshape(t1_shape)
        theta2_grad = initial_thetas[t1_shape[0] * t1_shape[1]:].reshape(t2_shape)

        for i in range(self.m):
            a1, a2, a3, z2, z3 = self._calc_inner_layers(x=x[i], theta1=theta1_grad, theta2=theta2_grad)
            d3 = a3 - y[i] / 4
            d2 = np.matmul(d3, theta2_grad[:, 1:]) * sigmoid_gradient(z2)
            theta1_grad = theta1_grad + np.matmul(np.transpose(d2), a1)
            theta2_grad = theta2_grad + np.matmul(np.transpose(d3), a2)

        t1_tmp = theta1_grad.copy()
        t1_tmp[:, t1_tmp.shape[1] - 1] = np.ones(t1_tmp.shape[0])
        theta1_grad = theta1_grad + (my_lambda * t1_tmp) / self.m

        t2_tmp = theta2_grad.copy()
        t2_tmp[:, t2_tmp.shape[1] - 1] = np.ones(t2_tmp.shape[0])
        theta2_grad = theta2_grad + (my_lambda * t2_tmp) / self.m

        return np.hstack((theta1_grad.ravel(order='F'), theta2_grad.ravel(order='F')))

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
        result = op.fmin_cg(f=self.cost_function,
                            x0=np.hstack((self.theta1.ravel(order='F'), self.theta2.ravel(order='F'))),
                            fprime=self.optimize_thetas,
                            args=(self.X, self.Y, self.my_lambda, self.theta1.shape, self.theta2.shape),
                            maxiter=50)

        self.w1 = result[0:self.theta1.shape[0] * self.theta1.shape[1]].reshape(self.theta1.shape)
        self.w2 = result[self.theta1.shape[0] * self.theta1.shape[1]:].reshape(self.theta2.shape)

        print(self.w1)
        print(self.w2)

        return result

    def predict(self, layer_1_input, theta1, theta2):
        a1, a2, a3, z2, z3 = self._calc_inner_layers(x=layer_1_input, theta1=theta1, theta2=theta2)
        a3 = a3.tolist()[0]
        print(a3, a3.index(max(a3)))
        return a3.index(max(a3))


def main():
    nn = NeuralNetwork(file_name="dump_nn.txt", my_lambda=1)
    res = nn.optimize()
    plt.plot(nn.j)
    plt.show()

    # print(nn.w1, nn.w2)


if __name__ == "__main__":
    main()
