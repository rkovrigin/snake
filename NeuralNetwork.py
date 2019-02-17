import numpy as np
from Statistics import Statistic
from LogisticRegression import sigmoid


def sigmoid_gradient(np_array):
    # g = np.zeros(np_array.shape)
    sg = sigmoid(np_array)
    g = sg * (1 - sg)
    return g

class NeuralNetwork(object):

    def __init__(self, output_layer_size=None, internal_layers=1):
        statistic = Statistic(output="dump_nn.txt")
        self.X, self.Y, self.m = statistic.get_training_set()

        self.input_layer_size = self.X.shape[1]
        if not output_layer_size:
            self.output_layer_size = len(set(self.Y))
        self.internal_layer_size = 25

        if not internal_layers:
            self.internal_layers = [self.internal_layer_size]
        else:
            self.internal_layers = internal_layers

        self.theta1 = np.zeros((self.internal_layer_size, self.input_layer_size + 1))
        self.theta2 = np.zeros((self.output_layer_size, self.internal_layer_size + 1))


    def init_weights(self):
        pass

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

    def learning(self):
        theta1_grad = np.zeros(self.theta1.shape)
        theta2_grad = np.zeros(self.theta2.shape)

        for i in range(2):
            n = self.X[0].shape[0]
            a1 = np.hstack((np.ones(1), self.X[1])).reshape(1, n+1)
            z2 = np.matmul(a1, np.transpose(self.theta1))
            a2 = sigmoid(z2)
            a2 = np.transpose(np.vstack(np.hstack((np.ones(1), a2[0]))))
            z3 = np.matmul(a2, np.transpose(self.theta2))
            a3 = sigmoid(z3)

            d3 = a3 - self.Y[i]
            d2 = np.matmul(d3, self.theta2[:, 1:]) * sigmoid_gradient(z2)

            theta1_grad = theta1_grad + np.matmul(np.transpose(d2), a1)
            theta2_grad = theta2_grad + np.matmul(np.transpose(d3), a2)

def main():
    nn = NeuralNetwork()
    nn.learning()


if __name__ == "__main__":
    main()