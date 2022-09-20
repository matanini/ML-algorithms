import numpy as np


class Perceptron:
    def __init__(self, inputs, weights, bias, activation_func: str, learning_const=0.1):

        if len(inputs) != len(weights):
            raise Exception("Vectors must have same size")

        self.inputs = np.array(inputs)
        self.weights = np.array(weights)
        self.bias = bias
        self.activation_func = activation_func
        self.learning_const = learning_const
        self._activation_fucntion = {
            "sign": self._sign,
            "sigmoid": self._sigmoid,
            "tanh": self._tanh,
            "relu": self._relu,
        }
        self._activation_fucntion_der = {
            "sign": self._sign_der,
            "sigmoid": self._sigmoid_der,
            "tanh": self._tanh_der,
            "relu": self._relu_der,
        }

        self.output = self.feedforward()

    def feedforward(self):
        total = (self.inputs * self.weights).sum() + self.bias
        activation = self._activation_fucntion.get(self.activation_func)
        return activation(total)

    def gradient_descent(self, _class):
        delta_weight = np.zeros(len(self.weights))
        prev_delta_weights = np.array([None] * len(self.weights))

        while np.not_equal(delta_weight, prev_delta_weights).any():

            # Debugging
            # print(f"weights:{self.weights}, prev_delta_weights:{prev_delta_weights}")
            # print(f"delta_weight:{delta_weight} , learning_const:{self.learning_const} , _class - self.output:{(_class - self.output)}")
            # print(f"mult:{self.learning_const * (_class - self.output) * self.inputs}")
            # print("----")

            prev_delta_weights = delta_weight.copy()
            delta_weight = delta_weight + self.learning_const * (_class - self.output) * self.inputs
            self.weights = self.weights + delta_weight
            self.output = self.feedforward()

    def _sign(self, x):
        return 1 if x > 0 else 0

    def _sign_der(self,_class):
        return self.learning_const * (_class - self.output) * self.inputs

    def _sigmoid(self, x):
        sig = 1 / (1 + np.exp(-x))
        return 1 if sig >= 0.5 else 0

    def _sigmoid_der(self, _class):
        return self.learning_const * (_class - self.output) * self.output* (1-self.output)*self.inputs

    def _tanh(self, x):
        sig = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        return 1 if sig >= 0 else 0

    def _tanh_der(self, _class):
        return self.learning_const * (_class - self.output) * self.inputs

    def _relu(self, x):
        return max(0, x)

    def _relu_der(self, _class):
        return self.learning_const * (_class - self.output) * self.inputs

    def set_weights(self, weights):
        self.weights = np.array(weights)
    

    def __str__(self):
        return f"Perceptron: {self.output}"
