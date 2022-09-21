import numpy as np


class Perceptron:
    def __init__(self, inputs=None, weights=None, bias=None, activation_func: str = "sign", learning_const=0.1):

        if input is not None and weights is not None and len(inputs) != len(weights):
            raise Exception("Vectors must have same size")

        self.inputs = np.array(inputs)
        self.weights = np.array(weights)
        self.bias = bias
        self.activation_func = activation_func
        self.learning_const = learning_const
        self.output = None
        self._activation_fucntion = {
            "sign": self._sign,
            "sigmoid": self._sigmoid,
            "tanh": self._tanh,
            "relu": self._relu,
        }
        self._activation_fucntion_deriv = {
            "sign": self._sign_deriv,
            "sigmoid": self._sigmoid_deriv,
            "tanh": self._tanh_deriv,
            "relu": self._relu_deriv,
        }

    def predict(self):
        self.output = self.feedforward()

    def feedforward(self):
        total = (self.inputs * self.weights).sum() + self.bias
        activation = self._activation_fucntion.get(self.activation_func)
        return activation(total)

    def gradient_descent(self, _class):
        delta_weight = np.zeros(len(self.weights))
        prev_delta_weights = np.array([None] * len(self.weights))

        while np.not_equal(delta_weight, prev_delta_weights).any():
            """Debugging prints - uncomment to use
            print(f"delta_weight:{delta_weight}, prev_delta_weights:{prev_delta_weights}")
            print(f"weights:{self.weights}, learning_const:{self.learning_const}")
            print("----")"""

            prev_delta_weights = delta_weight.copy()
            deriv = self._activation_fucntion_deriv.get(self.activation_func)
            delta_weight = delta_weight + self.learning_const * self.inputs * deriv(_class)
            self.weights = self.weights + delta_weight
            self.predict()

    def _sign(self, x):
        return 1 if x > 0 else 0

    def _sign_deriv(self, _class):
        return _class - self.output

    def _sigmoid(self, x):
        sig = 1 / (1 + np.exp(-x))
        return 1 if sig >= 0.5 else 0

    def _sigmoid_deriv(self, _class):
        return (_class - self.output) * self.output * (1 - self.output)

    def _tanh(self, x):
        sig = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        return 1 if sig >= 0 else 0

    def _tanh_deriv(self, _class):
        return _class - self.output

    def _relu(self, x):
        return max(0, x)

    def _relu_deriv(self, _class):
        return _class - self.output

    def set_weights(self, weights):
        self.weights = np.array(weights)

    def set_inputs(self, inputs):
        self.inputs = np.array(inputs)

    def __str__(self):
        o = self.output if self.output != None else 'na'
        return f"Perceptron: {o}"
