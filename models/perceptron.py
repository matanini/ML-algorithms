import numpy as np


class Perceptron:
    def __init__(self, inputs=None, weights=None, bias=None, activation_func: str = "sign", learning_const=0.1,output_layer = False):

        if inputs is not None and weights is not None and len(inputs) != len(weights):
            raise Exception("Vectors must have same size")

        self.inputs = np.array(inputs)
        self.weights = np.array(weights)
        if bias == None:
            self.bias = np.random.random()
        else:
            self.bias = bias
        self.activation_func = activation_func
        self.learning_const = learning_const
        self.output = None
        self.is_output_layer = output_layer
        self._activation_fucntion = {
            "sign": self._sign,
            "sigmoid": self._sigmoid,
            "tanh": self._tanh,
            "relu": self._relu,
        }
        # self._activation_fucntion_deriv = {
        #     "sign": self._sign_deriv,
        #     "sigmoid": self._sigmoid_deriv,
        #     "tanh": self._tanh_deriv,
        #     "relu": self._relu_deriv,
        # }

    def fit(self, y):
        self.output = self.feedforward()
        # self.gradient_descent(y)
        return self.output

    def feedforward(self):
        total = (self.inputs * self.weights).sum() + self.bias
        activation = self._activation_fucntion.get(self.activation_func)
        return activation(total)

    def gradient_descent(self, _class):
        delta_weight = np.zeros(len(self.weights))
        prev_delta_weights = np.array([None] * len(self.weights))

        while np.not_equal(delta_weight, prev_delta_weights).any():
            # Debugging prints - uncomment to use
            print(f"delta_weight:{delta_weight}, prev_delta_weights:{prev_delta_weights}")
            print(f"weights:{self.weights}, learning_const:{self.learning_const}")
            print("----")

            prev_delta_weights = delta_weight.copy()
            # deriv = self._activation_fucntion_deriv.get(self.activation_func)
            delta_weight = delta_weight + self.learning_const * self.inputs * self.deriv * (_class - self.output)
            self.weights = self.weights + delta_weight
            self.output = self.feedforward()

    def _sign(self, x):
        self.deriv = 1
        return 1 if x > 0 else 0

    # def _sign_deriv(self, _class):
    #     return _class - self.output

    def _sigmoid(self, x):
        f = 1 / (1 + np.exp(-x))
        self.deriv = f * (1 - f)
        return f

    # def _sigmoid_deriv(self, _class):
    #     return (_class - self.output) * self.output * (1 - self.output)

    def _tanh(self, x):
        f = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        self.deriv = 1 - (f * f)
        return f

    # def _tanh_deriv(self, _class):
    #     # TODO implement
    #     pass

    def _relu(self, x):
        self.deriv = 1
        return max(0, x)

    # def _relu_deriv(self, _class):
    #     # TODO implement
    #     pass

    def set_weights(self, weights):
        if self.inputs.all() != None:
            if len(self.inputs) != len(weights):
                print(f"new weights: {weights}, inputs: {self.inputs}")
                raise Exception("Vectors must have same size")
        self.weights = np.array(weights)
        return self

    def set_inputs(self, inputs):
        if self.weights.all() != None:
            if len(self.weights) != len(inputs):
                print(f"weights: {self.weights}, new inputs: {inputs}")
                raise Exception("Vectors must have same size")
        self.inputs = np.array(inputs)
        return self

    def __str__(self):
        o = self.output if self.output != None else "na"
        return f"Perceptron output {o}, input:{self.inputs}, weights:{self.weights}\n"
