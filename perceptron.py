class Perceptron:
    def __init__(self, inputs, weights, bias):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.output = self.feedforward()

    def feedforward(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return 1 if total + self.bias > 0 else 0

    def __str__(self):
        return f"Perceptron: {self.output}"