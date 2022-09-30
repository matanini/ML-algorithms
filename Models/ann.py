from Models.perceptron import Perceptron


import numpy as np
import pandas as pd


class ANN:
    def __init__(self, layout: tuple, activation: str):
        self.network = []
        self.activation = activation
        self.layout = layout
        # self._init_layers(self.layout, size)

    def _init_layers(self, f_vsize, output_nodes):
        self.output_layer = np.array([] * output_nodes)
        self.vector_size_array = [f_vsize] + list(self.layout) + [output_nodes]
        print(self.vector_size_array)

        for i in range(1, len(self.vector_size_array)):
            inner_layer = []
            for _ in range(self.vector_size_array[i]):
                inner_layer.append(Perceptron(weights=self.generate_init_weights(self.vector_size_array[i - 1]), activation_func=self.activation))
            self.network.append(inner_layer)

    def generate_init_weights(self, size):
        random_weights = np.random.rand(size)
        random_weights = [x if np.random.randint(2) != 1 else -x for x in random_weights]
        return random_weights

    def fit(self, X: pd.DataFrame, y: pd.Series):

        self._init_layers(X.shape[1], len(y.unique()))
        # Initialize backpropagation weights
        error_term = np.array([[0]*len(self.vector_size_array)]*(max(self.vector_size_array)))
        
        for index, row in X.iterrows():
            
            # Feed forward
            input_x = row.tolist()
            for layer in self.network:
                pred = []
                for p in layer:
                    p.set_inputs(input_x)
                    pred.append(p.fit(y.loc[index]))
                input_x = pred

            # Backpropagation
            
            for i, o in enumerate(pred):
                error_term[i][-1] = o * (1-o) * (y.loc[index] - o)
                

            for i in range(len(self.vector_size_array)-2, -1,-1):
                for j in range(max(self.vector_size_array)):
                    if j < self.vector_size_array[i]:
                        print(f"[{i},{j}]",self.network[i][j])
                    # error_term[i] = error_term[j] * self.network[i][j].weights



    def __str__(self):
        text = f"ANN of dim {tuple(self.vector_size_array[1:])}:\n"
        for i, layer in enumerate(self.network):
            if i != len(self.network)-1:
                text += f"Hidden layer {i+1}:\n" 
            else:
                text += "Output layer:\n"
            for p in layer:
                text += str(p) + " "
            text += "\n"
        return text
