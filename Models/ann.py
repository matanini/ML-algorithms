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
                inner_layer.append(Perceptron(weights=self._generate_init_weights(self.vector_size_array[i - 1]), activation_func=self.activation, output_layer=True))
            self.network.append(inner_layer)

    def _generate_init_weights(self, size):
        random_weights = np.random.rand(size)
        random_weights = [x if np.random.randint(3) != 2 else -x for x in random_weights]
        return random_weights

    def fit(self, X: pd.DataFrame, y: pd.Series):

        self._init_layers(X.shape[1], len(y.unique()))
        
        for index, row in X.iterrows():
            
            # Feed forward
            f_v = row.tolist()
            for layer in self.network:
                pred = []
                for p in layer:
                    p.set_inputs(f_v)
                    pred.append(p.fit(y.loc[index]))
                f_v = pred

            # Backpropagation
            output_delta_array = []
            for o in pred:
                output_delta_array.append(o * (1-o) * (y.loc[index] - o))
            print(f"output delta array:{output_delta_array}")
            for 



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
