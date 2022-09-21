from Models.perceptron import Perceptron


import numpy as np
import pandas as pd

class ANN:
    def __init__(self, layout:tuple, activation:str, size):
        self.network=[]
        self.activation = activation
        self.layout = layout
        self._init_layers(self.layout, size)
        

    def _init_layers(self ,layout, size):
        for layer in layout:
            inner_layer = []
            for i in range(layer):
                inner_layer.append(Perceptron())
                inner_layer[i].set_weights(self._generate_init_weights(size))
            self.network.append(inner_layer)

    def _generate_init_weights(self, size):
        random_weights = np.random.rand(size)
        random_weights = [x if np.random.randint(3)!=2 else -x for x in random_weights]
        return random_weights

        
    def fit(self, X : pd.DataFrame, y:pd.Series):

        self._init_layers(self.layout, X.shape[1])
        for index, row in X.iterrows():
            pass

    def __str__(self):
        text = ''
        for layer in self.network:
            for p in layer:
                text += str(p) + ' '
            text += '\n'
        return text
    