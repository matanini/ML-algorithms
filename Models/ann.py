from Models.perceptron import Perceptron


import numpy as np
import pandas as pd

class ANN:
    def __init__(self, layout:tuple, activation:str):
        self.network=[]
        self.activation = activation
        self._init_layers(layout)

    def _init_layers(self ,layout):
        for layer in layout:
            inner_layer = []
            for _ in range(layer):
                inner_layer.append(Perceptron([],[],0,self.activation))
            self.network.append(inner_layer)

    def _generate_init_weights(self):
        pass

        
    def fit(X : pd.DataFrame, y:pd.Series):
        pass
