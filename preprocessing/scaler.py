import numpy as np
import pandas as pd

class Scaler:
    def __init__(self, mode = 'minmax'):
        """
        @params
        mode: 'minmax' or 't-distribution'
        """
        if mode not in ['t-distribution', 'minmax']:
            raise ValueError('Invalid mode')
        self.mode = mode
    
    def fit(self, X: pd.DataFrame):
        
        idx = ['mean','std'] if self.mode == 't-distribution' else ['min','max']
        self.scale_data = pd.DataFrame(columns = X.columns, index = idx)
        if self.mode == 't-distribution':
            for col in X.columns:
                self.scale_data.loc['mean',col] = X[col].mean()
                self.scale_data.loc['std',col] = X[col].std(ddof=1)
        else: 
            for col in X.columns:
                self.scale_data.loc['min',col] = X[col].min()
                self.scale_data.loc['max',col] = X[col].max()



    def transform(self, X: pd.DataFrame):
        funcs = {'minmax':self.__scale_minmax, 't-distribution':self.__scale_t_distribution}
        action = funcs.get(self.mode)
        return action(X)
    
    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)

    def __scale_minmax(self, X: pd.DataFrame) -> None:
        for col in X.columns:
            min_value = self.scale_data.iloc['min',col]
            max_value = self.scale_data.iloc['max',col]

            for i, value in enumerate(X[col]):
                X.loc[i, col] = (value - min_value) / (max_value - min_value)
        return X

    def __scale_t_distribution(self, X: pd.DataFrame) -> None:
        for col in X.columns:
            mean_value = self.scale_data.iloc['mean',col]
            std_value = self.scale_data.iloc['std',col]

            for i, value in enumerate(X[col]):
                X.loc[i, col] = (value - mean_value) / std_value
        return X
