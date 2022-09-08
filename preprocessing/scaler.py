import numpy as np
import pandas as pd

class Scaler:
    def __init__(self, mode = 'minmax'):
        """
        @params
        mode: 'minmax' or 't-distribution'
        """
        self.mode = mode
    
    def fit(self, X: pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame):
        funcs = {'minmax':self.__scale_minmax, 't-distribution':self.__scale_t_distribution}
        action = funcs.get(self.mode)
        return action(X)

    def __scale_minmax(self, data: pd.DataFrame) -> None:
        for col in data.columns:
            min_value = data[col].min()
            max_value = data[col].max()

            for i, value in enumerate(data[col]):
                data.loc[i, col] = (value - min_value) / (max_value - min_value)
        return data

    def __scale_t_distribution(self, data: pd.DataFrame) -> None:
        for col in data.columns:
            mean_value = data[col].mean()
            std_value = data[col].std(ddof=1)

            for i, value in enumerate(data[col]):
                data.loc[i, col] = (value - mean_value) / std_value
        return data
