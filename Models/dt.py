import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self) -> None:
        pass

    def build_tree(trainset, output):
        pass

    def __calc_entropy(self, X :pd.Series):
        val_count = X.value_counts()
        entropy = 0
        for _, count in val_count.items():
            prob = count / X.shape[0]
            entropy -= prob * np.log2(prob)
        return entropy

    def __specific_conditional_entropy(self, X :pd.DataFrame ,entropy_col, second_col, val):
        return self.__calc_entropy(X[ X[second_col] == val][entropy_col])

    def __conditional_entropy(self, df,x,y):
        entropy = 0
        for val, count in df[y].value_counts().items():
            prob = count / df.shape[0]
            entropy += prob * self.__specific_conditional_entropy(df, x,y,val)
        return entropy