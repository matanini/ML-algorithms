import pandas as pd

class KNN:

    def __init__(self, neighbors: int, distance_metric: str = "euclidean", scaling: str = None) -> None:
        self.k = neighbors
        self.scaling = scaling
        if distance_metric is not None:
            if distance_metric == "manhattan":
                self.p = 1
            elif distance_metric == "euclidean":
                self.p = 2

    def fit(self, Xtrain: pd.DataFrame, ytrain: pd.Series) -> None:
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        if self.scaling is not None:
            if self.scaling == "minmax":
                self.Xtrain = self.__scale_minmax(self.Xtrain)
            elif self.scaling == "t-distribution":
                self.Xtrain = self.__scale_t_distribution(self.Xtrain)

    def predict(self, Xtest: pd.DataFrame) -> pd.Series:
        if self.scaling is not None:
            if self.scaling == "minmax":
                Xtest = self.__scale_minmax(Xtest)
            elif self.scaling == "t-distribution":
                Xtest = self.__scale_t_distribution(Xtest)
        ypred = pd.Series(index=Xtest.index)
        for (i, feature_vector) in Xtest.iterrows():
            k_nearest_neighbors = self.__calculate_all_distance(feature_vector)
            pred_class = self.__vote(k_nearest_neighbors)
            ypred.iloc[i] = pred_class
        return ypred
            

    def __scale_minmax(self, data :pd.DataFrame) -> None:
        for col in data.columns:
            min_value = data[col].min()
            max_value = data[col].max()

            for i, value in enumerate(data[col]):
                data.loc[i, col] = (value - min_value) / (max_value - min_value)
        return data

    def __scale_t_distribution(self, data :pd.DataFrame) -> None:
        for col in data.columns:
            mean_value = data[col].mean()
            std_value = data[col].std(ddof=1)

            for i, value in enumerate(data[col]):
                data.loc[i, col] = (value - mean_value) / std_value
        return data

    def __calculate_all_distance(self, feature_vector):
        dist_list = []
        for (idx, vector) in self.Xtrain.iterrows():
            dist_list.append((idx,self.__calculate_distance(vector, feature_vector)))
        sorted_list = sorted(dist_list, key = lambda x: x[1])
        return sorted_list[:self.k]

    def __calculate_distance(self, vector1, vector2) -> float:
        if len(vector1) != len(vector2):
            raise ValueError("Vectors of different sizes")
        else:
            sum = 0
            for x, y in zip(vector1, vector2):
                sum += abs(x - y) ** self.p
            return sum ** (1 / self.p)

    def __vote(self, k_nearest_neighbors):
        cls_list = []
        for neighbor in k_nearest_neighbors:
            neighbor_cls = self.ytrain.loc[neighbor[0]]
            cls_list.append(neighbor_cls)
        return max(set(cls_list), key = cls_list.count)
    
    def calc_accuracy(self, ypred, ytest):
        correct = (ytest == ypred).sum()
        accuracy = correct / ytest.shape[0]
        return accuracy

