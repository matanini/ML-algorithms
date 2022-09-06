from math import dist
from mimetypes import init
import pandas as pd


class KMeans:
    def __init__(self, k: int, init:str = "lloyd", distance_metric: str = "euclidean", scaling: str = None, change_val = 0.1) -> None:
        self.k = k
        if distance_metric is not None:
            if distance_metric == "manhattan":
                self.p = 1
            elif distance_metric == "euclidean":
                self.p = 2
        self.scaling = scaling
        self.init_method = init
        self.change_val = change_val
        self.prototypes = None

    def fit(self, Xtrain: pd.DataFrame) -> None:
        self.Xtrain = Xtrain
        if self.scaling is not None:
            if self.scaling == "minmax":
                self.Xtrain = self.__scale_minmax(self.Xtrain)
            elif self.scaling == "t-distribution":
                self.Xtrain = self.__scale_t_distribution(self.Xtrain)

        self.__get_prototypes()
        self.pred = pd.Series(index=Xtrain.index)
        while True:
            for (i, feature_vector) in self.Xtrain.iterrows():
                self.pred.iloc[i] = self.__assign_prototype_to_vector(feature_vector)
            prev_iter_prototypes = self.prototypes.copy()
            self.__get_prototypes()
            if self.__check_prototypes_changes(prev_iter_prototypes) < self.change_val:
                break

    def predict(self, Xtest: pd.DataFrame) -> pd.Series:
        if self.scaling is not None:
            if self.scaling == "minmax":
                Xtest = self.__scale_minmax(Xtest)
            elif self.scaling == "t-distribution":
                Xtest = self.__scale_t_distribution(Xtest)
        pred = pd.Series(index=Xtest.index)
        for (i, feature_vector) in Xtest.iterrows():
            pred.iloc[i] = self.__assign_prototype_to_vector(feature_vector)
        return pred

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

    def __get_prototypes(self) -> list:
        if self.prototypes is None:
            if self.init_method == 'lloyd':
                # Prototypes random initialization
                prototypes = []
                for (i, feature_vector) in self.Xtrain.sample(self.k, ignore_index=True, random_state=42).iterrows():
                    prototypes.append((i, feature_vector))
            if self.init_method == 'kmeans++':
                prototypes = []
                prototypes.append((0,self.Xtrain.sample(ignore_index=True, random_state=42).to_numpy()[0]))
                print(prototypes)
                for centroid in range(1, self.k):
                    dist_to_prototype = []
                    for (i, feature_vector) in self.Xtrain.iterrows():
                        dist_list = []
                        for (idx, prototype) in prototypes:
                            print(prototype, feature_vector.to_numpy())
                            dist_list.append(self.__calculate_distance(prototype,feature_vector))
                        dist_to_prototype.append(sum(dist_list) / len(dist_list))
                    prototypes[i] = sorted(dist_to_prototype, key = lambda x:x[1])[0][1]

            self.prototypes = prototypes
        else:
            # Prototypes update
            for i in range(self.k):
                cls_vectors = self.Xtrain[self.pred == i]
                self.prototypes[i] = (i, cls_vectors.mean())

    def __calculate_distance(self, vector1, vector2) -> float:
        if len(vector1) != len(vector2):
            raise ValueError("Vectors of different sizes")
        else:
            sum = 0
            for x, y in zip(vector1, vector2):
                sum += abs(x - y) ** self.p
            return sum ** (1 / self.p)

    def __assign_prototype_to_vector(self, feature_vector):
        dist_list = []
        for (idx, prototype) in self.prototypes:

            dist_list.append((idx, self.__calculate_distance(prototype, feature_vector)))
        sorted_list = sorted(dist_list, key=lambda x: x[1])
        return sorted_list[0][0]

    def __check_prototypes_changes(self, prev_iter_prototypes):
        changes_sum = 0
        for i in range(len(self.prototypes)):
            for val1, val2 in zip(self.prototypes[i][1], prev_iter_prototypes[i][1]):
                changes_sum += abs(val1 - val2)

        return changes_sum

    def silhouette_score(self) -> float:
        silhouette = pd.DataFrame(columns=["a", "b", "s"], index=self.Xtrain.index)
        for (i, feature_vector) in self.Xtrain.iterrows():
            vector_class = self.pred.iloc[i]
            silhouette.loc[i, "a"] = self.__calc_a_for_silhouette(feature_vector, vector_class)
            silhouette.loc[i, "b"] = self.__calc_b_for_silhouette(feature_vector, vector_class)
            silhouette.loc[i, "s"] = (silhouette.loc[i, "b"] - silhouette.loc[i, "a"]) / max(
                silhouette.loc[i, "a"], silhouette.loc[i, "b"]
            )
        
        return silhouette.s.mean()

    def __calc_a_for_silhouette(self, feature_vector, cls):
        cls_vectors = self.Xtrain[self.pred == cls]
        a_sum = 0
        for (i, row) in cls_vectors.iterrows():
            a_sum += self.__calculate_distance(row, feature_vector)
        return a_sum / (cls_vectors.shape[0] - 1)

    def __calc_b_for_silhouette(self, feature_vector, cls):
        cls_sum = []
        for i in range(self.k):
            if i != cls:
                cls_vectors = self.Xtrain[self.pred == i]
                b_sum = 0
                for (i, row) in cls_vectors.iterrows():
                    b_sum += self.__calculate_distance(row, feature_vector)
                cls_sum.append((i, b_sum / cls_vectors.shape[0]))
        return min(cls_sum, key=lambda x: x[1])[1]
