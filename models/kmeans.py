import random
import pandas as pd
import numpy as np


class KMeans:
    def __init__(
        self,
        n_clusters: int,
        init: str = "lloyd",
        distance_metric: str = "euclidean",
        max_iter=300,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        if distance_metric is None:
            raise ValueError("distance_metric cant be None")
        if distance_metric == "manhattan":
            self.p = 1
        elif distance_metric == "euclidean":
            self.p = 2

        self.init_method = init
        self.centroids = None

    def fit(self, Xtrain: pd.DataFrame) -> None:
        self.Xtrain = Xtrain

        # Initialize centroids
        self.__get_centroids()

        # Main loop
        self.iterations = 0
        self.prev_centroids = None

        while (
            np.not_equal(self.centroids, self.prev_centroids).any()
            # and self.iterations < self.max_iter
        ):
            self.sorted_vectors = [[] for _ in range(self.n_clusters)]
            for feature_vector in self.Xtrain:
                dists = self.__calc_euclidean(feature_vector, self.centroids)
                centroid_index = np.argmin(dists)
                self.sorted_vectors[centroid_index].append(feature_vector)

            self.__get_centroids()
            self.iterations += 1

    def predict(self, Xtest: pd.DataFrame) -> pd.Series:

        pred = np.ndarray(self.Xtrain)
        for (i, feature_vector) in Xtest.iterrows():
            pred[i] = self.__assign_centroid_to_vector(feature_vector)
        return pred

    def __get_centroids(self) -> list:
        if self.centroids is None:
            if self.init_method == "lloyd":
                # centroids random initialization

                self.centroids = [
                    random.choice(self.Xtrain)
                    for _ in range(self.n_clusters)
                ]

            if self.init_method == "kmeans++":
                self.centroids = [random.choice(self.Xtrain)]
                for _ in range(self.n_clusters - 1):
                    dists = np.sum(
                        [
                            self.__calc_euclidean(centroid, self.Xtrain)
                            for centroid in self.centroids
                        ],
                        axis=0,
                    )
                    dists /= np.sum(dists)
                    new_centroid = np.random.choice(
                        range(len(self.Xtrain)), size=1, p=dists
                    )
                    self.centroids.append(self.Xtrain[new_centroid])

        else:
            # Update centroids
            self.prev_centroids = self.centroids
            self.centroids = [
                np.mean(centroid, axis=0) for centroid in self.sorted_vectors
            ]
            for i in range(self.n_clusters):
                if np.isnan(self.centroids[i]).any():
                    self.centroids[i] = self.prev_centroids[i]

    def __calculate_distance(self, vector1, vector2) -> float:
        if len(vector1) != len(vector2):
            raise ValueError("Vectors of different sizes")
        else:
            sum = 0
            for x, y in zip(vector1, vector2):
                sum += abs(x - y) ** self.p
            return sum ** (1 / self.p)

    def __calc_euclidean(self, vector, centroids):
        return np.sqrt(np.sum((vector - centroids) ** 2, axis=1))

    def __assign_centroid_to_vector(self, feature_vector):
        dist_list = []
        for (idx, centroid) in self.centroids:

            dist_list.append((idx, self.__calculate_distance(centroid, feature_vector)))
        sorted_list = sorted(dist_list, key=lambda x: x[1])
        return sorted_list[0][0]

    def evaluate(self):
        centroids = []
        centroid_idxs = []
        for x in self.Xtrain:
            dists = self.__calc_euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs

    def silhouette_score(self) -> float:
        silhouette = pd.DataFrame(columns=["a", "b", "s"], index=self.Xtrain.index)
        for (i, feature_vector) in self.Xtrain.iterrows():
            vector_class = self.pred.iloc[i]

            silhouette.loc[i, "a"] = self.__calc_a_for_silhouette(
                feature_vector, vector_class
            )
            silhouette.loc[i, "b"] = self.__calc_b_for_silhouette(
                feature_vector, vector_class
            )
            silhouette.loc[i, "s"] = (
                silhouette.loc[i, "b"] - silhouette.loc[i, "a"]
            ) / max(silhouette.loc[i, "a"], silhouette.loc[i, "b"])

        return silhouette.s.mean()

    def __calc_a_for_silhouette(self, feature_vector, cls):
        cls_vectors = self.Xtrain[self.pred == cls]
        a_sum = 0
        for (i, row) in cls_vectors.iterrows():
            a_sum += self.__calculate_distance(row, feature_vector)
        return a_sum / (cls_vectors.shape[0] - 1)

    def __calc_b_for_silhouette(self, feature_vector, cls):
        cls_sum = []
        for i in range(self.n_clusters):
            if i != cls:
                cls_vectors = self.Xtrain[self.pred == i]
                b_sum = 0
                for (i, row) in cls_vectors.iterrows():
                    b_sum += self.__calculate_distance(row, feature_vector)
                cls_sum.append((i, b_sum / cls_vectors.shape[0]))
        return min(cls_sum, key=lambda x: x[1])[1]
