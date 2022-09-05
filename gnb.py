import numpy as np
import pandas as pd


class GNB:
    def __init__(self):
        self.cls_list = None
        self.cls_priors = None
        self.mean_likelihood = None
        self.std_likelihood = None
        self.aposterior_probs = None

    def fit(self, Xtrain: pd.DataFrame, ytrain: pd.Series) -> None:
        self.cls_list = sorted(ytrain.unique().tolist())
        self.cls_priors = self.__calc_class_priors(ytrain)
        self.mean_likelihood = self.__calc_mean_for_likelihood(Xtrain, ytrain)
        self.std_likelihood = self.__calc_std_for_likelihood(Xtrain, ytrain)

    def predict(self, xtest: pd.DataFrame) -> pd.Series:
        self.aposterior_probs = self.__calc_aposterior_probs(xtest)
        return self.aposterior_probs.idxmax(axis=1)

    def __calc_class_priors(self, ytrain: pd.Series):
        lst_priors = []
        for cls in self.cls_list:
            lst_priors.append((ytrain.value_counts()[cls]) / ytrain.shape[0])

        return lst_priors

    def __calc_mean_for_likelihood(self, xtrain: pd.DataFrame, ytrain: pd.Series):
        df_mean_likelihood = pd.DataFrame(columns=xtrain.columns, index=self.cls_list)
        for feature in df_mean_likelihood.columns:
            for cls in self.cls_list:
                cls_instance = ytrain == cls
                df_mean_likelihood.loc[cls, feature] = xtrain[cls_instance][
                    feature
                ].mean()

        return df_mean_likelihood

    def __calc_std_for_likelihood(self, xtrain: pd.DataFrame, ytrain: pd.Series):
        df_std_likelihood = pd.DataFrame(columns=xtrain.columns, index=self.cls_list)
        for feature in df_std_likelihood.columns:
            for cls in self.cls_list:
                cls_instance = ytrain == cls
                df_std_likelihood.loc[cls, feature] = xtrain[cls_instance][
                    feature
                ].std()

        return df_std_likelihood

    def __calc_gaussian_pdf_prob(self, x_feature_val, feature_mean, feature_std):
        exponent = np.exp(
            -((x_feature_val - feature_mean) ** 2 / (2 * feature_std**2))
        )
        return (1 / ((2 * np.pi) ** (1 / 2) * feature_std)) * exponent

    def __calc_aposterior_probs(self, Xtest: pd.DataFrame):
        num_classes = len(self.cls_list)
        df_aposterior_probs = pd.DataFrame(
            np.zeros((Xtest.shape[0], num_classes)),
            columns=self.cls_list,
            index=Xtest.index,
        )

        for row in Xtest.iterrows():
            for cls in self.cls_list:
                calc_apos = self.cls_priors[cls]
                for j, feature in enumerate(row[1]):
                    calc_apos = calc_apos * self.__calc_gaussian_pdf_prob(
                        feature,
                        self.mean_likelihood.iloc[cls, j],
                        self.std_likelihood.iloc[cls, j],
                    )
                df_aposterior_probs.loc[row[0], cls] = calc_apos

        return df_aposterior_probs

    def calc_accuracy(self, ypred, ytest):
        correct = (ytest == ypred).sum()
        accuracy = correct / ytest.shape[0]
        return accuracy
