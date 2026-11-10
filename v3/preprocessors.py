"""Leak-free preprocessing building blocks (impute -> scale -> resample)."""
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler


class HotDeckImputer(BaseEstimator, TransformerMixin):
    """Random hot-deck imputation.

    fit():       remember the observed values of every column in the TRAINING data (the donors).
    transform(): replace each missing cell by a value drawn at random from that column's donors.
    Test rows therefore only ever receive values donated by training rows (no leakage).
    """

    def __init__(self, random_state=0):
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.donors_ = [col[~np.isnan(col)] for col in X.T]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        rng = np.random.RandomState(self.random_state)
        for j in range(X.shape[1]):
            miss = np.isnan(X[:, j])
            if miss.any():
                X[miss, j] = rng.choice(self.donors_[j], size=int(miss.sum()), replace=True)
        return X


IMPUTERS = ["mean", "median", "knn", "hotdeck"]
SCALERS = ["none", "minmax"]
RESAMPLERS = ["none", "smote", "adasyn", "smotetomek"]


def make_imputer(name, seed=0):
    if name == "mean":
        return SimpleImputer(strategy="mean")
    if name == "median":
        return SimpleImputer(strategy="median")
    if name == "knn":
        return KNNImputer(n_neighbors=5)
    if name == "hotdeck":
        return HotDeckImputer(random_state=seed)
    raise ValueError(name)


def make_scaler(name):
    return None if name == "none" else MinMaxScaler()


def make_resampler(name, seed=0):
    if name == "none":
        return None
    if name == "smote":
        return SMOTE(random_state=seed)
    if name == "adasyn":
        return ADASYN(random_state=seed)
    if name == "smotetomek":
        return SMOTETomek(random_state=seed)
    raise ValueError(name)
