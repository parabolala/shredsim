"""Locality Sensitive Hashing forest classifier using sklearn implementation."""
from sklearn import neighbors

from shredsim import classifiers


class LSHClassifer(classifiers.StatelessClassifierBase):
    def __init__(self):
        self._dbn = None

    def train(self, dataset):
        (trainX, trainY) = dataset
        lshf = neighbors.LSHForest(n_candidates=1)
        lshf.fit(trainX, trainY)

        self._trainY = trainY
        self._lshf = lshf

    def predict(self, X):
        distances, indices = self._lshf.kneighbors(X, 1)
        return self._trainY[indices[:,0]]
