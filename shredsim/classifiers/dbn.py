"""Implements a deep belief network classifer.

The source code is dervied from
http://www.pyimagesearch.com/2014/09/22/getting-started-deep-learning-python/
"""
from nolearn.dbn import DBN

from shredsim import classifiers


class DBNClassifer(classifiers.ClassifierBase):
    _HIDDEN_LAYER = 300

    def __init__(self):
        self._dbn = None

    def train(self, dataset):
        (trainX, trainY) = dataset
        dbn = DBN(
            [trainX.shape[1], 300, len(set(trainY))],
            learn_rates = 0.5,
            learn_rate_decays = 0.9,
            epochs = 100,
            verbose = 1)
        dbn.fit(trainX, trainY)

        self._dbn = dbn

    def predict(self, X):
        return self._dbn.predict(X)

    def predict_proba(self, *args):
        return self._dbn.predict_proba(*args)
