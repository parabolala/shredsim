"""A K nearest neighbors classifier, OpenCV implementation."""
import cv2
import numpy as np

from shredsim import classifiers


class OpenCVKNNClassifier(classifiers.StatelessClassifierBase):
    def __init__(self):
        self.knn = None

    def train(self, dataset):
        self.knn = cv2.KNearest()
        self.knn.train(dataset[0],
                       self._labels_to_label_idx(dataset[1]),
                       isRegression=False, maxK=1)

    def _labels_to_label_idx(self, labels):
        self._labels = list(set(labels))
        self._label_to_ids = {
                label: idx for (idx, label) in enumerate(self._labels)}
        return np.array([self._label_to_ids[label] for label in labels])

    def predict(self, X):
        ret, results, neighbours, dist = self.knn.find_nearest(
                X.astype(np.float32), 1)
        return [self._labels[int(idx[0])] for idx in results]

