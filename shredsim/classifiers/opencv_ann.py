"""Neural network classifier using OpenCV ANN implementation."""
import cPickle as pickle

import cv2
import numpy as np
from sklearn import preprocessing

from shredsim.classifiers import ClassifierBase


class OpenCVANNClassifier(ClassifierBase):
    #MAX_TRAINING_CHUNK = 214750
    MAX_TRAINING_CHUNK = 100000

    def __init__(self):
        pass

    def load(self, fname, dataset):
        self.nn = cv2.ANN_MLP()
        self.nn.load(fname)
        self.lb = pickle.load(open(fname + '_lb'))
        return self

    def save(self, fname):
        self.nn.save(fname)
        pickle.dump(self.lb, open(fname + '_lb', 'w'))

    def train(self, dataset):
        all_samples, all_labels = dataset
        nn_config = np.array(
                (all_samples.shape[1], 300, len(set(all_labels))),
                dtype=np.int32)
        nn = cv2.ANN_MLP(nn_config)

        trainX = dataset[0]
        self.lb = preprocessing.LabelBinarizer()
        trainY = self.lb.fit_transform(dataset[1]).astype(np.float32)

        # Have to split into batches. See http://code.opencv.org/issues/4407.
        batch = 0
        while batch * self.MAX_TRAINING_CHUNK < trainX.shape[0]:
            print 'training batch %d of %d' % (batch+1, 1+trainX.shape[0]/self.MAX_TRAINING_CHUNK)
            batch_slice = (slice(self.MAX_TRAINING_CHUNK * batch,
                                 self.MAX_TRAINING_CHUNK * (batch + 1)),
                           slice(None, None))
            trainXbatch = trainX[batch_slice]
            trainYbatch = trainY[batch_slice]
            sampleWeights = np.ones((len(trainYbatch), 1))
            nn.train(trainXbatch, trainYbatch, sampleWeights=sampleWeights)
            batch += 1
        self.nn = nn

    def predict(self, X):
        outputs = self.nn.predict(X.astype(np.float32))[1]
        return self.lb.inverse_transform(outputs)
