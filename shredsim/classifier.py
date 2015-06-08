import cPickle as pickle
import itertools
import logging
import multiprocessing
import os
import re
import time

import cv2
import numpy as np

from nolearn.dbn import DBN
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn import neighbors
from sklearn.metrics import classification_report

import dataset


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


_CLASSIFIER_FNAME_TMPL = os.path.join(dataset.DATADIR, 'classifiers', '%s.dat')


class Profile(object):
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        log.info("%s start.", self.name)
        self.start = time.time()
    def __exit__(self, *args):
        log.info("%s done: %s", self.name, time.time() - self.start)


def _load_dataset_for_label(label):
    samples = []
    gen_dir = os.path.join(dataset.DATADIR, 'gen', label)
    if not os.path.isdir(gen_dir):
        os.makedirs(gen_dir)
    for sample_fname in os.listdir(os.path.join(dataset.DATADIR, 'gen', label)):
        path = os.path.join(dataset.DATADIR, 'gen', label, sample_fname)
        samples.append(cv2.cvtColor(cv2.imread(path),
                                    cv2.cv.CV_BGR2GRAY).reshape((1,2500)))
    return samples

def get_dataset():
    with Profile("Loading the dataset."):
        labels = []

        for src_fname in os.listdir(os.path.join(dataset.DATADIR, 'src')):
            label, ext = os.path.splitext(src_fname)
            if ext != '.png': continue
            labels.append(label)

        samples = []
        labels_for_samples = []

        p = multiprocessing.Pool()
        for samples_for_label, label in itertools.izip(
                p.imap(_load_dataset_for_label, labels),
                labels):
            samples.extend(samples_for_label)
            labels_for_samples.extend([label] * len(samples_for_label))
        p.close()
        p.join()

        log.info("Loaded %d samples.", len(samples))
        return np.concatenate(samples, axis=0), np.array(labels_for_samples)


def train_dbn_classifier(dataset):
    (trainX, testX, trainY, testY) = dataset
    dbn = DBN(
        [trainX.shape[1], 300, len(set(trainY) | set(testY))],
        learn_rates = 0.4,
        learn_rate_decays = 0.9,
        epochs = 50,
        verbose = 1)
    dbn.fit(trainX, trainY)

    return dbn


def train_knn_classifier(dataset):
    (trainX, testX, trainY, testY) = dataset

    classifier = neighbors.KNeighborsClassifier()
    classifier.fit(trainX, trainY)

    return classifier


def train_rf_classifier(dataset):
    # RF classifier is faster to train, but takes around 2G in pickled state and
    # gives only 0.77 precision.
    (trainX, testX, trainY, testY) = dataset
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    classifier.fit(trainX, trainY)
    return classifier


def get_classifier(dataset=None, classifier_type='dbn'):
    assert re.match('^[-a-z]+$', classifier_type)

    classifier_fname = _CLASSIFIER_FNAME_TMPL % classifier_type
    if not os.path.exists(classifier_fname):
        if dataset is None:
            samples, labels = get_dataset()
            dataset = train_test_split(samples / 255.0, labels, test_size = 0.33)

        log.info("Classifier type %s not found. Training one.", classifier_type)

        classifiers = {
            'dbn': train_dbn_classifier,
            'rf': train_rf_classifier,
            'knn': train_knn_classifier,
        }

        if classifier_type in classifiers:
            classifier = classifiers[classifier_type](dataset)
        else:
            raise NotImplementedError("Unknown classifier type: %s" %
                                      classifier_type)

        classifier_dir = os.path.dirname(classifier_fname)
        log.info("Training done. Saving.")
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        pickle.dump(classifier, open(classifier_fname, 'w'))
    else:
        log.info("Found existing classifier. Loading.")
        classifier = pickle.load(open(classifier_fname))

    return classifier


def main():
    samples, labels = get_dataset()

    with Profile("Splitting data"):
        dataset = (trainX, testX, trainY, testY) = train_test_split(
            samples / 255.0, labels, test_size = 0.33)

    classifier = get_classifier(dataset, 'dbn')

    preds = classifier.predict(testX)
    print classification_report(testY, preds)

    #for i in np.random.choice(np.arange(0, len(testY)), size = (10,)):
    #        # classify the digit
    #        pred = dbn.predict(np.atleast_2d(testX[i]))

    #        # reshape the feature vector to be a 28x28 pixel image, then change
    #        # the data type to be an unsigned 8-bit integer
    #        image = (testX[i] * 255).reshape((50, 50)).astype("uint8")

    #        # show the image and prediction
    #        print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])
    #        cv2.imshow("Digit", image)
    #        cv2.waitKey(0)

    #import ipdb;ipdb.set_trace()
if __name__ == '__main__':
    main()
