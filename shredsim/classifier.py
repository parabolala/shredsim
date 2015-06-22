import itertools
import logging
import multiprocessing
import os
import re
import sys
import time

import cv2
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from classifiers import dbn
from classifiers import nn
from classifiers import lsh
from classifiers import opencv_knn
from classifiers import opencv_ann

import dataset


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


DEFAULT_CLASSIFIER_TYPE = 'dbn'

_DATASET = None
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
                                    cv2.cv.CV_BGR2GRAY).reshape(
                                        (1,2500)).astype(np.float32))
    return samples

def get_dataset():
    global _DATASET
    if _DATASET is None:
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
            _DATASET = np.concatenate(samples, axis=0), np.array(labels_for_samples)
    return _DATASET

def get_classifier(dataset=None, classifier_type='dbn'):
    """

    Args:
        dataset: optional dataset to use for training/loading. If given, must be
            a 4-tuple: (trainX, testX, trainY, testY).
        classifier_type: string classifier type.

    Returns:
        A classifiers.ClassifierBase instance.
    """
    assert re.match('^[-a-z_]+$', classifier_type)

    classifiers = {
        'dbn': dbn.DBNClassifer,
        'nn': nn.NNClassifier,
        'lsh': lsh.LSHClassifer,
        'opencv_knn': opencv_knn.OpenCVKNNClassifier,
        'opencv_ann': opencv_ann.OpenCVANNClassifier,
    }

    classifier = classifiers[classifier_type]()

    classifier_fname = _CLASSIFIER_FNAME_TMPL % classifier_type

    if dataset is None:
        samples, labels = get_dataset()
        samples[0] /= 255
        dataset = trainX, testX, trainY, testY = train_test_split(
                samples, labels, test_size=0.03)
    else:
        trainX, testX, trainY, testY = dataset
        samples = np.concatenate([trainX, testX], axis=0)
        labels = np.concatenate([trainY, testY], axis=0)

    if not os.path.exists(classifier_fname):
        with Profile("Classifier type %s not found. Training one." % classifier_type):
            classifier.train((samples, labels))

        classifier_dir = os.path.dirname(classifier_fname)
        log.info("Training done. Saving.")
        if not os.path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        #classifier.save(classifier_fname)
        log.info("Saved")
    else:
        log.info("Found existing classifier. Loading.")
        classifier = classifier.load(classifier_fname, dataset)

    return classifier


def main(argv):
    samples, labels = get_dataset()

    with Profile("Splitting data"):
        dataset = (trainX, testX, trainY, testY) = train_test_split(
            samples / 255.0, labels, test_size = 0.33)

    classifier_type = DEFAULT_CLASSIFIER_TYPE
    if len(argv) > 1:
        classifier_type = argv[1]

    classifier = get_classifier(dataset, classifier_type)

    with Profile("Getting test predictions for %d samples." % testX.shape[0]):
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
    main(sys.argv)
