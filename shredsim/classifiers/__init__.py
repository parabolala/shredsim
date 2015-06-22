"""Contains the base classes for all tested classifiers."""
import cPickle as pickle


class ClassifierBase(object):
    """Base abstract class for all classifiers.

    Classifiers must implement at least the following methods:
        train(self, dataset)
        predict(X)
    """
    def __init__(self):
        pass

    def train(self, dataset):
        """Trains a classifier.

        Called once when the classifier is first constructed.

        Args:
            dataset: a 2-tuple (trainX, trainY).
        """
        raise NotImplementedError()

    def save(self, fname):
        """Serializes a classifier to a file with a given name."""
        with open(fname, 'w') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fname, dataset):
        """Loads the classifier from a file with the given name.

        Args:
            fname: base file name to load the classifier from.
            dataset: a 4-tuple (trainX, trainY, testX, testY) for recovering a
                classifier.

        Returns:
            An instance of the current class.
        """
        with open(fname) as f:
            return pickle.load(f)

class StatelessClassifierBase(ClassifierBase):
    """A base class for classifiers that don't store any state.

    Instead on load it trains a new classifier on the given data.

    This is useful for classifiers that might take too much space, but are fast
    to train, e.g. various nearest neighbors methods.
    """
    def save(self, fname):
        with open(fname, 'w') as f:
            f.write('empty')

    def load(self, fname, dataset):
        self.train((dataset[0], dataset[2]))
        return self
