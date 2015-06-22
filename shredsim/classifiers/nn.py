"""A simple parallel nearest neighbors classifier.


This one is parallelized using multiprocessing. It splits the training data into
chunks and processes them in parallel using a pool of worker processes.
"""
import multiprocessing

import numpy as np

from shredsim import classifiers


def _init_nn_global(init_all_samples):
    global _all_samples
    _all_samples = init_all_samples


def _find_closest_idx(arg):
    image, range = arg
    global all_samples

    all_distances = [(np.linalg.norm(image - sample), idx)
                     for idx, sample in enumerate(_all_samples[range])]

    closest = min(all_distances)
    closest = closest[0], closest[1] + range.start

    return closest


class NNClassifier(classifiers.StatelessClassifierBase):
    _pool = None

    def __init__(self):
        pass

    def train(self, dataset):
        self._all_samples, self._all_labels = dataset
        num_ranges = multiprocessing.cpu_count()
        items_per_range = int(np.ceil(
            float(len(self._all_samples)) / num_ranges))
        self._ranges = [slice(i * items_per_range, (i+1)*items_per_range)
                        for i in range(num_ranges)]
        self._init_pool()

    def _init_pool(self):
        if self._pool is None:
            self._pool = multiprocessing.Pool(processes=None,
                                              initializer=_init_nn_global,
                                              initargs=(self._all_samples,))

    def _predict_one(self, x):
        args = ((x, range) for range in self._ranges)
        closest_per_range = self._pool.map(_find_closest_idx, args)
        res = min(closest_per_range)[1]
        print 'predict idx', res
        return self._all_labels[res]

    def predict(self, X):
        return map(self._predict_one, X)

    def close(self):
        self._pool.close()
        self._pool.join()
        self._pool = None

    def __del__(self):
        if self._pool is not None:
            self.close()

