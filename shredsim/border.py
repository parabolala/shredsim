import cv2
import numpy as np

import dataset


class ShredMaskBorder(object):
    _erode_kernel = np.ones((3, 3))
    def __init__(self, shred_mask, border_depth=dataset.WINDOW_SIDE / 4,
                 border_thickness=1):
        self._shred_mask = shred_mask
        self._border_depth = border_depth
        self._border_thickness = border_thickness


    @property
    def _erode_iterations_outer(self):
        return self._border_depth

    @property
    def _erode_iterations_inner(self):
        return self._border_depth + self._border_thickness

    def get_border_mask(self):
        outer_erosion = cv2.erode(self._shred_mask, self._erode_kernel,
                                  iterations=self._erode_iterations_outer)
        inner_erosion = cv2.erode(self._shred_mask, self._erode_kernel,
                                  iterations=self._erode_iterations_inner)

        return outer_erosion - inner_erosion

    def get_border_points(self):
        return np.transpose((self.get_border_mask() != 0).nonzero())


