import logging
import os
import sys

import cv2
import numpy as np


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


_MYDIR = os.path.realpath(os.path.dirname(__file__))

DATADIR = os.path.join(_MYDIR, "dataset")
WINDOW_SIDE = 50
WINDOW_SHAPE = np.array((WINDOW_SIDE, WINDOW_SIDE))

def clean_image(rgba_image):
    """Takes alpha channel of the given image and strips empty borders.

    Args:
        rgba_image: 4-channel image numpy array.

    Returns:
        2D numpy array of the image alpha channel with empty borders stripped.
    """
    image = rgba_image[:,:,3] # Just alpha-channel
    for axis in [0, 1]:
        for i in list(range(image.shape[axis])[::-1]):
            line = np.split(image, [i,i+1], axis=axis)[1]
            if np.all(line==0):
                image = np.delete(image, i, axis=axis)
    return image


def get_dataset():
    """Loads source characters dataset.

    Returns:
        A dict mapping labels (characers) to their 2d images.
    """
    res = {}

    dataset_src_dir = os.path.join(DATADIR, 'src')
    for f in os.listdir(dataset_src_dir):
        label, ext = os.path.splitext(f)
        if ext != '.png':
            continue
        # -1 for alpha channel.
        src_image = cv2.imread(os.path.join(dataset_src_dir, f), -1)
        res[label] = clean_image(src_image)
    return res


def to_slice(start, size):
    """Creates a slice tuple for the given window.

    Args:
        start: 2-tuple of slice start coordinates.
        size: 2-tuple of window size.

    Returns:
        A tuple of slice objects for the requested window.
    """
    return tuple(slice(start_i, start_i + size_i)
                 for start_i, size_i in zip(start, size))


def pad_image(image, padding_size):
    """Adds zero padding around the image.

    Args:
        image: 2D numpy array instance.
        padding_size: border size of padding to add.

    Returns:
        A padded 2D image.
    """
    padding_offset = np.array((padding_size, padding_size))
    padded_shape = np.array(image.shape) + padding_offset * 2

    res = np.zeros(shape=padded_shape, dtype=image.dtype)
    s = to_slice(padding_offset, image.shape)
    res[s] = image
    return res


def _all_window_coords(image, window_shape=WINDOW_SHAPE):
    """Generates coordinates for all windows of a given size inside an image.

    Args:
        image: a 2D numpy array image.
        window_shape: 2-tuple of a window size to assume.

    Yields:
        2-tuples of window coordinates.
    """
    for i in range(0, image.shape[0] - window_shape[0]):
        for j in range(0, image.shape[1] - window_shape[1]):
            yield (i, j)

def _non_empty_windows(image, window_shape=WINDOW_SHAPE):
    """Returns non-zero submatrices of specified size from a given image.

    Args:
        image: source images to cut windows from.
        window_shape: 2-tuple of windows shape.

    Yields:
        2D numpy arrays of subimages of requested size.
    """
    skipped_some = False
    for i, j in _all_window_coords(image, window_shape):
        idx = to_slice((i,j), window_shape)
        window = image[idx]

        # More than 1% filled.
        num_non_zeros = np.count_nonzero(window)
        if (float(num_non_zeros) / window.size) > 0.01:
            yield window
        else:
            if not skipped_some:
                log.warning("Skipping empty window.")
                skipped_some = True


def main(argv=[], datadir=DATADIR):
    dataset = get_dataset()

    labels_requested = argv[1:] or None
    for label, image in sorted(dataset.items()):
        if labels_requested is not None and label not in labels_requested:
            continue

        log.info("Generating data for label: %s", label)

        gen_dir = os.path.join(datadir, 'gen', label)
        if not os.path.exists(gen_dir):
            os.mkdir(gen_dir)
        else:
            cleaning = False
            for existing in os.listdir(gen_dir):
                if not cleaning:
                    log.info("Cleaning existing data for label: %s",
                             label)
                    cleaning = True
                os.unlink(os.path.join(gen_dir, existing))

        i = -1
        for i, w in enumerate(_non_empty_windows(pad_image(image,
                                                           WINDOW_SIDE * 0.5))):
            fname = os.path.join(gen_dir, "%d.png" % i)
            cv2.imwrite(fname, w)
        if i == -1:
            logging.error("No good images found for label %s", label)
            sys.exit(1)
        log.info("Wrote %d images for label: %s", i, label)


if __name__ == '__main__':
    main(sys.argv)

