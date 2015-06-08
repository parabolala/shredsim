shredsim
===

**shredsim** is a set of tools for simulating paper shredders and evaluating
document recovery approaches.

The project is closely related to the [unshred-tag] effort.

The only codepath being evaluated at the moment is the code for detecting the characters at the edges of the shreds.

The code in the current form:

1. Takes a set of "golden" character images and generates lots of "windows" of these characters' sections: `dataset.py`.
1. Takes the generated windows and trains a classifier to determine the character by its section: `classifier.py`.
1. Splits the included document into shreds and tries to apply the classifier from 1) to the edges of the cut shreds to find out which characters are on the edges: `Main.ipynb`.


Installation
---

Create a virtualenv, activate. e.g. using virtualenvwrapper:

    mkvirtualenv shredsim

Install the dependencies:

    pip install -r requirements.txt
    # Next install OpenCV+python in your favorite way.

Generate the training dataset:

    make dataset

Train the Deep Belief Network classifier:

    make classifier

Explore the results in an IPython notebook:

    ipython notebook Main.ipynb

[unshred-tag]: https://github.com/dchaplinsky/unshred-tag
