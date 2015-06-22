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

    # Install OpenCV+python in your favorite way.
    # e.g. sudo port install opencv +debug+python27+tbb 
    # or   sudo apt-get install python-opencv

    # Next install all the deps and the package itself.
    pip install .


Generate the training dataset:

    make dataset

Train the Deep Belief Network classifier:

    make classifier

Or a LSH one:

    make classifier-lsh


Explore the results in an IPython notebook:

    ipython notebook Main.ipynb

Optional components
---

`cudamat` speeds up DBN training.

`opencv` with TBB support helps OpenCV MLP ANN.

[unshred-tag]: https://github.com/dchaplinsky/unshred-tag
