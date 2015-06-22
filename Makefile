
DATASET_SRC = $(wildcard shredsim/dataset/src/*.png)
DATASET_DST = $(patsubst shredsim/dataset/src/%.png,shredsim/dataset/gen/%/0.png,$(DATASET_SRC))

CLASSIFIERS = dbn nn opencv_ann opencv_knn lsh
ALL_CLASSIFIERS = $(patsubst %,shredsim/dataset/classifiers/%.dat,$(CLASSIFIERS))

shredsim/dataset/gen/%/0.png: shredsim/dataset.py
	python shredsim/dataset.py $*
dataset: $(DATASET_DST)

shredsim/dataset/classifiers/%.dat:
	python shredsim/classifier.py $*

classifier-%: shredsim/dataset/classifiers/%.dat ;

classifier: classifier-dbn
all-classifiers: $(ALL_CLASSIFIERS)


.PHONY: clean clean-dataset clean-classifiers
clean-dataset:
	rm -rf shredsim/dataset/gen/*
clean-classifiers:
	rm -rf shredsim/dataset/classifiers/*
clean: clean-dataset clean-classifiers

