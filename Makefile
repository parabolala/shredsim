
DATASET_SRC = $(wildcard shredsim/dataset/src/*.png)
DATASET_DST = $(patsubst shredsim/dataset/src/%.png,shredsim/dataset/gen/%/0.png,$(DATASET_SRC))

shredsim/dataset/gen/%/0.png:
	python shredsim/dataset.py $*
dataset: $(DATASET_DST)

shredsim/dataset/classifiers/dbn.zip: dataset
	python shredsim/classifier.py
classifier-dbn: shredsim/dataset/classifiers/dbn.zip

classifier: classifier-dbn


.PHONY: clean clean-dataset clean-classifier
clean-dataset:
	rm -rf shredsim/dataset/gen/*
clean-classifier:
	rm -rf shredsim/dataset/classifiers/*
clean: clean-dataset clean-classifier

