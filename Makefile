shredsim/dataset/gen/generated:
	python shredsim/dataset.py && touch $@
data: shredsim/dataset/gen/generated

shredsim/dataset/classifiers/dbn.zip: data
	python shredsim/classifier.py
classifier-dbn: shredsim/dataset/classifiers/dbn.zip

classifier: classifier-dbn


.PHONY: clean clean-dataset clean-classifier
clean-dataset:
	rm -rf shredsim/dataset/gen/*
clean-classifier:
	rm -rf shredsim/dataset/classifiers/*
clean: clean-dataset clean-classifier

