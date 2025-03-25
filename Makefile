.PHONY: section1
section1: train_mobilenet train_vgg train_resnet train_resnet_fc512

PHONY: train_mobilenet
train_mobilenet:
	experiments/section1/train_mobilenet.sh

.PHONY: train_vgg
train_vgg:
	experiments/section1/train_vgg.sh

.PHONY: train_resnet
train_resnet:
	experiments/section1/train_resnet.sh

.PHONY: train_resnet_fc512
train_resnet_fc512:
	experiments/section1/train_resnet_fc512.sh

.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf logs/*
