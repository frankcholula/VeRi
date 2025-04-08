.PHONY: section1
section1: train_mobilenet train_vgg train_resnet train_resnet_fc512

.PHONY: section2
section2:
	python experiments/section2/data_augmentation.py --dry-run

.PHONY: section3.1
section3.1:
	python experiments/section3/lr_exploration.py

.PHONY: section3.2
section3.2:
	python experiments/section3/batch_size_exploration.py

.PHONY: section3.3
section3.3:
	python experiments/section3/optimizer_exploration.py

PHONY: train_mobilenet
train_mobilenet:
	experiments/section1/train_mobilenet.shs

.PHONY: train_vgg
train_vgg:
	experiments/section1/train_vgg.sh

.PHONY: train_resnet
train_resnet:
	experiments/section1/train_resnet.sh

.PHONY: train_resnet_fc512
train_resnet_fc512:
	experiments/section1/train_resnet_fc512.sh

.PHONY: train_resnet_fc512_no-aug
train_resnet_fc512_no-aug:
	python main.py \
		-s veri \
		-t veri \
		-a resnet50_fc512 \
		--root src/datasets \
		--height 224 \
		--width 224 \
		--optim amsgrad \
		--lr 0.0003 \
		--max-epoch 10 \
		--stepsize 20 40 \
		--train-batch-size 64 \
		--test-batch-size 100 \
		--save-dir logs/resnet50_fc512_no-aug_10 \
		--no-aug

.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf logs/*
