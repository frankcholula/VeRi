.PHONY: train_mobilenet
train_mobilenet:
	python main.py \
	-s veri \
	-t veri \
	-a mobilenet_v3_small \
	--root src/datasets/ \
	--height 224 \
	--width 224 \
	--optim amsgrad \
	--lr 0.0003 \
	--max-epoch 10 \
	--stepsize 20 40 \
	--train-batch-size 64 \
	--test-batch-size 100 \
	--save-dir logs/mobilenet_v3_small-veri

.PHONY: train_resnet18
train_resnet18:
	python main.py \
	-s veri \
	-t veri \
	-a resnet18 \
	--root src/datasets/ \
	--height 224 \
	--width 224 \
	--optim amsgrad \
	--lr 0.0003 \
	--max-epoch 10 \
	--stepsize 20 40 \
	--train-batch-size 64 \
	--test-batch-size 100 \
	--save-dir logs/resnet18-veri

.PHONY: train_resnet34
train_resnet18:
	python main.py \
	-s veri \
	-t veri \
	-a resnet34 \
	--root src/datasets/ \
	--height 224 \
	--width 224 \
	--optim amsgrad \
	--lr 0.0003 \
	--max-epoch 10 \
	--stepsize 20 40 \
	--train-batch-size 64 \
	--test-batch-size 100 \
	--save-dir logs/resnet34-veri

.PHONY: train_resnet50
train_resnet50:
	python main.py \
	-s veri \
	-t veri \
	-a resnet50 \
	--root src/datasets/ \
	--height 224 \
	--width 224 \
	--optim amsgrad \
	--lr 0.0003 \
	--max-epoch 10 \
	--stepsize 20 40 \
	--train-batch-size 64 \
	--test-batch-size 100 \
	--save-dir logs/resnet50-veri

.PHONY: train_vgg16
train_vgg16:
	python main.py \
	-s veri \
	-t veri \
	-a vgg16 \
	--root src/datasets/ \
	--height 224 \
	--width 224 \
	--optim amsgrad \
	--lr 0.0003 \
	--max-epoch 10 \
	--stepsize 20 40 \
	--train-batch-size 64 \
	--test-batch-size 100 \
	--save-dir logs/vgg16-veri

.PHONY: eval
eval:
	python main.py \
	--evaluate \
	--visualize-ranks \
	--resume logs/mobilenet_v3_small-veri/model.pth.tar-10 \
	-s veri \
	-t veri \
	-a mobilenet_v3_small \
	--root src/datasets/ \
	--height 224 \
	--width 224 \
	--test-batch-size 100 \
	--gpu-devices 0 \
	--save-dir logs/mobilenet_v3_small-veri
