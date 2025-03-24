.PHONY: train
train:
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
