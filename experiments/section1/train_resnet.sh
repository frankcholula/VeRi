#!/bin/bash

echo "==== Running ResNet18 with Default Settings ===="
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

echo "==== Running ResNet34 with Default Settings ===="
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

echo "==== Running ResNet50 with Default Settings ===="
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
