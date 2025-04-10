echo "==== Running ResNet18_se with Default Settings ===="
python main.py \
-s veri \
-t veri \
-a resnet18_se \
--root src/datasets/ \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0001 \
--max-epoch 20 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir logs/resnet18_se-veri
