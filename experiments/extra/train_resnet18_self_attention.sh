echo "==== Running ResNet50 (Self-Attention) with Default Settings ===="

python main.py \
-s veri \
-t veri \
-a resnet18_self_attention \
--root src/datasets/ \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 50 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir logs/resnet18_self_attention-veri