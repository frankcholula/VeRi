echo "==== Running ResNet18_se_fc512 with Optimized Settings ===="
python main.py \
-s veri \
-t veri \
-a resnet50_se_fc512 \
--root src/datasets/ \
--height 224 \
--width 224 \
--optim amsgrad \
--lr 0.0003 \
--max-epoch 10 \
--stepsize 20 40 \
--train-batch-size 64 \
--test-batch-size 100 \
--save-dir logs/resnet50_se_fc512-veri
