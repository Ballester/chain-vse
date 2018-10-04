
# python train.py \
# --data_path /opt/datasets/chain/coco/ \
# --data_name 10resnet152_precomp \
# --img_dim 2048 \
# --text_encoder gru \
# --max_violation \
# --vocab ./vocab \
# --add_data \
# --noise 0.0 \
# --use_restval \
# --consistency_weight 0.

python train.py \
--data_path /opt/datasets/chain/coco/ \
--data_name 10resnet152_precomp \
--img_dim 2048 \
--text_encoder gru \
--max_violation \
--vocab ./vocab \
--add_data \
--noise 0.0 \
--use_restval \
--consistency_weight 20.
