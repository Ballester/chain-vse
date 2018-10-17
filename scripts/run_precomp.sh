
# python train.py \
# --data_path /opt/datasets/chain/coco/ \
# --data_name 10resnet152_precomp \
# --img_dim 2048 \
# --text_encoder gru \
# --logger_name runs/baseline \
# --max_violation \
# --vocab ./vocab \
# --add_data \
# --noise 0.0 \
# --use_restval \
# --consistency_weight 0.

# python train.py \
# --data_path /opt/datasets/chain/coco/ \
# --data_name 10resnet152_precomp \
# --img_dim 2048 \
# --text_encoder gru \
# --logger_name runs/ramplr/ \
# --max_violation \
# --vocab ./vocab \
# --add_data \
# --noise 0.0 \
# --use_restval \
# --ramp_lr 

# python train.py \
# --data_path /opt/datasets/chain/coco/ \
# --data_name 10resnet152_precomp \
# --logger_name runs/noise_0.1_weight_10 \
# --img_dim 2048 \
# --text_encoder gru \
# --max_violation \
# --vocab ./vocab \
# --add_data \
# --noise 0.1 \
# --use_restval \
# --consistency_weight 10.


# python train.py \
# --data_path /opt/datasets/chain/coco/ \
# --data_name 10resnet152_precomp \
# --logger_name runs/noise_0.1_weight_5 \
# --img_dim 2048 \
# --text_encoder gru \
# --max_violation \
# --vocab ./vocab \
# --add_data \
# --noise 0.1 \
# --use_restval \
# --consistency_weight 5.


# python train.py \
# --data_path /opt/datasets/chain/coco/ \
# --data_name 10resnet152_precomp \
# --logger_name runs/noise_0.1_weight_10_lr1e-3_10 \
# --img_dim 2048 \
# --text_encoder gru \
# --max_violation \
# --vocab ./vocab \
# --add_data \
# --noise 0.1 \
# --use_restval \
# --consistency_weight 10. \
# --learning_rate 1e-3 \
# --lr_update 10


# python train.py \
#  --data_path /opt/datasets/chain/coco/ \
#  --data_name 10resnet152_precomp \
#  --img_dim 2048 \
#  --logger_name runs/noise_0.1/ \
#  --text_encoder gru \
#  --max_violation \
#  --vocab ./vocab \
#  --add_data \
#  --noise 0.1 \
#  --use_restval \
#  --consistency_weight 20.


# python train.py \
# --data_path /opt/datasets/chain/coco/ \
# --data_name 10resnet152_precomp \
# --img_dim 2048 \
# --logger_name runs/noise_0.01/ \
# --text_encoder gru \
# --max_violation \
# --vocab ./vocab \
# --add_data \
# --noise .01 \
# --use_restval \
# --consistency_weight 20.



# python train.py \
# --data_path /opt/datasets/chain/coco/ \
# --data_name 10resnet152_precomp \
# --img_dim 2048 \
# --logger_name runs/noise_0.1_c10_lrB/ \
# --text_encoder gru \
# --max_violation \
# --vocab ./vocab \
# --add_data \
# --noise .1 \
# --use_restval \
# --consistency_weight 10. \
# --learning_rate 1e-3 \
# --lr_update 10 \


# python train.py \
# --data_path /opt/datasets/chain/coco/ \
# --data_name 10resnet152_precomp \
# --img_dim 2048 \
# --logger_name runs/noise_0.1_c10_lrC/ \
# --text_encoder gru \
# --max_violation \
# --vocab ./vocab \
# --add_data \
# --noise .1 \
# --use_restval \
# --consistency_weight 10. \
# --learning_rate 4e-4 \
# --lr_update 5 \
# --lr_decay 0.5 \
