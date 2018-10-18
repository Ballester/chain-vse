

python train.py \
--data_path /home/teia/datasets/vse/ \
--data_name 10resnet152_precomp \
--adapt_data flickr30k_precomp \
--adapt_split train \
--adapt_batch_size 128 \
--val_data flickr30k_precomp \
--val_split val \
--val_batch_size 128 \
--img_dim 2048 \
--logger_name runs/10resnet152_precomp/da_flickr30_t0/ \
--text_encoder gru \
--max_violation \
--add_data \
--use_restval \
--noise 0.1 \
--consistency_weight 20. \
--vocab ./vocab \
--consistency_rampup 10 \
--ema_late_epoch 5 \
--ramp_lr \
--initial_lr_rampup 2 \
--num_epochs 50

