

python train.py \
--data_path /A/VSE/data/ \
--data_name coco_precomp \
--adapt_data f30k_precomp \
--adapt_split train \
--adapt_batch_size 128 \
--val_data f30k_precomp \
--val_split val \
--val_batch_size 128 \
--img_dim 2048 \
--logger_name runs/coco_f30k_da_c1e-1/ \
--text_encoder gru \
--max_violation \
--add_data \
--use_restval \
--noise 0.01 \
--dropout_noise 0.10 \
--consistency_weight 0.1 \
--consistency_rampup 15 \
--ema_late_epoch 15 \
--vocab ./vocab \



python train.py \
--data_path /A/VSE/data/ \
--data_name coco_precomp \
--adapt_data f30k_precomp \
--adapt_split train \
--adapt_batch_size 128 \
--val_data f30k_precomp \
--val_split val \
--val_batch_size 128 \
--img_dim 2048 \
--logger_name runs/coco_f30k_da_c1e-2/ \
--text_encoder gru \
--max_violation \
--add_data \
--use_restval \
--noise 0.01 \
--dropout_noise 0.10 \
--consistency_weight 0.01 \
--consistency_rampup 15 \
--ema_late_epoch 15 \
--vocab ./vocab \


python train.py \
--data_path /A/VSE/data/ \
--data_name coco_precomp \
--adapt_data f30k_precomp \
--adapt_split train \
--adapt_batch_size 128 \
--val_data f30k_precomp \
--val_split val \
--val_batch_size 128 \
--img_dim 2048 \
--logger_name runs/coco_f30k_da_c1e-3/ \
--text_encoder gru \
--max_violation \
--add_data \
--use_restval \
--noise 0.01 \
--dropout_noise 0.10 \
--consistency_weight 0.001 \
--consistency_rampup 15 \
--ema_late_epoch 15 \
--vocab ./vocab \
