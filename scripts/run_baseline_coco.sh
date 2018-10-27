

python train.py \
--data_path /A/VSE/data/ \
--data_name coco_precomp \
--adapt_data coco_precomp \
--adapt_split train \
--adapt_batch_size 1 \
--val_data coco_precomp \
--val_split val \
--val_batch_size 128 \
--img_dim 2048 \
--logger_name runs/coco_baseline/ \
--text_encoder gru \
--max_violation \
--add_data \
--use_restval \
--noise 1e-2 \
--dropout_noise 0.1 \
--consistency_weight 0. \
--vocab ./vocab \


python train.py \
--data_path /A/VSE/data/ \
--data_name coco_precomp \
--adapt_data coco_precomp \
--adapt_split train \
--adapt_batch_size 1 \
--val_data coco_precomp \
--val_split val \
--val_batch_size 128 \
--img_dim 2048 \
--logger_name runs/coco_baseline_noiseless/ \
--text_encoder gru \
--max_violation \
--add_data \
--use_restval \
--noise 1e-7 \
--dropout_noise 0.0 \
--consistency_weight 0. \
--vocab ./vocab \


