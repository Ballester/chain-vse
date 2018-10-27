

python train.py \
--data_path /A/VSE/data/ \
--data_name coco_precomp \
--adapt_data coco_precomp \
--adapt_split train \
--adapt_batch_size 1 \
--val_data f30k_precomp \
--val_split val \
--val_batch_size 128 \
--img_dim 2048 \
--logger_name runs/coco_f30k_baseline/ \
--text_encoder gru \
--max_violation \
--add_data \
--use_restval \
--noise 0.01 \
--dropout_noise 0.10 \
--consistency_weight 0. \
--vocab ./vocab \

