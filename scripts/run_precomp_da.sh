

python train.py \
--data_path /home/teia/datasets/chain/ \
--data_name 10resnet152_precomp \
--adapt_data flickr30k_precomp \
--adapt_split train \
--adapt_batch_size 128 \
--val_data flickr30k_precomp \
--val_split val \
--val_batch_size 128 \
--img_dim 2048 \
--logger_name runs/10resnet152_precomp/da_flickr30_default/ \
--text_encoder gru \
--max_violation \
--add_data \
--use_restval \
--noise 0.1 \
--consistency_weight 10. \
--lr_update 15 \
--vocab ./vocab \
