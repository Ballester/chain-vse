
# 1
# python train.py \
# --data_path /home/teia/datasets/chain/ \
# --data_name flickr30k_precomp \
# --adapt_data flickr30k_precomp \
# --adapt_split val \
# --adapt_batch_size 1 \
# --val_data flickr30k_precomp \
# --val_split val \
# --val_batch_size 128 \
# --img_dim 2048 \
# --logger_name runs/flickr30k_precomp/baseline_ema/ \
# --text_encoder gru \
# --max_violation \
# --add_data \
# --use_restval \
# --noise 0.0 \
# --consistency_weight 0 \
# --lr_update 15 \
# --vocab ./vocab \