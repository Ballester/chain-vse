
python train.py \
--data_path /opt/datasets/vse/data/ \
--data_name f30k_tensor_precomp \
--adapt_data f30k_tensor_precomp \
--adapt_split train \
--adapt_batch_size 1 \
--val_data f30k_tensor_precomp \
--val_split val \
--val_batch_size 128 \
--img_dim 2048 \
--logger_name runs/f30k_tensor_precomp/b2 \
--hard_gamma 0.5 \
--text_encoder gru \
--add_data \
--use_restval \
--consistency_weight 0. \
--vocab ./vocab 


