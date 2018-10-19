
python train.py \
--data_path /opt/datasets/chain/data/ \
--data_name f30ktensor_precomp \
--adapt_data f30ktensor_precomp \
--adapt_split train \
--adapt_batch_size 1 \
--val_data f30ktensor_precomp \
--val_split val \
--val_batch_size 128 \
--max_violation \
--img_dim 2048 \
--logger_name runs/f30k_tensor_precomp/baseline \
--text_encoder gru \
--add_data \
--use_restval \
--consistency_weight 0. \
--vocab ./vocab \


