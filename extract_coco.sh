# python extract_features.py \
# --datapath /A/VSE/data/coco/images/train2014/ \
# --outpath /A/VSE/temp/coco/ \
# --batch_size 50 \
# --num_workers 6 \
# --nb_crops 10 \
# --resize 256 \
# --crop_size 256 \

python extract_features.py \
--datapath /A/VSE/data/coco/images/val2014/ \
--outpath /A/VSE/temp/coco/ \
--batch_size 50 \
--num_workers 6 \
--nb_crops 10 \
--resize 256 \
--crop_size 256 \

python extract_features.py \
--datapath /A/VSE/data/coco/images/test2014/ \
--outpath /A/VSE/temp/coco/ \
--batch_size 50 \
--num_workers 6 \
--nb_crops 10 \
--resize 256 \
--crop_size 256 \

# python extract_features.py \
# --datapath /A/VSE/data/coco/images/unlabeled/ \
# --outpath /A/VSE/temp/coco/ \
# --batch_size 65 \
# --num_workers 6 \
# --nb_crops 10 \
# --resize 256 \
# --crop_size 256 \

