#python tests.py test --dataset_name 'Cave' --n_blocks 8  --batch_size 16 --n_subs 4 --n_ovls 1 --n_feats 256 --n_scale 8 --gpus "0,1"
#python tests.py test --dataset_name 'Cave'   --batch_size 16  --n_feats 256 --n_scale 4 --gpus "1"

#python tests.py test --dataset_name 'Chikusei'   --batch_size 16  --n_feats 256 --n_scale 8 --gpus "1"

#python tests.py test --dataset_name 'Pavia'   --batch_size 16  --n_feats 256 --n_scale 8 --gpus "0,1" --n_subs 8 --n_ovls 2 --n_blocks 6
#python tests.py test --dataset_name 'Pavia'   --batch_size 16  --n_feats 256 --n_scale 8 --gpus "0,1"


python tests.py test --dataset_name 'Cave'   --batch_size 16  --n_feats 256 --n_scale 4 --gpus "0"