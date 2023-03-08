cd ./image_classification

# t2t-vit-14-un on 8 gpus (V100)
save_dir=path/to/save/ckpt
data_dir=path/to/imagenet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash distributed_train.sh 8 \
    $data_dir \
    --model T2t_vit_14_all_un1d \
    --batch-size 64 \
    --lr 5e-4 \
    --weight-decay 0.05 \
    --img-size 224 \
    --output $save_dir
