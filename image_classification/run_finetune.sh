cd ./image_classification/

# # finetuning on CIFAR10/100
pretrained_weight=path_to_pretrained_weight

# # cifar10
for ((i=0;i<=4;i++))
do
{
data_dir=path/to/cifar10
save_dir=path/to/save/ckpt/seed=$i
mkdir $save_dir
CUDA_VISIBLE_DEVICES=0 python transfer_learning.py \
    --dataset 'cifar10' \
    --data-dir $data_dir \
    --lr 0.05 \
    --b 128 \
    --num-classes 10 \
    --img-size 224 \
    --model T2t_vit_14_all_un1d \
    --save-dir $save_dir \
    --transfer-model $pretrained_weight \
    --transfer-learning True \
    --seed $i
}

done

# # cifar100
for ((i=0;i<=4;i++))
do
{
data_dir=path/to/cifar100
save_dir=path/to/save/ckpt/seed=$i
mkdir $save_dir
CUDA_VISIBLE_DEVICES=0 python transfer_learning.py \
    --dataset 'cifar100' \
    --data-dir $data_dir \
    --lr 0.05 \
    --b 128 \
    --num-classes 100 \
    --img-size 224 \
    --model T2t_vit_14_all_un1d \
    --save-dir $save_dir \
    --transfer-model $pretrained_weight \
    --transfer-learning True \
    --seed $i
}

done
