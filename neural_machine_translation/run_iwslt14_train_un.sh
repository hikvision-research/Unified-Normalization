# # training transformer on iwslt14 w/ UN1d
save_root=path/to/save/ckpt

for ((i=0;i<=4;i++))
do
{
save_dir=$save_root/un1d_seed=$i
mkdir $save_dir
CUDA_VISIBLE_DEVICES=0 python train.py \
    data/iwslt14de2en/data-bin/iwslt14.tokenized.de-en/ \
    --arch transformer_iwslt_de_en \
    --norm-type "UN1d" \
    --un-win 4 \
    --un-warmup 4000 \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --share-all-embeddings \
    --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 0.0 \
    --lr 7e-4 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-7 --warmup-updates 6000 --max-update 100000 \
    --dropout 0.3 --attention-dropout 0.1 --relu-dropout 0.1 \
    --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --update-freq 1 \
    --keep-last-epochs 10 \
    --log-interval 500 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir $save_dir \
    --seed $i \
    | tee -a $save_dir/log.txt \

}

done