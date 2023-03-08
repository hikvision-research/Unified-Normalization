save_dir=path/to/saved/ckpt

# evaluate iwslt14de2en
num=10
python scripts/average_checkpoints.py \
--inputs $save_dir \
--num-epoch-checkpoints $num \
--output $save_dir/checkpoint_avg_$num.pt

python -W ignore fairseq_cli/generate.py \
data/iwslt14de2en/data-bin/iwslt14.tokenized.de-en/ \
--path $save_dir/checkpoint_avg_$num.pt \
--batch-size 256 \
--beam 5 \
--remove-bpe \
--quiet