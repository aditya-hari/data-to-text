#!/bin/bash
#SBATCH -c 19
#SBATCH --mem-per-cpu 2G
#SBATCH -A irel 
#SBATCH -w gnode048
#SBATCH --gres gpu:2
#SBATCH --time 3-00:00:00
#SBATCH --output br_single.log
#SBATCH --mail-user aditya.hari@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name br_nabu

python train_single.py \
  --train_path 'data/processed_graphs/br/gat/train' \
  --eval_path 'data/processed_graphs/br/gat/eval' \
  --test_path 'data/processed_graphs/br/gat/test' \
  --src_vocab 'vocabs/gat/br/src_vocab' \
  --tgt_vocab 'vocabs/gat/br/train_vocab.model' \
  --batch_size 32 --enc_type gat --dec_type transformer --model gat --vocab_size 32000 \
  --emb_dim 16 --hidden_size 16 --filter_size 16 --beam_size 5 --mode 'train' \
  --beam_alpha 0.1 --enc_layers 1 --dec_layers 1 --num_heads 1 --sentencepiece True \
  --steps 50000 --eval_steps 2500 --checkpoint 5000 --alpha 0.2 --dropout 0.2 \
  --reg_scale 0.0 --decay True --decay_steps 5000 --lang br --debug_mode False \
  --eval '/home2/aditya_hari/gsoc/data/processed/br/dev_src.txt' --eval_ref '/home2/aditya_hari/gsoc/data/processed/br/dev_ref.txt'
