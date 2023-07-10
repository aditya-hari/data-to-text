#!/bin/bash
#SBATCH --mem-per-cpu 2G
#SBATCH -c 9
#SBATCH -w gnode045
#SBATCH -A irel
#SBATCH --gres gpu:1
#SBATCH --time 3-00:00:00
#SBATCH --output revised_mt.log
#SBATCH --mail-user aditya.hari@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name ga_nabu

python preprocess.py \
  --train_src '/home2/aditya_hari/gsoc/data/processed/mt/train_src' \
  --train_tgt '/home2/aditya_hari/gsoc/data/processed/mt/train_tgt' \
  --eval_src '/home2/aditya_hari/gsoc/data/processed/mt/eval_src' \
  --eval_tgt '/home2/aditya_hari/gsoc/data/processed/mt/eval_tgt' \
  --test_src '/home2/aditya_hari/gsoc/data/processed/mt/eval_src' \
  --spl_sym 'data/processed_data/special_symbols' \
  --model gat --lang mt --sentencepiece True \
  --vocab_size 16000 --sentencepiece_model 'bpe'


python train_single.py \
  --train_path 'data/processed_graphs/mt/gat/train' \
  --eval_path 'data/processed_graphs/mt/gat/eval' \
  --test_path 'data/processed_graphs/mt/gat/test' \
  --src_vocab 'vocabs/gat/mt/src_vocab' \
  --tgt_vocab 'vocabs/gat/mt/train_vocab.model' \
  --batch_size 32 --enc_type gat --dec_type transformer --model gat --vocab_size 16000 \
  --emb_dim 256 --hidden_size 256 --filter_size 16 --beam_size 5 --mode 'train' \
  --beam_alpha 0.1 --enc_layers 6 --dec_layers 6 --num_heads 8 --sentencepiece True \
  --steps 25000 --eval_steps 250 --checkpoint 3333 --alpha 0.2 --dropout 0.2 \
  --reg_scale 0.0 --decay True --decay_steps 12500 --lang mt --debug_mode False \
  --eval '/home2/aditya_hari/gsoc/data/processed/mt/eval_src' --eval_ref '/home2/aditya_hari/gsoc/data/processed/mt/eval_tgt'

