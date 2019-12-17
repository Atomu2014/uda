#!/bin/bash
bert_vocab_file=pretrained_models/bert_base/vocab.txt


#python3 preprocess.py \
#  --raw_data_dir=data/IMDB_raw/csv/train.csv \
#  --output_base_dir=../back_translate/imdb.txt \
#  --data_type=back_trans \
#  --vocab_file=$bert_vocab_file


#python3 preprocess.py \
#  --raw_data_dir=data/IMDB_raw/csv \
#  --output_base_dir=data/proc_data/IMDB/pseudo \
#  --back_translation_dir=data/back_translation/imdb_back_trans \
#  --data_type=pseudo \
#  --sub_set=unsup_in \
#  --aug_ops=bt-1 \
#  --aug_copy_num=0 \
#  --vocab_file=$bert_vocab_file \
#  --max_seq_length=${MAX_SEQ_LENGTH}


## Preprocess supervised training set
#python3 preprocess.py \
#  --raw_data_dir=data/IMDB_raw/csv \
#  --output_base_dir=data/proc_data/IMDB/train_20 \
#  --data_type=sup \
#  --sub_set=train \
#  --sup_size=20 \
#  --vocab_file=$bert_vocab_file \
#  --max_seq_length=${MAX_SEQ_LENGTH}

python3 preprocess.py \
  --task_name=yelp-2 \
  --raw_data_dir=data/yelp-2 \
  --output_base_dir=data/proc_data/yelp-2/train \
  --data_type=sup \
  --sub_set=train \
  --sup_size=-1 \
  --vocab_file=$bert_vocab_file \
  --max_seq_length=${MAX_SEQ_LENGTH}

# Preprocess test set
python3 preprocess.py \
  --task_name=yelp-2 \
  --raw_data_dir=data/yelp-2 \
  --output_base_dir=data/proc_data/yelp-2/dev \
  --data_type=sup \
  --sub_set=dev \
  --vocab_file=$bert_vocab_file \
  --max_seq_length=${MAX_SEQ_LENGTH}


## Preprocess unlabeled set
#python3 preprocess.py \
#  --raw_data_dir=data/IMDB_raw/csv \
#  --output_base_dir=data/proc_data/IMDB/unsup \
#  --back_translation_dir=data/back_translation/imdb_back_trans \
#  --data_type=unsup \
#  --sub_set=unsup_in \
#  --aug_ops=bt-0.9 \
#  --aug_copy_num=0 \
#  --vocab_file=$bert_vocab_file \
#  --max_seq_length=${MAX_SEQ_LENGTH}
