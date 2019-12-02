bert_vocab_file=pretrained_models/bert_base/vocab.txt


python3 preprocess.py \
  --raw_data_dir=data/IMDB_raw/csv \
  --output_base_dir=data/proc_data/IMDB/pseudo \
  --back_translation_dir=data/back_translation/imdb_back_trans \
  --data_type=pseudo \
  --sub_set=unsup_in \
  --aug_ops=bt-0.9 \
  --aug_copy_num=0 \
  --vocab_file=$bert_vocab_file \
  --max_seq_length=${MAX_SEQ_LENGTH}
