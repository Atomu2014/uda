bert=imdb_bert_ft
data=bt-1
ckpt=bt_1_0
steps=30000
warmup=3000
lr=1e-5
train_bsz=16
eval_bsz=8

python3 main.py \
  --use_tpu=False \
  --tpu_name=kevin \
  --do_train=True \
  --do_eval=True \
  --sup_train_data_dir=$GS/uda/text/data/proc_data/IMDB/pseudo/${data}/0 \
  --eval_data_dir=$GS/uda/text/data/proc_data/IMDB/dev \
  --bert_config_file=$GS/uda/text/pretrained_models/${bert}/bert_config.json \
  --vocab_file=$GS/uda/text/pretrained_models/${bert}/vocab.txt \
  --init_checkpoint=$GS/uda/text/pretrained_models/${bert}/bert_model.ckpt \
  --task_name=IMDB \
  --model_dir=$GS/uda/text/ckpt/${ckpt} \
  --num_train_steps=${steps} \
  --learning_rate=${lr} \
  --num_warmup_steps=${warmup} \
  --max_seq_length=${MAX_SEQ_LENGTH} \
  --train_batch_size=${train_bsz} \
  --eval_batch_size=${eval_bsz}

