data=yelp-2
bert=bert_large
ckpt=large_0
step=10000
lr=1e-5
warmup=1000

python3 main.py \
  --use_tpu=False \
  --tpu_name=kevin \
  --do_train=True \
  --do_eval=True \
  --sup_train_data_dir=$GS/uda/text/data/proc_data/${data}/train \
  --eval_data_dir=$GS/uda/text/data/proc_data/${data}/dev \
  --bert_config_file=$GS/uda/text/pretrained_models/${bert}/bert_config.json \
  --vocab_file=$GS/uda/text/pretrained_models/${bert}/vocab.txt \
  --init_checkpoint=$GS/uda/text/pretrained_models/${bert}/bert_model.ckpt \
  --task_name=${data} \
  --model_dir=$GS/uda/text/ckpt/${data}/${ckpt} \
  --num_train_steps=${step} \
  --learning_rate=${lr} \
  --num_warmup_steps=${warmup} \
  --max_seq_length=${MAX_SEQ_LENGTH} \
  --train_batch_size=4 \
  --eval_batch_size=8
