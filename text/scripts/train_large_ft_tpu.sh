# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
train_tpu=kevin
eval_tpu=kevin
bert_model_dir=$GS/uda/text/pretrained_models/imdb_bert_ft
model_dir=$GS/uda/text/ckpt/large_ft_exp_1

python3 main.py \
  --use_tpu=True \
  --tpu_name=${train_tpu} \
  --do_train=True \
  --do_eval=False \
  --sup_train_data_dir=$GS/uda/text/data/proc_data/IMDB/train_20 \
  --eval_data_dir=$GS/uda/text/data/proc_data/IMDB/dev \
  --bert_config_file=${bert_model_dir}/bert_config.json \
  --vocab_file=${bert_model_dir}/vocab.txt \
  --init_checkpoint=${bert_model_dir}/bert_model.ckpt \
  --task_name=IMDB \
  --model_dir=${model_dir} \
  --max_seq_length=${MAX_SEQ_LENGTH} \
  --num_train_steps=3000 \
  --learning_rate=3e-05 \
  --train_batch_size=32 \
  --num_warmup_steps=300

python3 main.py \
  --use_tpu=True \
  --tpu_name=${eval_tpu} \
  --do_train=False \
  --do_eval=True \
  --sup_train_data_dir=$GS/uda/text/data/proc_data/IMDB/train_20 \
  --eval_data_dir=$GS/uda/text/data/proc_data/IMDB/dev \
  --bert_config_file=${bert_model_dir}/bert_config.json \
  --vocab_file=${bert_model_dir}/vocab.txt \
  --task_name=IMDB \
  --model_dir=${model_dir} \
  --max_seq_length=${MAX_SEQ_LENGTH} \
  --eval_batch_size=8 \
