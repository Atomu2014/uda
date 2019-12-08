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
python3 main.py \
  --use_tpu=False \
  --tpu_name=kevin \
  --do_train=False \
  --do_eval=True \
  --sup_train_data_dir=$GS/uda/text/data/proc_data/IMDB/train \
  --eval_data_dir=$GS/uda/text/data/proc_data/IMDB/dev \
  --bert_config_file=$GS/uda/text/pretrained_models/imdb_bert_ft/bert_config.json \
  --vocab_file=$GS/uda/text/pretrained_models/imdb_bert_ft/vocab.txt \
  --init_checkpoint=$GS/uda/text/pretrained_models/imdb_bert_ft/bert_model.ckpt \
  --task_name=IMDB \
  --model_dir=$GS/uda/text/ckpt/base_13 \
  --num_train_steps=30000 \
  --learning_rate=3e-06 \
  --num_warmup_steps=3000 \
  --max_seq_length=${MAX_SEQ_LENGTH}

python3 main.py \
  --use_tpu=False \
  --tpu_name=kevin \
  --do_train=False \
  --do_eval=True \
  --sup_train_data_dir=$GS/uda/text/data/proc_data/IMDB/train \
  --eval_data_dir=$GS/uda/text/data/proc_data/IMDB/dev \
  --bert_config_file=$GS/uda/text/pretrained_models/imdb_bert_ft/bert_config.json \
  --vocab_file=$GS/uda/text/pretrained_models/imdb_bert_ft/vocab.txt \
  --init_checkpoint=$GS/uda/text/pretrained_models/imdb_bert_ft/bert_model.ckpt \
  --task_name=IMDB \
  --model_dir=$GS/uda/text/ckpt/base_14 \
  --num_train_steps=30000 \
  --learning_rate=1e-06 \
  --num_warmup_steps=3000 \
  --max_seq_length=${MAX_SEQ_LENGTH}

python3 main.py \
  --use_tpu=False \
  --tpu_name=kevin \
  --do_train=False \
  --do_eval=True \
  --sup_train_data_dir=$GS/uda/text/data/proc_data/IMDB/pseudo/bt-0.9/0 \
  --eval_data_dir=$GS/uda/text/data/proc_data/IMDB/dev \
  --bert_config_file=$GS/uda/text/pretrained_models/imdb_bert_ft/bert_config.json \
  --vocab_file=$GS/uda/text/pretrained_models/imdb_bert_ft/vocab.txt \
  --init_checkpoint=$GS/uda/text/pretrained_models/imdb_bert_ft/bert_model.ckpt \
  --task_name=IMDB \
  --model_dir=$GS/uda/text/ckpt/pseudo_15 \
  --num_train_steps=30000 \
  --learning_rate=3e-06 \
  --num_warmup_steps=3000 \
  --max_seq_length=${MAX_SEQ_LENGTH}

python3 main.py \
  --use_tpu=False \
  --tpu_name=kevin \
  --do_train=False \
  --do_eval=True \
  --sup_train_data_dir=$GS/uda/text/data/proc_data/IMDB/pseudo/bt-0.9/0 \
  --eval_data_dir=$GS/uda/text/data/proc_data/IMDB/dev \
  --bert_config_file=$GS/uda/text/pretrained_models/imdb_bert_ft/bert_config.json \
  --vocab_file=$GS/uda/text/pretrained_models/imdb_bert_ft/vocab.txt \
  --init_checkpoint=$GS/uda/text/pretrained_models/imdb_bert_ft/bert_model.ckpt \
  --task_name=IMDB \
  --model_dir=$GS/uda/text/ckpt/pseudo_16 \
  --num_train_steps=30000 \
  --learning_rate=1e-06 \
  --num_warmup_steps=3000 \
  --max_seq_length=${MAX_SEQ_LENGTH}
