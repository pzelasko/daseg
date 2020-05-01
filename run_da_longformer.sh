#!/bin/bash

ROOT_DIR=$(pwd)

# My own setup for the CLSP grid - you might need to adjust it for your own purposes
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  cd /home/pzelasko/daseg/deps/transformers/examples/ner
  source /home/pzelasko/miniconda3/bin/activate
  conda activate swda
  num_gpus=1
  CUDA_VISIBLE_DEVICES=$(free-gpu -n $num_gpus)
  export CUDA_VISIBLE_DEVICES
else
  cd deps/transformers/examples/ner
fi
export OUTPUT_DIR=longformer-t46-crf
export BERT_MODEL=/home/ubuntu/daseg/deps/longformer/longformer-base-4096
export BERT_MODEL_TOK=roberta-base
export MODEL_TYPE=longformer
export MAX_LENGTH=4096

set -eou pipefail

#wget "https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py"
#python3 preprocess.py train.txt.tmp $BERT_MODEL_TOK $MAX_LENGTH > train.txt
#python3 preprocess.py dev.txt.tmp $BERT_MODEL_TOK $MAX_LENGTH > dev.txt
#python3 preprocess.py test.txt.tmp $BERT_MODEL_TOK $MAX_LENGTH > test.txt
#cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt

export BATCH_SIZE=1
export NUM_EPOCHS=10
export SAVE_STEPS=128  # approx once per epoch
export SEED=1

python3 "${ROOT_DIR}"/daseg/run_ner.py \
--data_dir ./ \
--tokenizer_name $BERT_MODEL_TOK \
--model_type $MODEL_TYPE \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size 8 \
--save_steps $SAVE_STEPS \
--logging_steps $SAVE_STEPS \
--seed $SEED \
--evaluate_during_training \
--do_train \
--do_eval \
--do_predict \
--use_longformer \
--use_crf \
--crf_loss_weight 0.01
#--fp16


