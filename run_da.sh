#!/bin/bash

# My own setup for the CLSP grid - you might need to adjust it for your own purposes

if [ $# -ne 1 ]; then
  echo "Usage: run_da.sh <experiment-directory>"
  exit 1
fi

EXP_DIR="$(readlink -f "$1")"
mkdir -p $EXP_DIR
cd $EXP_DIR

if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  source /home/pzelasko/miniconda3/bin/activate
  conda activate swda
  num_gpus=1
  CUDA_VISIBLE_DEVICES=$(free-gpu -n $num_gpus)
  export CUDA_VISIBLE_DEVICES
fi

set -eou pipefail

export OUTPUT_DIR=xlnet-t46
export BERT_MODEL=xlnet-base-cased
export MODEL_TYPE=xlnet
export MAX_LENGTH=512

dasg prepare-data \
  --strip-punctuation-and-lowercase \
  --continuations-allowed \
  --swda-path ../deps/swda/swda \
  ./

wget "https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py"
python3 preprocess.py train.txt.tmp $BERT_MODEL_TOK $MAX_LENGTH > train.txt
python3 preprocess.py dev.txt.tmp $BERT_MODEL_TOK $MAX_LENGTH > dev.txt
python3 preprocess.py test.txt.tmp $BERT_MODEL_TOK $MAX_LENGTH > test.txt
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt

export BATCH_SIZE=8
export NUM_EPOCHS=10
export SAVE_STEPS=500
export SEED=1

python3 run_ner.py --data_dir ./ \
--model_type $MODEL_TYPE \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--evaluate_during_training \
--overwrite_cache \
--do_eval \
--do_predict \
--do_train
