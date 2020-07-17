#!/bin/bash

export LD_LIBRARY_PATH=/home/pzelasko/miniconda3/envs/swda/lib/

if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  source /home/pzelasko/miniconda3/bin/activate
  conda activate swda
  num_gpus=1
  CUDA_VISIBLE_DEVICES=$(free-gpu -n $num_gpus)
  export CUDA_VISIBLE_DEVICES
fi


set -eou pipefail

# My own setup for the CLSP grid - you might need to adjust it for your own purposes

if [ $# -ne 1 ]; then
  echo "Usage: run_da.sh <experiment-directory-absolute-path>"
  exit 1
fi

echo $(pwd)

EXP_DIR=$1
mkdir -p $EXP_DIR
cd $EXP_DIR

export OUTPUT_DIR=longformer
export BERT_MODEL=/export/c12/pzelasko/daseg/daseg/deps/longformer/longformer-base-4096
export BERT_MODEL_TOK=roberta-base
export MODEL_TYPE=longformer
export MAX_LENGTH=4096

dasg prepare-data \
  --continuations-allowed \
  --dataset-path /export/c12/pzelasko/daseg/daseg/deps/mrda \
  --tagset basic \
  ./

wget "https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py"
python3 preprocess.py train.txt.tmp $BERT_MODEL_TOK $MAX_LENGTH > train.txt
python3 preprocess.py dev.txt.tmp $BERT_MODEL_TOK $MAX_LENGTH > dev.txt
python3 preprocess.py test.txt.tmp $BERT_MODEL_TOK $MAX_LENGTH > test.txt
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt

export GRAD_ACCUM_STEPS=12
export BATCH_SIZE=1
export NUM_EPOCHS=150
export SAVE_STEPS=500
export SEED=1

python3 /export/c12/pzelasko/daseg/daseg/daseg/run_ner.py --data_dir ./ \
--crf_loss_weight 0.0 \
--tokenizer_name $BERT_MODEL_TOK \
--model_type $MODEL_TYPE \
--labels ./labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--gradient_accumulation_steps $GRAD_ACCUM_STEPS \
--evaluate_during_training \
--fp16 \
--do_eval \
--do_predict \
--do_train \
--use_longformer
