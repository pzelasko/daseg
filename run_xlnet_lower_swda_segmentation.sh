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

export OUTPUT_DIR=xlnet
export BERT_MODEL=xlnet-base-cased
export MODEL_TYPE=xlnet
export MAX_LENGTH=512

dasg prepare-data \
  --continuations-allowed \
  --strip-punctuation-and-lowercase \
  --dataset-path /export/c12/pzelasko/daseg/daseg/deps/swda/swda \
  --window-size 40 \
  --window-overlap 4 \
  --tagset segmentation \
  ./

wget "https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py"
python3 preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH >train.txt
python3 preprocess.py dev.txt.tmp $BERT_MODEL $MAX_LENGTH >dev.txt
python3 preprocess.py test.txt.tmp $BERT_MODEL $MAX_LENGTH >test.txt
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$" | sort | uniq >labels.txt

export GRAD_ACCUM_STEPS=2
export BATCH_SIZE=6
export NUM_EPOCHS=30
export SAVE_STEPS=500
export SEED=1

python3 /export/c12/pzelasko/daseg/daseg/daseg/bin/run_ner.py --data_dir ./ \
  --crf_loss_weight 0.0 \
  --model_type $MODEL_TYPE \
  --labels ./labels.txt \
  --model_name_or_path $BERT_MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --save_steps $SAVE_STEPS \
  --seed $SEED \
  --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
  --evaluate_during_training \
  --fp16 \
  --do_eval \
  --do_predict \
  --do_train
