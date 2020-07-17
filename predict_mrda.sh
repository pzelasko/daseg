#!/bin/bash

set -eou pipefail

mkdir -p mrda_results

for LABEL in segmentation basic general full; do
  for CASE in lower nolower; do
    CASE_OPT=
    if [[ $CASE == lower ]]; then CASE_OPT="-p"; fi
    dasg evaluate --dataset-path deps/mrda --batch_size 1 --window_len 4096 --device cpu --tagset ${LABEL} ${CASE_OPT} /export/c12/pzelasko/daseg/daseg/mrda_${LABEL}_longformer_${CASE}/longformer &> mrda_results/${LABEL}_longformer_${CASE}.txt &
  done
done

wait
