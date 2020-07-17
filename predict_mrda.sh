#!/bin/bash

set -eou pipefail

mkdir -p mrda_results

for LABEL in segmentation basic general full; do
  for CASE in lower nolower; do
    for RECONSTRUCTION in default begin_based; do
      CASE_OPT=
      if [[ $CASE == lower ]]; then CASE_OPT="-p"; fi
      RECONSTRUCTION_OPT=
      if [[ $RECONSTRUCTION == begin_based ]]; then RECONSTRUCTION_OPT='--begin-determines-act'; fi

      RESULTS_NAME=mrda_results/${LABEL}_longformer_${CASE}_${RECONSTRUCTION}

      dasg evaluate \
        --dataset-path deps/mrda \
        --batch_size 1 \
        --window_len 4096 \
        --device cpu \
        --tagset ${LABEL} \
        -o ${RESULTS_NAME}.pkl \
        ${CASE_OPT} \
        ${RECONSTRUCTION_OPT} \
        /export/c12/pzelasko/daseg/daseg/mrda_${LABEL}_longformer_${CASE}/longformer \
        &>${RESULTS_NAME}.txt &
    done
    wait
  done
done

