#!/bin/bash

set -eou pipefail

for MODEL in longformer xlnet; do
  for LABEL in segmentation basic; do
    for CASE in lower nolower; do
      CASE_OPT=
      if [ $CASE -eq lower ]; then CASE_OPT="-p"; fi
      WINDOW=4096
      if [ $MODEL -eq xlnet ]; then WINDOW=512; fi
      dasg evaluate --dataset-path deps/swda/swda --batch_size 1 --window_len $WINDOW --device cpu --tagset ${LABEL} ${CASE_OPT} /export/c12/pzelasko/daseg/daseg/swda_${LABEL}_${MODEL}_${CASE}/${MODEL}
    done
  done
done
