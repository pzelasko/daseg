#!/bin/bash

set -eou pipefail

for LABEL in segmentation basic general full; do
  for CASE in lower nolower; do
        qsub -l "gpu=1,hostname=c*" -q g.q -e $(pwd)/mrda_${LABEL}_longformer_${CASE}_stderr.txt -o $(pwd)/mrda_${LABEL}_longformer_${CASE}_stdout.txt run_longformer_${CASE}_mrda_${LABEL}.sh /export/c12/pzelasko/daseg/daseg/mrda_${LABEL}_longformer_${CASE}
  done
done
