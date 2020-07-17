#!/bin/bash

set -eou pipefail

for MODEL in longformer xlnet; do
    for LABEL in segmentation basic; do
      for CASE in lower nolower; do
        echo qsub -l "gpu=1,hostname=c*" -q g.q -e $(pwd)/swda_${LABEL}_${MODEL}_${CASE}_stderr.txt -o $(pwd)/swda_${LABEL}_${MODEL}_${CASE}_stdout.txt run_${MODEL}_${CASE}_swda_${LABEL}.sh /export/c12/pzelasko/daseg/daseg/swda_${LABEL}_${MODEL}_${CASE}
    done
  done
done
