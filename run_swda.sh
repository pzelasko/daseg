#!/bin/bash

set -eou pipefail

mkdir -p joint_coding 

for MODEL in longformer xlnet; do
    for LABEL in segmentation basic; do
      for CASE in lower nolower; do
        set -x
        qsub -l "gpu=1,hostname=c*" -q g.q -e $(pwd)/swda_${LABEL}_${MODEL}_${CASE}_stderr.txt -o $(pwd)/swda_${LABEL}_${MODEL}_${CASE}_stdout.txt run_${MODEL}_${CASE}_swda_${LABEL}.sh /export/c12/pzelasko/daseg/daseg/joint_coding/swda_${LABEL}_${MODEL}_${CASE}
        set +x
    done
  done
done
