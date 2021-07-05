#!/bin/bash

set -eou pipefail

if [[ "$OSTYPE" == "darwin"* ]]; then
  conda install pytorch torchvision -c pytorch=1.6
else
  conda install pytorch torchvision cudatoolkit=10.1 -c pytorch=1.6
fi
conda install numpy matplotlib pandas nltk tqdm jupyterlab pytest
