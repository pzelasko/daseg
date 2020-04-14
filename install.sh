#!/bin/bash

set -eou pipefail

CWD="$(pwd)"

mkdir -p deps

## Obtain the swda repo and make it a detectable python package
git clone https://github.com/cgpotts/swda deps/swda
cat << EOM > deps/swda/setup.py
from setuptools import setup, find_packages
setup(
    name='swda',
    version='1.0',
    packages=find_packages(),
)
EOM
cd deps/swda
pip install -e .
unzip swda.zip
cd "$CWD"

# Grab a spacy model
python -m spacy download en_core_web_sm

# Get Transformers
git clone https://github.com/huggingface/transformers deps/transformers
cd deps/transformers
pip install -e .
cd "$CWD"

# Get AllenNLP
git clone https://github.com/allenai/allennlp deps/allennlp
cd deps/allennlp
git checkout v1.0-prerelease
pip install -e .
cd "$CWD"

# Get Longformer
git clone https://github.com/allenai/longformer deps/longformer
cd deps/longformer
sed -i 's/==.*//g' requirements.txt
pip install -e .
wget https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-base-4096.tar.gz
tar xf longformer-base-4096.tar.gz
rm longformer-base-4096.tar.gz
cd "$CWD"
