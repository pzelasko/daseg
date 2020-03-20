#!/bin/bash

set -eou pipefail

CWD="$(pwd)"

mkdir -p deps

# Obtain the swda repo and make it a detectable python package
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