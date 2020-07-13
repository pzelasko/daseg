#!/bin/bash

set -eou pipefail

CWD="$(pwd)"

mkdir -p deps

# Obtain the swda repo and make it a detectable python package
if [ ! -d deps/swda ]; then
  git clone https://github.com/cgpotts/swda deps/swda
  cat <<EOM >deps/swda/setup.py
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
fi

# Obtain the mrda repo and make it a detectable python package
if [ ! -d deps/mrda ]; then
  git clone https://github.com/NathanDuran/MRDA-Corpus deps/mrda
  cd deps/mrda
  mkdir mrda
  mv *.py mrda/
  touch __init__.py
  touch mrda/__init__.py
  cat <<EOM >setup.py
from setuptools import setup, find_packages
setup(
    name='mrda',
    version='1.0',
    packages=find_packages(),
)
EOM
  pip install -e .
  cd "$CWD"
fi

# Grab a spacy model
python -m spacy download en_core_web_sm
