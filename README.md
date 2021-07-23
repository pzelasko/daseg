# Daseg

A library for working with dialog acts.

## Usage

```python
from daseg import DialogActCorpus
swda = DialogActCorpus.from_path('deps/swda/swda')
for call_id, call in swda.dialogues.items():
    print(call_id, "number of segments:", len(call))

# Some useful methods/properties:
swda.calls
swda.call_ids
swda.turns
swda.dialog_acts
swda.joint_coding_dialog_act_labels
swda.train_dev_test_split()

mrda = DialogActCorpus.from_path('deps/mrda')
```

## Installation

The preferred way to use `daseg` is with an anaconda environment. We tested it with Python 3.8 - it might or might not work with an earlier version.

```
conda create -n daseg python=3.8
conda activate daseg
```

Installing requirements and downloading datasets:

```bash
git clone https://github.com/pzelasko/daseg
cd daseg

# Installs some dependencies using conda, for faster execution.
./setup_anaconda.sh

# Installs the rest of requirements from pip
pip install -r requirements.txt 

python3 -m spacy download en

# Downloads and "installs" SWDA and MRDA
./install.sh

# Installs daseg in your python env so that you can "import daseg"
# "-e" means that if you can modify the code in this directory,
# and the changes are visible next time you import without re-installation
pip install -e .
```
