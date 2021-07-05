# Daseg

A library for working with dialog acts.

## Usage

```python
from daseg import DialogActCorpus

# Reading SWDA
swda = DialogActCorpus.from_path('deps/swda/swda')
for call_id, call in swda.dialogues.items():
    print(call_id, "number of segments:", len(call))

# Reading MRDA
mrda = DialogActCorpus.from_path('deps/mrda')

# Some useful methods/properties:
swda.calls
swda.call_ids
swda.turns
swda.dialog_acts
swda.joint_coding_dialog_act_labels
swda.train_dev_test_split()

# Utilities for visualizing calls:
call = swda.calls[0]
call.render(max_turns=20)
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

# Downloads and "installs" SWDA and MRDA
./install.sh

# Installs daseg in your python env so that you can "import daseg"
# "-e" means that if you can modify the code in this directory,
# and the changes are visible next time you import without re-installation
pip install -e .
```

## Running the experiments

Experiments can be run using dasg CLI:

```bash
# Preproces the text data and dump training examples to disk.
$ dasg prepare-exp --dataset-path deps/swda/swda --windows-if-exceeds-max-len --max-sequence-length 512 --model-name-or-path 'allenai/longformer-base-4096' exp-swda-longformer-512
# Run actual training for 10 epochs, with batch size 10, 1 GPU and half-precision.
$ dasg train-transformer exp-swda-longformer-512 -e 10 -g 1 -b 10 -f
# Run model evaluation on the test split with the checkpoint from epoch 9, displaying progress bar.
$ dasg evaluate 'exp-swda-longformer-512/checkpointepoch=9.ckpt' --device cuda -b 1 -o exp-swda-longformer-512/results.pkl -v
```

### Replicating paper results

The specific commands to replicate the results in the paper can be generated using the script that submits everything to
compute on CLSP grid in `dry-run` mode. You might need to modify some paths to make it run in your computing
environment (if you are using SGE/qsub, perhaps with minor modifications the script can leverage your computing queue):

```bash
# Command to replicate the turns/full-context, lower/nolower, and BiGRU/XLNet/XLNet+prop/Longformer experiments:
$ python daseg/bin/run_journal_jobs.py --general-exps 1 --use-grid 0 --dry-run 1
# Command to replicate the full/general/basic/segmentation label set experiments:
$ python daseg/bin/run_journal_jobs.py --general-exps 0 --label-sets 1 --use-grid 0 --dry-run 1
```

For reference, these are the commands outputted by the script for "general exps":

```bash
dasg prepare-exp --model-name-or-path allenai/longformer-base-4096 --dataset-path deps/swda/swda -p -s basic -l 4096 -w /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_dialog_lower_basic
dasg prepare-exp --model-name-or-path allenai/longformer-base-4096 --dataset-path deps/swda/swda -p -s basic --turns -l 128 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_turn_lower_basic
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/swda/swda -p -s basic -l 512 -w /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_lower_basic
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/swda/swda -p -s basic --turns -l 128 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_turn_lower_basic
dasg train-bigru -g 1 -s basic -b 30 -e 10 -r 42 --dataset-path deps/swda/swda -p /export/c12/pzelasko/daseg/daseg/journal/bigru_swda_turn_lower_basic_42
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 8 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_turn_lower_basic_42
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 1 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_dialog_lower_basic_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 8 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_turn_lower_basic_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_lower_basic_42
dasg train-bigru -g 1 -s basic -b 30 -e 10 -r 43 --dataset-path deps/swda/swda -p /export/c12/pzelasko/daseg/daseg/journal/bigru_swda_turn_lower_basic_43
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 8 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_turn_lower_basic_43
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 1 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_dialog_lower_basic_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 8 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_turn_lower_basic_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_lower_basic_43
dasg train-bigru -g 1 -s basic -b 30 -e 10 -r 44 --dataset-path deps/swda/swda -p /export/c12/pzelasko/daseg/daseg/journal/bigru_swda_turn_lower_basic_44
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 8 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_turn_lower_basic_44
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 1 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_dialog_lower_basic_44
dasg train-transformer --model-name-or-path xlnet-base-cased -b 8 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_turn_lower_basic_44
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_lower_basic_44
dasg prepare-exp --model-name-or-path allenai/longformer-base-4096 --dataset-path deps/swda/swda  -s basic -l 4096 -w /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_dialog_nolower_basic
dasg prepare-exp --model-name-or-path allenai/longformer-base-4096 --dataset-path deps/swda/swda  -s basic --turns -l 128 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_turn_nolower_basic
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/swda/swda  -s basic -l 512 -w /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_nolower_basic
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/swda/swda  -s basic --turns -l 128 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_turn_nolower_basic
dasg train-bigru -g 1 -s basic -b 30 -e 10 -r 42 --dataset-path deps/swda/swda  /export/c12/pzelasko/daseg/daseg/journal/bigru_swda_turn_nolower_basic_42
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 8 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_turn_nolower_basic_42
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 1 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_dialog_nolower_basic_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 8 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_turn_nolower_basic_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_nolower_basic_42
dasg train-bigru -g 1 -s basic -b 30 -e 10 -r 43 --dataset-path deps/swda/swda  /export/c12/pzelasko/daseg/daseg/journal/bigru_swda_turn_nolower_basic_43
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 8 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_turn_nolower_basic_43
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 1 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_dialog_nolower_basic_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 8 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_turn_nolower_basic_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_nolower_basic_43
dasg train-bigru -g 1 -s basic -b 30 -e 10 -r 44 --dataset-path deps/swda/swda  /export/c12/pzelasko/daseg/daseg/journal/bigru_swda_turn_nolower_basic_44
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 8 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_turn_nolower_basic_44
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 1 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_swda_dialog_nolower_basic_44
dasg train-transformer --model-name-or-path xlnet-base-cased -b 8 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_turn_nolower_basic_44
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_nolower_basic_44
dasg prepare-exp --model-name-or-path allenai/longformer-base-4096 --dataset-path deps/mrda -p -s basic -l 4096 -w /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_dialog_lower_basic
dasg prepare-exp --model-name-or-path allenai/longformer-base-4096 --dataset-path deps/mrda -p -s basic --turns -l 128 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_turn_lower_basic
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/mrda -p -s basic -l 512 -w /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_basic
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/mrda -p -s basic --turns -l 128 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_turn_lower_basic
dasg train-bigru -g 1 -s basic -b 30 -e 10 -r 42 --dataset-path deps/mrda -p /export/c12/pzelasko/daseg/daseg/journal/bigru_mrda_turn_lower_basic_42
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 8 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_turn_lower_basic_42
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 1 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_dialog_lower_basic_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 8 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_turn_lower_basic_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_basic_42
dasg train-bigru -g 1 -s basic -b 30 -e 10 -r 43 --dataset-path deps/mrda -p /export/c12/pzelasko/daseg/daseg/journal/bigru_mrda_turn_lower_basic_43
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 8 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_turn_lower_basic_43
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 1 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_dialog_lower_basic_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 8 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_turn_lower_basic_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_basic_43
dasg train-bigru -g 1 -s basic -b 30 -e 10 -r 44 --dataset-path deps/mrda -p /export/c12/pzelasko/daseg/daseg/journal/bigru_mrda_turn_lower_basic_44
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 8 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_turn_lower_basic_44
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 1 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_dialog_lower_basic_44
dasg train-transformer --model-name-or-path xlnet-base-cased -b 8 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_turn_lower_basic_44
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_basic_44
dasg prepare-exp --model-name-or-path allenai/longformer-base-4096 --dataset-path deps/mrda  -s basic -l 4096 -w /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_dialog_nolower_basic
dasg prepare-exp --model-name-or-path allenai/longformer-base-4096 --dataset-path deps/mrda  -s basic --turns -l 128 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_turn_nolower_basic
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/mrda  -s basic -l 512 -w /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_basic
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/mrda  -s basic --turns -l 128 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_turn_nolower_basic
dasg train-bigru -g 1 -s basic -b 30 -e 10 -r 42 --dataset-path deps/mrda  /export/c12/pzelasko/daseg/daseg/journal/bigru_mrda_turn_nolower_basic_42
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 8 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_turn_nolower_basic_42
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 1 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_dialog_nolower_basic_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 8 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_turn_nolower_basic_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_basic_42
dasg train-bigru -g 1 -s basic -b 30 -e 10 -r 43 --dataset-path deps/mrda  /export/c12/pzelasko/daseg/daseg/journal/bigru_mrda_turn_nolower_basic_43
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 8 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_turn_nolower_basic_43
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 1 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_dialog_nolower_basic_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 8 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_turn_nolower_basic_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_basic_43
dasg train-bigru -g 1 -s basic -b 30 -e 10 -r 44 --dataset-path deps/mrda  /export/c12/pzelasko/daseg/daseg/journal/bigru_mrda_turn_nolower_basic_44
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 8 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_turn_nolower_basic_44
dasg train-transformer --model-name-or-path allenai/longformer-base-4096 -b 1 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/longformer_mrda_dialog_nolower_basic_44
dasg train-transformer --model-name-or-path xlnet-base-cased -b 8 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_turn_nolower_basic_44
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_basic_44
```

For reference, these are the commands outputted by the script for "label sets":

```bash
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/mrda -p -s segmentation -l 512 -w /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_segmentation
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_segmentation_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_segmentation_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_segmentation_44
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/mrda  -s segmentation -l 512 -w /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_segmentation
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_segmentation_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_segmentation_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_segmentation_44
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/mrda -p -s general -l 512 -w /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_general
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_general_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_general_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_general_44
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/mrda  -s general -l 512 -w /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_general
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_general_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_general_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_general_44
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/mrda -p -s full -l 512 -w /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_full
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_full_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_full_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_lower_full_44
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/mrda  -s full -l 512 -w /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_full
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_full_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_full_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_mrda_dialog_nolower_full_44
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/swda/swda -p -s segmentation -l 512 -w /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_lower_segmentation
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_lower_segmentation_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_lower_segmentation_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_lower_segmentation_44
dasg prepare-exp --model-name-or-path xlnet-base-cased --dataset-path deps/swda/swda  -s segmentation -l 512 -w /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_nolower_segmentation
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 42 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_nolower_segmentation_42
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 43 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_nolower_segmentation_43
dasg train-transformer --model-name-or-path xlnet-base-cased -b 6 -c 8 -e 10 -a 1 -r 44 -g 1 /export/c12/pzelasko/daseg/daseg/journal/xlnet_swda_dialog_nolower_segmentation_44
```