import argparse
import shutil
import subprocess
from pathlib import Path
from time import sleep

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run(cmd: str):
    if args.dry_run:
        print(cmd)
    else:
        print(cmd)
        subprocess.run(cmd, shell=True, text=True)


def submit(cmd: str, name: str, work_dir: str = None, num_gpus: int = 1):
    print(cmd)
    if args.use_grid:
        script_path = f'{outdir()}/run_task_{name}.sh'
        script = SCRIPT_TEMPLATE.format(
            num_gpus=num_gpus,
            work_dir=work_dir,
            cmd=cmd
        )
        qsub = QSUB_TEMPLATE.format(
            num_gpus=num_gpus,
            mem=4,
            script=script_path,
            logerr=f'{outdir()}/stderr_{name}.txt',
            logout=f'{outdir()}/stdout_{name}.txt',
            queue='g.q' if num_gpus else 'all.q',
            name=(cmd.split()[0] + '-' + Path(cmd.split()[-1]).stem).replace(' ', '-')
        )
        if args.dry_run:
            print(qsub)
            print(script, end='\n\n')
        else:
            with open(script_path, 'w') as f:
                print(script, file=f)
            with open(f'{outdir()}/qsub_{name}.sh', 'w') as f:
                print(f'#!/usr/bin/env bash\n{qsub}', file=f)
            run(qsub)
    else:
        run(cmd)
    if args.pause:
        input()
    elif not args.dry_run:
        sleep(15 if num_gpus else 2)


def outdir(use_seed=True, mkdir=True):
    def inner():
        if use_seed:
            return f'{EXP_DIR}'
            #return f'{EXP_DIR}/{model}_{corpus}_{seed}'
    path = inner()
    if mkdir and not args.dry_run:
        Path(path).mkdir(exist_ok=True, parents=True)
    return path


parser = argparse.ArgumentParser()
parser.add_argument('--general-exps', type=str2bool, default=True)
parser.add_argument('--use-grid', type=str2bool, default=True)
parser.add_argument('--pause', type=str2bool, default=False)
parser.add_argument('--train', type=str2bool, default=True)
parser.add_argument('--evaluate', type=str2bool, default=False)
parser.add_argument('--dry-run', type=str2bool, default=False)
parser.add_argument('--exp-dir', default='journal')
parser.add_argument('--data-dir', type=str)
parser.add_argument('--train-mode', type=str)
parser.add_argument('--max-sequence-length', type=int, default=4096)
parser.add_argument('--frame-len', type=float, default=0.1)
parser.add_argument('--label-scheme', default='Exact', type=str)
parser.add_argument('--segmentation-type', default='smooth',  type=str)
parser.add_argument('--num-gpus', default=0, type=int)
parser.add_argument('--batch-size', default=6, type=int)
parser.add_argument('--gacc', default=1, type=int)
parser.add_argument('--results-suffix', default='.pkl', type=str)
parser.add_argument('--concat-aug', default=-1, type=int)
parser.add_argument('--corpus', default='IEMOCAP', type=str)
parser.add_argument('--emospotloss-wt', default=1.0, type=float)
parser.add_argument('--no-epochs', default=50, type=int)
parser.add_argument('--emospot-concat', default=False, type=lambda x:x.lower()=='true')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--label-smoothing-alpha', default=0, type=float)
parser.add_argument('--model-name', default='longformer', type=str)
parser.add_argument('--pre-trained-model', default=False, type=str2bool)
parser.add_argument('--test-file', default='test.tsv', type=str)
parser.add_argument('--full-speech', default=False, type=lambda x:x.lower()=='true')
parser.add_argument('--monitor-metric', default='macro_f1', type=str)
parser.add_argument('--monitor-metric-mode', default='max', type=str, help='max, min')
parser.add_argument('--pretrained-model-path', default='/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5', type=str)


args = parser.parse_args()

SCRIPT_TEMPLATE = """#!/usr/bin/env bash
source /home/rpapagari/.bashrc
source activate daseg_v2
export CUDA_VISIBLE_DEVICES=$(free-gpu -n {num_gpus})
cd {work_dir}
{cmd}
"""

QSUB_TEMPLATE = "qsub -l \"hostname=!c01*&!c24*&c*,gpu={num_gpus},mem_free={mem}G,ram_free={mem}G\" -q {queue} -e {logerr} -o {logout} -N {name} {script}"

WORK_DIR = '/export/c02/rpapagari/daseg_erc/daseg'
EXP_DIR = str(Path(WORK_DIR) / args.exp_dir)
#SEEDS = [42] #(42, 43, 44)

opts = {
    'longformer': '--model-name-or-path allenai/longformer-base-4096',
    'bilstm': '--model-name-or-path bilstm',
    'xformer': '--model-name-or-path xformer',
    'ResNet': '--model-name-or-path ResNet',
    'xformersegpool': '--model-name-or-path xformersegpool',
    'xformer_cnnop_segpool': '--model-name-or-path xformer_cnnop_segpool',

    'longformer_text_SeqClassification': '--model-name-or-path longformer_text_SeqClassification',
    'longformer_speech_SeqClassification': '--model-name-or-path longformer_speech_SeqClassification',
    'resnet_SeqClassification': '--model-name-or-path resnet_SeqClassification',
    'bert-base-uncased': '--model-name-or-path bert-base-uncased',
    'bert-base-cased': '--model-name-or-path bert-base-cased',
    'TransformerMultiModalSeqClassification': '--model-name-or-path TransformerMultiModalSeqClassification',
    'TransformerMultiModalMultiLossSeqClassification': '--model-name-or-path TransformerMultiModalMultiLossSeqClassification',
    'bilstm_SeqClassification': '--model-name-or-path bilstm_SeqClassification',

    'truecasing_longformer_tokenclassif': '--model-name-or-path truecasing_longformer_tokenclassif',
}


if not args.dry_run:
    Path(EXP_DIR).mkdir(parents=True, exist_ok=True)

corpus = args.corpus #'IEMOCAP'
seed = args.seed

submit(f"dasg train-transformer {opts[args.model_name]} -b {args.batch_size} -c 1 -e {args.no_epochs} "
       f"-r {args.seed} -g {args.num_gpus} --gradient-accumulation-steps {args.gacc} "
       f"--max-sequence-length {args.max_sequence_length} --pre-trained-model {args.pre_trained_model} "
       f"--frame-len {args.frame_len} --data-dir {args.data_dir} --train-mode {args.train_mode} "
       f"--label-scheme {args.label_scheme} --segmentation-type {args.segmentation_type} "
       f"--results-suffix {args.results_suffix} --concat-aug {args.concat_aug} "
       f"--emospotloss-wt {args.emospotloss_wt} --emospot-concat {args.emospot_concat} "
       f"--label-smoothing-alpha {args.label_smoothing_alpha} --test-file {args.test_file} "
       f"--full-speech {args.full_speech} --monitor-metric {args.monitor_metric} "
       f"--monitor-metric-mode {args.monitor_metric_mode} --pretrained-model-path {args.pretrained_model_path}  "
       f" {outdir()} ", 
       name='train', work_dir=WORK_DIR, num_gpus=args.num_gpus)



