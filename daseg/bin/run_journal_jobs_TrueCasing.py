import argparse
import shutil
import subprocess
from pathlib import Path
from time import sleep
import glob
import os, sys


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
    #import pdb; pdb.set_trace()
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
        sleep(2 if num_gpus else 2)


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
parser.add_argument('--loss-wts', default='1,1', type=str)
parser.add_argument('--hf-model-name', default=None, type=str)
parser.add_argument('--pretrained-full-model-path', default=None, type=str, help='if you want to finetune crossdomain')
parser.add_argument('--gpu-ind', default=-1, type=int, help='gpu number, useful if you are running in stndalone systems as opposed to grid')


args = parser.parse_args()


SCRIPT_TEMPLATE = """#!/usr/bin/env bash
source /home/rpapagari/.bashrc
source activate topic_seg
export CUDA_VISIBLE_DEVICES=$(free-gpu -n {num_gpus})
cd {work_dir}
{cmd}
"""

QSUB_TEMPLATE = "qsub -l \"hostname=!c01*&!c24*&c*,gpu={num_gpus},mem_free={mem}G,ram_free={mem}G\" -q {queue} -e {logerr} -o {logout} -N {name} {script}"

WORK_DIR = '/home/rpappagari/daseg/daseg'
EXP_DIR = str(Path(WORK_DIR) / args.exp_dir)

if args.model_name == 'longformer':
    args.model_name = 'allenai/longformer-base-4096'

if not args.dry_run:
    Path(EXP_DIR).mkdir(parents=True, exist_ok=True)

if 'T' in args.train_mode:
    already_trained = glob.glob(EXP_DIR + '/checkpointepoch*.ckpt')
    if len(already_trained) > 0:
        print(f'{EXP_DIR}')
        raise ValueError(f'{EXP_DIR} already was trained so we can not train again')

if args.train_mode == 'E':
    already_trained = glob.glob(EXP_DIR + '/checkpointepoch*.ckpt')
    if  len(already_trained) != 1:
        print(f'{EXP_DIR}')
        raise ValueError(f'{EXP_DIR} does not have checkpoints or multiple checkpoints')

if not os.path.isdir(args.data_dir):
    raise ValueError(f'{args.data_dir} does not exist')
else:
    utt2csvpath_path = args.data_dir+'/utt2csvpath'
    utt2csvpath_train_path = args.data_dir+'/utt2csvpath_train'
    utt2csvpath_dev_path = args.data_dir+'/utt2csvpath_dev'
    utt2csvpath_test_path = args.data_dir+'/utt2csvpath_test'

    if (not os.path.exists(utt2csvpath_path)) and os.path.exists(utt2csvpath_train_path) and os.path.exists(utt2csvpath_dev_path) and os.path.exists(utt2csvpath_test_path):
        print(f'{utt2csvpath_path} does not exist but exists for each split. Hence, concatenating them to create {utt2csvpath_path} ')
        cmd = 'cat ' + utt2csvpath_train_path + ' ' + utt2csvpath_dev_path + ' ' + utt2csvpath_test_path + ' > ' + utt2csvpath_path
        subprocess.call(cmd, shell=True)    


corpus = args.corpus #'IEMOCAP'
seed = args.seed

cmd = (f"dasg_TrueCasing train-transformer --model-name-or-path {args.model_name} -b {args.batch_size} -c 1 -e {args.no_epochs} "
    f"-r {args.seed} -g {args.num_gpus} --gradient-accumulation-steps {args.gacc} "
    f"--max-sequence-length {args.max_sequence_length} --pre-trained-model {args.pre_trained_model} "
    f"--frame-len {args.frame_len} --data-dir {args.data_dir} --train-mode {args.train_mode} "
    f"--label-scheme {args.label_scheme} --segmentation-type {args.segmentation_type} "
    f"--results-suffix {args.results_suffix} --concat-aug {args.concat_aug} "
    f"--emospotloss-wt {args.emospotloss_wt} --emospot-concat {args.emospot_concat} "
    f"--label-smoothing-alpha {args.label_smoothing_alpha} --test-file {args.test_file} "
    f"--full-speech {args.full_speech} --monitor-metric {args.monitor_metric} "
    f"--monitor-metric-mode {args.monitor_metric_mode} --pretrained-model-path {args.pretrained_model_path} "
    f"--loss-wts {args.loss_wts} --hf-model-name {args.hf_model_name} --pretrained-full-model-path {args.pretrained_full_model_path}  {outdir()} ")



if (args.gpu_ind != -1) and (not args.use_grid) and (args.num_gpus == 1):
    cmd = 'CUDA_VISIBLE_DEVICES=' + str(args.gpu_ind) + ' ' + cmd

submit(cmd, name='train', work_dir=WORK_DIR, num_gpus=args.num_gpus)


#submit(f"dasg_TrueCasing train-transformer --model-name-or-path {args.model_name} -b {args.batch_size} -c 1 -e {args.no_epochs} "
#       f"-r {args.seed} -g {args.num_gpus} --gradient-accumulation-steps {args.gacc} "
#       f"--max-sequence-length {args.max_sequence_length} --pre-trained-model {args.pre_trained_model} "
#       f"--frame-len {args.frame_len} --data-dir {args.data_dir} --train-mode {args.train_mode} "
#       f"--label-scheme {args.label_scheme} --segmentation-type {args.segmentation_type} "
#       f"--results-suffix {args.results_suffix} --concat-aug {args.concat_aug} "
#       f"--emospotloss-wt {args.emospotloss_wt} --emospot-concat {args.emospot_concat} "
#       f"--label-smoothing-alpha {args.label_smoothing_alpha} --test-file {args.test_file} "
#       f"--full-speech {args.full_speech} --monitor-metric {args.monitor_metric} "
#       f"--monitor-metric-mode {args.monitor_metric_mode} --pretrained-model-path {args.pretrained_model_path} "
#       f"--loss-wts {args.loss_wts} --hf-model-name {args.hf_model_name} --pretrained-full-model-path {args.pretrained_full_model_path}  {outdir()} ", 
#       name='train', work_dir=WORK_DIR, num_gpus=args.num_gpus)



