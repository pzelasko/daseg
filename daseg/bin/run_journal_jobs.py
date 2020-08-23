import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
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

parser = argparse.ArgumentParser()
parser.add_argument('--use-grid', type=str2bool, default=True)
parser.add_argument('--pause', type=str2bool, default=False)
parser.add_argument('--skip-data-prep', type=str2bool, default=False)
parser.add_argument('--bigru', type=str2bool, default=True)
parser.add_argument('--longformer', type=str2bool, default=True)
parser.add_argument('--xlnet', type=str2bool, default=True)
parser.add_argument('--train', type=str2bool, default=True)
parser.add_argument('--evaluate', type=str2bool, default=False)
parser.add_argument('--dry-run', type=str2bool, default=False)
args = parser.parse_args()

SCRIPT_TEMPLATE = """#!/usr/bin/env bash

conda activate swda
export CUDA_VISIBLE_DEVICES=$(free-gpu -n {num_gpus})
cd {work_dir}
{cmd}
"""

QSUB_TEMPLATE = "qsub -l \"hostname=c*,gpu={num_gpus}\" -q {queue} -e {logerr} -o {logout} -N {name} {script}"

WORK_DIR = '/export/c12/pzelasko/daseg/daseg'
EXP_DIR = str(Path(WORK_DIR) / 'journal')
SEEDS = (42, 43, 44)

opts = {
    'longformer': '--model-name-or-path allenai/longformer-base-4096',
    'xlnet': '--model-name-or-path xlnet-base-cased',
    'lower': '-p',
    'nolower': '',
    'swda': '--dataset-path deps/swda/swda',
    'mrda': '--dataset-path deps/mrda'
}

seqlen = {
    'longformer': 4096,
    'xlnet': 512
}

bsize = {
    'longformer': 1,
    'xlnet': 6
}

gacc = {
    'longformer': 6,
    'xlnet': 1
}


def run(cmd: str):
    if args.dry_run:
        print(cmd)
    else:
        subprocess.run(cmd, shell=True, text=True)


def submit(cmd: str, name: str, work_dir: str = WORK_DIR, num_gpus: int = 1):
    if args.use_grid:
        with NamedTemporaryFile('w+') as f:
            script = SCRIPT_TEMPLATE.format(
                num_gpus=num_gpus,
                work_dir=work_dir,
                cmd=cmd
            )
            f.write(script)
            f.flush()
            qsub = QSUB_TEMPLATE.format(
                num_gpus=num_gpus,
                script=f.name,
                logerr=f'{outdir()}/stderr_{name}.txt',
                logout=f'{outdir()}/stdout_{name}.txt',
                queue='g.q' if num_gpus else 'all.q',
                name=(cmd.split()[0] + '-' + Path(cmd.split()[-1]).stem).replace(' ', '-')
            )
            if args.dry_run:
                print(qsub)
                print(script, end='\n\n')
            else:
                run(qsub)
    else:
        run(cmd)
    if args.pause:
        input()
    elif not args.dry_run:
        sleep(15)


if not args.dry_run:
    Path(EXP_DIR).mkdir(parents=True, exist_ok=True)


def outdir(use_seed=True, mkdir=True):
    def inner():
        if use_seed:
            return f'{EXP_DIR}/{model}_{corpus}_{context}_{case}_{tagset}_{seed}'
        return f'{EXP_DIR}/{model}_{corpus}_{context}_{case}_{tagset}'
    path = inner()
    if mkdir:
        Path(path).mkdir(exist_ok=True, parents=True)
    return path


MODELS = (['longformer'] if args.longformer else []) + (['xlnet'] if args.xlnet else [])

tagset = 'basic'
for corpus in ('swda', 'mrda'):
    for case in ('lower', 'nolower'):
        # Data preparation
        if not args.skip_data_prep:
            for model in MODELS:
                # Transformers dialog-level baseline data-prep
                context = 'dialog'
                run(f'dasg prepare-exp {opts[model]} {opts[corpus]} '
                    f'{opts[case]} -s {tagset} -l {seqlen[model]} -w {outdir(use_seed=False)}')
                # Transformers turn-level baseline data-prep
                context = 'turn'
                run(f'dasg prepare-exp {opts[model]} {opts[corpus]} '
                    f'{opts[case]} -s {tagset} --turns -l 128 {outdir(use_seed=False)}')
        # Model training
        for seed in SEEDS:
            # BiGRU turn-level baseline
            if args.bigru:
                model = 'bigru'
                context = 'turn'
                submit(
                    f'dasg train-bigru -g 1 -s {tagset} -b 30 -e 10 -r {seed} {opts[corpus]} {opts[case]} {outdir()}',
                    name='train')
            # Transformers
            for model in MODELS:
                for context in ('turn', 'dialog'):
                    if not args.dry_run:
                        try:
                            shutil.copytree(outdir(use_seed=False), outdir(use_seed=True, mkdir=False))
                        except FileExistsError:
                            pass
                # Transformers turn-level baseline
                context = 'turn'
                if args.train:
                    submit(f"dasg train-transformer {opts[model]} -b 8 -c 8 -e 10 "
                           f"-a 1 -r {seed} -g 1 {outdir()}", name='train')
                if args.evaluate:
                    ckpts = list(Path(outdir()).glob('checkpoint*.ckpt'))
                    if ckpts:
                        submit(f'dasg evaluate {opts[corpus]} --split test -b 8 --device cpu '
                               f'-o {outdir()}/results.pkl {opts[case]} -s {tagset} --turns {ckpts[0]}', num_gpus=0,
                               name='test')
                    else:
                        print('No checkpoint in directory:', outdir())
                context = 'dialog'
                # Transformers dialog-level
                if args.train:
                    submit(f"dasg train-transformer {opts[model]} -b {bsize[model]} -c 8 -e 10 "
                           f"-a {gacc[model]} -r {seed} -g 1 {outdir()}", name='train')
                if args.evaluate:
                    ckpts = list(Path(outdir()).glob('checkpoint*.ckpt'))
                    if ckpts:
                        submit(f'dasg evaluate {opts[corpus]} -l {seqlen[model]} --split test -b 1 --device cpu '
                               f'-o {outdir()}/results.pkl {opts[case]} -s {tagset} {ckpt}', num_gpus=0, name='test')
                        if model == 'xlnet':
                            submit(f'dasg evaluate {opts[corpus]} -l {seqlen[model]} --split test -b 1 --device cpu '
                                   f'-o {outdir()}/results_noprop.pkl {opts[case]} -s {tagset} -d {ckpt}', num_gpus=0,
                                   name='test_noprop')
                    else:
                        print('No checkpoint in directory:', outdir())
