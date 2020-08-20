import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

parser = argparse.ArgumentParser()
parser.add_argument('--use-grid', type=bool, default=True)
parser.add_argument('--pause', type=bool, default=False)
parser.add_argument('--skip-data-prep', type=bool, default=False)
args = parser.parse_args()

SCRIPT_TEMPLATE = """#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$(free-gpu -n {num_gpus})
cd {work_dir}
/home/pzelasko/miniconda3/envs/swda/bin/python {cmd}
"""

QSUB_TEMPLATE = "qsub -l \"hostname=c*,gpu={num_gpus} -q {queue} -e {logerr} -o {logout} bash {script}"

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
    subprocess.run(cmd, shell=True, text=True)


def submit(cmd: str, work_dir: str = WORK_DIR, num_gpus: int = 1):
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
                logerr=f'{outdir()}/stderr.txt',
                logout=f'{outdir()}/stdout.txt',
                queue='g.q' if num_gpus else 'all.q'
            )
            print(qsub)
            print(script, end='\n\n')
            run(qsub)
    else:
        run(cmd)
    if args.pause:
        input()


Path(EXP_DIR).mkdir(parents=True, exist_ok=True)


def outdir(use_seed=True):
    if use_seed:
        return f'{EXP_DIR}/{model}_{corpus}_{context}_{case}_{tagset}_{seed}'
    return f'{EXP_DIR}/{model}_{corpus}_{context}_{case}_{tagset}'


tagset = 'basic'
for corpus in ('swda', 'mrda'):
    for case in ('lower', 'nolower'):
        # Data preparation
        if not args.skip_data_prep:
            for model in ('longformer', 'xlnet'):
                # Transformers dialog-level baseline data-prep
                context = 'turn'
                run(f'dasg prepare-exp {opts[model]} {opts[corpus]} '
                    f'{opts[case]} -s {tagset} -l {seqlen[model]} -w {outdir(use_seed=False)}')
                # Transformers turn-level baseline data-prep
                context = 'dialog'
                run(f'dasg prepare-exp {opts[model]} {opts[corpus]} '
                    f'{opts[case]} -s {tagset} --turns {outdir(use_seed=False)}')
        # Model training
        for seed in SEEDS:
            # BiGRU turn-level baseline
            model = 'bigru'
            context = 'turn'
            submit(f'dasg train-bigru -s {tagset} -b 30 -e 10 -r {seed} {opts[corpus]} {opts[case]} {outdir()}')
            # Transformers
            for model in ('longformer', 'xlnet'):
                for context in ('turn', 'dialog'):
                    try:
                        shutil.copytree(outdir(use_seed=False), outdir(use_seed=True))
                    except FileExistsError:
                        pass
                # Transformers turn-level baseline
                context = 'turn'
                submit(f"dasg train-transformer {opts[model]} -b 30 -c 30 -e 10 "
                       f"-a 1 -r {seed} -g 1 {outdir()}")
                context = 'dialog'
                submit(f"dasg train-transformer {opts[model]} -b {bsize[model]} -c 8 -e 10 "
                       f"-a {gacc[model]} -r {seed} -g 1 {outdir()}")
