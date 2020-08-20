import shutil
from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile

use_grid = True
pause = True

SCRIPT_TEMPLATE = """#!/usr/bin/env bash

conda activate swda
export CUDA_VISIBLE_DEVICES=$(free-gpu -n {num_gpus})
cd {work_dir}
{cmd}
"""

QSUB_TEMPLATE = "qsub -l \"hostname=c*,gpu={num_gpus} -q g.q -e {logerr} -o {logout} {script}"

WORK_DIR = '/export/c12/pzelasko/daseg/daseg'
EXP_DIR = str(Path(WORK_DIR) / 'journal')
SEEDS = (42, 43, 44)

opts = {
    'longformer': '--model-name-or-path allenai/longformer-base',
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


def submit(cmd: str, work_dir: str = WORK_DIR, num_gpus: int = 1):
    if use_grid:
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
                logout=f'{outdir()}/stdout.txt'
            )
            print(qsub)
            print(script, end='\n\n')
            run(qsub, text=True, shell=True)
    else:
        run(cmd)
    if pause:
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
                    shutil.copytree(outdir(use_seed=False), outdir(use_seed=True))
                # Transformers turn-level baseline
                context = 'turn'
                submit(f"dasg train-transformer {opts[model]} -b 30 -c 30 -e 10 "
                       f"-a 1 -r {seed} -g 1 {outdir()}")
                context = 'dialog'
                submit(f"dasg train-transformer {opts[model]} -b {bsize[model]} -c 8 -e 10 "
                       f"-a {gacc[model]} -r {seed} -g 1 {outdir()}")
