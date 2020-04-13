#!/usr/bin/env python

import logging

import hydra
import torch
from hydra import utils
from omegaconf import DictConfig
from seqeval.metrics import classification_report

from daseg import SwdaDataset, TransformerModel

log = logging.getLogger(__name__)


@hydra.main(config_path="../config.yaml")
def my_app(cfg: DictConfig):
    logging.info(cfg.pretty())
    torch.set_num_interop_threads(cfg.interop_threads)
    torch.set_num_threads(cfg.intraop_threads)
    available_modes = {
        'prepare_data': prepare_data,
        'train': lambda _: train,
        'test': test
    }
    available_modes.get(
        cfg.mode,
        lambda cfg_: logging.error(f'No such mode: {cfg_.mode} - available modes: {" ".join(available_modes.keys())}')
    )(cfg)


def prepare_data(cfg: DictConfig):
    dataset = SwdaDataset.from_path(utils.to_absolute_path(cfg.data.swda_path))
    for split_name, split_dataset in dataset.train_dev_test_split().items():
        split_dataset.dump_for_transformers_ner(
            f'{utils.to_absolute_path(cfg.data.text_path)}/{split_name}.txt.tmp',
            acts_count_per_sample=cfg.data.dialog_acts_per_window if split_name != 'test' else None,
            acts_count_overlap=cfg.data.dialog_acts_overlap if split_name != 'test' else None
        )


def train(cfg: DictConfig):
    raise NotImplementedError()


def test(cfg: DictConfig):
    dataset = SwdaDataset.from_path(
        utils.to_absolute_path(cfg.data.swda_path)
    ).train_dev_test_split()['test']
    model = TransformerModel.from_path(utils.to_absolute_path(cfg.model.path))
    results = model.predict(dataset, batch_size=4, window_len=512)
    for x in 'accuracy f1 precision recall'.split():
        logging.info(results[x], results[x])
    logging.info(classification_report(results['labels'], results['predictions']))


if __name__ == "__main__":
    my_app()
