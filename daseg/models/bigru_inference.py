from functools import partial
from pathlib import Path
from typing import Any, Dict, Union

import torch
from cytoolz.itertoolz import identity
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from daseg.data import DialogActCorpus
from daseg.dataloaders.turns import SingleTurnDataset, padding_collate_fn
from daseg.models.bigru import ZhaoKawaharaBiGru


class BigruSegmenter:
    @staticmethod
    def from_path(model_path: Path, device: str = 'cpu'):
        pl_model = ZhaoKawaharaBiGru.load_from_checkpoint(str(model_path), map_location=device)
        return BigruSegmenter(
            model=pl_model,
            device=device
        )

    def __init__(self, model: ZhaoKawaharaBiGru, device: str):
        self.model: ZhaoKawaharaBiGru = model.to(device).eval()
        self.device = device

    def predict(
            self,
            dataset: Union[DialogActCorpus, DataLoader],
            batch_size: int = 1,
            verbose: bool = False,
    ) -> Dict[str, Any]:
        maybe_tqdm = partial(tqdm, desc='Iterating batches') if verbose else identity

        if isinstance(dataset, DialogActCorpus):
            dataloader = DataLoader(
                dataset=SingleTurnDataset(dataset, word2idx=self.model.vocab, tag2idx=self.model.labels),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=padding_collate_fn,
                num_workers=1
            )
        else:
            dataloader = dataset

        outputs = []
        for idx, batch in maybe_tqdm(enumerate(dataloader)):
            with torch.no_grad():
                outputs.append(self.model.test_step(batch, idx))
        results = self.model.test_epoch_end(outputs)
        return results
