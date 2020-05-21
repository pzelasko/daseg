import json
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional, List, Tuple, Union

import numpy as np
import torch
from more_itertools import flatten
from torch import nn
from torch.nn import DataParallel
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
from tqdm.autonotebook import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, PreTrainedTokenizer

from daseg.data import SwdaDataset, to_transformers_ner_dataset

__all__ = ['TransformerModel']

from daseg.metrics import compute_sklearn_metrics, compute_seqeval_metrics, compute_zhao_kawahara_metrics


class TransformerModel:
    @staticmethod
    def from_path(model_dir: Path, device: str = 'cpu'):
        return TransformerModel(
            tokenizer=AutoTokenizer.from_pretrained(
                model_dir,
                **json.load(open(Path(model_dir) / 'tokenizer_config.json'))
            ),
            model=AutoModelForTokenClassification.from_pretrained(model_dir),
            device=device
        )

    def __init__(self, model: nn.Module, tokenizer: PreTrainedTokenizer, device: str):
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.device = device

    @property
    def config(self):
        if isinstance(self.model, DataParallel):
            return self.model.module.config
        return self.model.config

    def predict(
            self,
            dataset: Union[SwdaDataset, DataLoader],
            batch_size: int = 1,
            window_len: Optional[int] = None,
            window_overlap: Optional[int] = None,
            crf_decoding: bool = False
    ) -> Dict[str, Any]:
        # TODO:
        # if self.config.model_type == 'xlnet':
        #     self.model.set_memory(window_len)

        # TODO: cleanup the dataloader stuff

        labels = list(self.config.label2id.keys())

        if isinstance(dataset, SwdaDataset):
            # Tokenize just to find what's the maximum length after tokenization
            tok_results = self.tokenizer.batch_encode_plus(
                [' '.join(c.words(add_turn_token=True)) for c in dataset.calls],
                return_input_lengths=True
            )
            max_len = max(tok_results['input_len'])
            dataloader = dataset.to_transformers_ner_format(
                tokenizer=self.tokenizer,
                max_seq_length=max_len,
                model_type=self.config.model_type,
                batch_size=batch_size,
                labels=self.config.label2id.keys()
            )
        else:
            dataloader = dataset

        eval_ce_losses, eval_crf_losses, logits, out_label_ids = zip(*list(tqdm(
            map(
                partial(
                    predict_batch_in_windows,
                    model=self.model,
                    config=self.config,
                    window_len=window_len,
                    window_overlap=window_overlap,
                    device=self.device
                ),
                dataloader
            ),
            desc=f'Predicting dialog acts (batches of {batch_size})',
            leave=False
        )))

        out_label_ids = np.concatenate(out_label_ids, axis=0)
        logits = np.concatenate(logits, axis=0)
        if crf_decoding and hasattr(self.model, 'crf'):
            preds, lls = zip(*self.model.crf.viterbi_tags(torch.from_numpy(logits)))
        else:
            preds = np.argmax(logits, axis=2)

        pad_token_label_id = CrossEntropyLoss().ignore_index
        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list: List[List[str]] = [[] for _ in range(out_label_ids.shape[0])]
        preds_list: List[List[str]] = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        eval_ce_losses = list(flatten(eval_ce_losses))
        eval_crf_losses = list(flatten(eval_crf_losses))
        results = {
            "losses": np.array(eval_ce_losses),
            "crf_losses": np.array(eval_crf_losses),
            "loss": np.mean(eval_ce_losses),
            "crf_loss": np.mean(eval_crf_losses),
            "predictions": preds_list,
            "logits": logits,
            "true_labels": out_label_list,
            "sklearn_metrics": compute_sklearn_metrics(out_label_list, preds_list),
            "seqeval_metrics": compute_seqeval_metrics(out_label_list, preds_list)
        }
        if isinstance(dataset, SwdaDataset):
            results["dataset"] = predictions_to_dataset(dataset, preds_list)
            results["zhao_kawahara_metrics"] = compute_zhao_kawahara_metrics(
                true_dataset=dataset, pred_dataset=results['dataset']
            )

        return results


def predict_batch_in_windows(
        batch: Tuple[torch.Tensor],
        model,
        config,
        window_len: Optional[int] = None,
        window_overlap: Optional[int] = None,
        device: str = 'cpu'
):
    if window_overlap is not None:
        raise ValueError("Overlapping windows processing not implemented.")
    else:
        window_overlap = 0

    use_xlnet_memory = (config.model_type == 'xlnet' and config.output_past
                        and config.mem_len is not None and config.mem_len > 0)

    has_crf = hasattr(model, 'crf') or (isinstance(model, DataParallel) and hasattr(model.module, 'crf'))

    batch = tuple(t.to('cpu') for t in batch)

    if window_len is None:
        windows = [batch]
    else:
        maxlen = batch[0].shape[1]
        window_shift = window_len - window_overlap
        windows = (
            [t[:, i: i + window_len].contiguous().to(device) for t in batch]
            for i in trange(0, maxlen, window_shift, leave=False, desc='Traversing batch in windows')
        )

    ce_loss, crf_loss, logits = [], [], []

    mems = None
    with torch.no_grad():
        for window in windows:
            # Construct the input according to specific Transformer model
            inputs = {"input_ids": window[0], "attention_mask": window[1], "labels": window[3]}
            if config.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    window[2] if config.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don't use segment_ids
            if use_xlnet_memory:
                inputs['mems'] = mems
            # Compute
            outputs = model(**inputs)
            batch_ce_loss = outputs[0]
            batch_crf_loss = outputs[1] if has_crf else torch.zeros_like(batch_ce_loss)
            batch_logits = outputs[2] if has_crf else outputs[1]
            # Consume the outputs according to a specific model
            try:
                ce_loss.extend(batch_ce_loss.detach().cpu().numpy())
                crf_loss.extend(batch_crf_loss.detach().cpu().numpy())
            except:
                ce_loss.append(batch_ce_loss.detach().cpu().numpy())
                crf_loss.append(batch_crf_loss.detach().cpu().numpy())
            logits.append(batch_logits.detach().cpu().numpy())
            if use_xlnet_memory:
                mems = outputs[2]
    return ce_loss, crf_loss, np.concatenate(logits, axis=1), batch[3].detach().cpu().numpy()


def predictions_to_dataset(original_dataset: SwdaDataset, predictions: List[List[str]]) -> SwdaDataset:
    # Does some possibly unnecessary back-and-forth, but gets the job done!
    with NamedTemporaryFile('w+') as f:
        for call, pred_tags in zip(original_dataset.calls, predictions):
            lines = to_transformers_ner_dataset(call)
            words, tags = zip(*[l.split() for l in lines])
            for w, t in zip(words, pred_tags):
                print(f'{w} {t}', file=f)
            print(file=f)
        f.flush()
        return SwdaDataset.from_transformers_predictions(f.name)
