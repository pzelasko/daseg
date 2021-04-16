import json
from copy import deepcopy
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from cytoolz.itertoolz import identity
from more_itertools import flatten
from torch import nn
from torch.nn import DataParallel
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForTokenClassification, AutoTokenizer, LongformerTokenizer, PreTrainedTokenizer)

from daseg.conversion import predictions_to_dataset
from daseg.data import DialogActCorpus
from daseg.dataloaders.transformers import pad_list_of_arrays, to_transformers_eval_dataloader
from daseg.metrics import compute_original_zhao_kawahara_metrics, compute_segeval_metrics, compute_seqeval_metrics, \
    compute_sklearn_metrics, \
    compute_zhao_kawahara_metrics
from daseg.models.longformer_model import LongformerForTokenClassification

__all__ = ['TransformerModel']


class TransformerModel:
    @staticmethod
    def from_path(model_path: Path, device: str = 'cpu'):
        try:
            return TransformerModel.from_pl_checkpoint(model_path, device)
        except:
            if 'longformer' in str(model_path):
                # This is because we started training Longformer models before they were a part of HF repo;
                # so they were not registered in the AutoModel API
                model_cls = LongformerForTokenClassification
                tok_cls = LongformerTokenizer
            else:
                model_cls = AutoModelForTokenClassification
                tok_cls = AutoTokenizer
            tokenizer = tok_cls.from_pretrained(
                str(model_path),
                **json.load(open(Path(model_path) / 'tokenizer_config.json'))
            )
            model = model_cls.from_pretrained(str(model_path))
        return TransformerModel(
            tokenizer=tokenizer,
            model=model,
            device=device
        )

    @staticmethod
    def from_pl_checkpoint(path: Path, device: str = 'cpu'):
        """
        Works around incompatibility of my old pretrained models with
        the latest huggingface transformers version.
        """
        from daseg.models.transformer_pl import DialogActTransformer

        ckpt = torch.load(path, map_location='cpu')
        labels = ckpt['hyper_parameters']['labels']
        mname = ckpt['hyper_parameters']['model_name_or_path']

        # Create a model but do not try to populate the weights with pretrained transformers
        # as that will fail due to incompatibilities of my code and their latest checkpoints.
        pl_model = DialogActTransformer(labels, mname, pretrained=False)
        if pl_model.config.model_type == 'longformer':
            # We have to manually add position ids to the state dict in the same way as transformers lib does...
            ckpt['state_dict']['model.longformer.embeddings.position_ids'] = \
                torch.arange(pl_model.config.max_position_embeddings).expand((1, -1))
            # Remove extra keys that are no longer needed...
            for k in ["model.longformer.pooler.dense.weight", "model.longformer.pooler.dense.bias"]:
                del ckpt['state_dict'][k]
        # Manually load the state dict
        pl_model.load_state_dict(ckpt['state_dict'])

        # Voila
        return TransformerModel(model=pl_model.model, tokenizer=pl_model.tokenizer, device=device)

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
            dataset: Union[DialogActCorpus, DataLoader],
            batch_size: int = 1,
            window_len: Optional[int] = None,
            window_overlap: Optional[int] = None,
            propagate_context: bool = True,
            compute_metrics: bool = True,
            begin_determines_act: bool = False,
            verbose: bool = False,
            use_joint_coding: bool = True,
            use_turns: bool = False
    ) -> Dict[str, Any]:
        maybe_tqdm = partial(tqdm, desc='Iterating batches') if verbose else identity

        if self.config.model_type == 'xlnet' and propagate_context:
            if isinstance(self.model, DataParallel):
                self.model.module.transformer.mem_len = window_len
                self.model.module.config.mem_len = window_len
            else:
                self.model.transformer.mem_len = window_len
                self.model.config.mem_len = window_len

        # TODO: cleanup the dataloader stuff

        if isinstance(dataset, DialogActCorpus):
            dataloader = to_transformers_eval_dataloader(
                corpus=dataset,
                tokenizer=self.tokenizer,
                model_type=self.config.model_type,
                batch_size=batch_size,
                labels=[label for label, idx in sorted(self.config.label2id.items(), key=itemgetter(1))],
                max_seq_length=None,
                use_joint_coding=use_joint_coding,
                use_turns=use_turns
            )
        else:
            dataloader = dataset

        eval_ce_losses, logits, out_label_ids = zip(*list(maybe_tqdm(
            map(
                partial(
                    predict_batch_in_windows,
                    model=self.model,
                    config=self.config,
                    window_len=window_len,
                    window_overlap=window_overlap,
                    propagate_context=propagate_context,
                    device=self.device,
                ),
                dataloader
            ),
        )))

        pad_token_label_id = CrossEntropyLoss().ignore_index
        out_label_ids = pad_list_of_arrays(out_label_ids, value=pad_token_label_id)
        logits = pad_list_of_arrays(logits, value=0)
        out_label_ids = np.concatenate(out_label_ids, axis=0)
        logits = np.concatenate(logits, axis=0)
        preds = np.argmax(logits, axis=2)

        label_map = {int(k): v for k, v in self.config.id2label.items()}

        out_label_list: List[List[str]] = [[] for _ in range(out_label_ids.shape[0])]
        preds_list: List[List[str]] = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        eval_ce_losses = list(flatten(eval_ce_losses))
        results = {
            "losses": np.array(eval_ce_losses),
            "loss": np.mean(eval_ce_losses),
            "predictions": preds_list,
            "logits": logits,
            "true_labels": out_label_list,
        }
        if isinstance(dataset, DialogActCorpus):
            results["true_dataset"] = dataset
        if compute_metrics:
            results.update({
                "sklearn_metrics": compute_sklearn_metrics(out_label_list, preds_list),
                "seqeval_metrics": compute_seqeval_metrics(out_label_list, preds_list),
                # We show the metrics obtained with Zhao-Kawahara code which computes them differently
                # (apparently the segment insertion errors are not counted)
                "ORIGINAL_zhao_kawahara_metrics": compute_original_zhao_kawahara_metrics(
                    true_turns=out_label_list, pred_turns=preds_list
                ),
            })
            # Pk and B metrics

        if isinstance(dataset, DialogActCorpus):
            if use_turns:
                dataset = DialogActCorpus(dialogues={str(i): turn for i, turn in enumerate(dataset.turns)})
            results["dataset"] = predictions_to_dataset(
                dataset,
                preds_list,
                begin_determines_act=begin_determines_act,
                use_joint_coding=use_joint_coding,
            )
            if compute_metrics:
                results["zhao_kawahara_metrics"] = compute_zhao_kawahara_metrics(
                    true_dataset=dataset, pred_dataset=results['dataset']
                )
                results["segeval_metrics"] = compute_segeval_metrics(
                    true_dataset=dataset, pred_dataset=results['dataset']
                )

        return results


def predict_batch_in_windows(
        batch: Tuple[torch.Tensor],
        model,
        config,
        window_len: Optional[int] = None,
        window_overlap: Optional[int] = None,
        propagate_context: bool = True,
        device: str = 'cpu',
):
    if window_overlap is not None:
        raise ValueError("Overlapping windows processing not implemented.")
    else:
        window_overlap = 0

    use_xlnet_memory = (config.model_type == 'xlnet' and propagate_context
                        and config.mem_len is not None and config.mem_len > 0)

    batch = tuple(t.to(device) for t in batch)

    if window_len is None:
        windows = [batch]
    else:
        maxlen = batch[0].shape[1]
        window_shift = window_len - window_overlap
        windows = (
            [t[:, i: i + window_len].contiguous().to(device) for t in batch]
            for i in range(0, maxlen, window_shift)
        )

    ce_loss, logits = [], []

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
            batch_logits = outputs[1]
            # Consume the outputs according to a specific model
            try:
                ce_loss.extend(batch_ce_loss.detach().cpu().numpy())
            except:
                ce_loss.append(batch_ce_loss.detach().cpu().numpy())
            logits.append(batch_logits.detach().cpu().numpy())
            if use_xlnet_memory:
                mems = outputs[2]
    # workaround for PyTorch file descriptor leaks:
    # https://github.com/pytorch/pytorch/issues/973
    returns = ce_loss, np.concatenate(logits, axis=1), deepcopy(batch[3].detach().cpu().numpy())
    for t in batch:
        del t
    return returns
