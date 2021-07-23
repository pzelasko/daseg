import json
from copy import deepcopy
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Union

import numpy as np
import torch
from cytoolz.itertoolz import identity
from more_itertools import flatten
from torch import nn
from torch.nn import DataParallel
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    PreTrainedTokenizer, LongformerTokenizer
)

from daseg.conversion import predictions_to_dataset
from daseg.data import DialogActCorpus
from daseg.dataloaders.transformers import to_transformers_eval_dataloader, pad_list_of_arrays
from daseg.metrics import compute_sklearn_metrics, compute_seqeval_metrics, compute_zhao_kawahara_metrics, \
    compute_original_zhao_kawahara_metrics, compute_zhao_kawahara_metrics_speech
from daseg.models.longformer_model import LongformerForTokenClassification
from daseg.models.transformer_pl import DialogActTransformer, XFormer

__all__ = ['TransformerModel']


class TransformerModel:
    @staticmethod
    def from_path(model_path: Path, device: str = 'cpu'):
        pl_model = DialogActTransformer.load_from_checkpoint(str(model_path), map_location=device)
        #model, tokenizer = pl_model.model, pl_model.tokenizer
        model, tokenizer = pl_model.model, None
        return TransformerModel(
            tokenizer=tokenizer,
            model=model,
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
            dataset: Union[DialogActCorpus, DataLoader],
            label_scheme: str,
            batch_size: int = 1,
            window_len: Optional[int] = None,
            window_overlap: Optional[int] = None,
            propagate_context: bool = False,
            compute_metrics: bool = True,
            begin_determines_act: bool = False,
            verbose: bool = True,
            use_joint_coding: bool = False,
            use_turns: bool = False,
    ) -> Dict[str, Any]:
        maybe_tqdm = partial(tqdm, desc='Iterating batches') if verbose else identity

        dataloader = dataset

        eval_ce_losses, logits, out_label_ids, true_seg_boundaries, batch_keys = zip(*list(maybe_tqdm(
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
            "true_seg_boundaries": true_seg_boundaries,
            "utt_id": batch_keys,
        }
        if compute_metrics:
            results.update({
                "sklearn_metrics": compute_sklearn_metrics(out_label_list, preds_list),
                "seqeval_metrics": compute_seqeval_metrics(out_label_list, preds_list),
                # We show the metrics obtained with Zhao-Kawahara code which computes them differently
                # (apparently the segment insertion errors are not counted)
                "ORIGINAL_zhao_kawahara_metrics": compute_original_zhao_kawahara_metrics(
                    true_turns=out_label_list, pred_turns=preds_list),
                "zhao_kawahara_metrics": compute_zhao_kawahara_metrics_speech(true_labels=out_label_list, pred_labels=preds_list, true_seg_boundaries=true_seg_boundaries, label_scheme=label_scheme)
            })
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

    use_xlnet_memory = False #(config.model_type == 'xlnet' and propagate_context and config.mem_len is not None and config.mem_len > 0)

    batch_keys = batch[-1]
    print(f'assuming last field in the batch contains key of the utterance')
    batch = batch[:-1]
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

    ce_loss, logits, labels = [], [], []

    mems = None
    with torch.no_grad():
        for window in windows:
            # Construct the input according to specific Transformer model
            #inputs = {"input_ids": window[0], "attention_mask": window[1], "labels": window[3]}
            inputs = {"inputs_embeds": window[0], "attention_mask": window[1], "labels": window[2], "token_type_ids":window[4]}

            #if config.model_type != "distilbert":
            #    inputs["token_type_ids"] = (
            #        window[2] if config.model_type in ["bert", "xlnet"] else None
            #    )  # XLM and RoBERTa don't use segment_ids
            #if use_xlnet_memory:
            #    inputs['mems'] = mems
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

            try:
                out_label_ids = outputs.subsampled_labels.detach().cpu().numpy()
            except:
                out_label_ids = batch[2].detach().cpu().numpy()
            labels.append(out_label_ids)
            if use_xlnet_memory:
                mems = outputs[2]
    # workaround for PyTorch file descriptor leaks:
    # https://github.com/pytorch/pytorch/issues/973
    returns = ce_loss, np.concatenate(logits, axis=1), np.concatenate(labels, axis=1), deepcopy(batch[3].detach().cpu().numpy()), batch_keys 
    #returns = ce_loss, np.concatenate(logits, axis=1), deepcopy(batch[3].detach().cpu().numpy())
    for t in batch:
        del t
    return returns


class TransformerTextModelTokenClassification:
    @staticmethod
    def from_path(model_path: Path, device: str = 'cpu'):
        pl_model = DialogActTransformer.load_from_checkpoint(str(model_path), map_location=device)
        model, tokenizer = pl_model.model, pl_model.tokenizer
        return TransformerModel(
            tokenizer=tokenizer,
            model=model,
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
            dataset: Union[DialogActCorpus, DataLoader],
            label_scheme: str,
            batch_size: int = 1,
            window_len: Optional[int] = None,
            window_overlap: Optional[int] = None,
            propagate_context: bool = False,
            compute_metrics: bool = True,
            begin_determines_act: bool = False,
            verbose: bool = True,
            use_joint_coding: bool = False,
            use_turns: bool = False,
    ) -> Dict[str, Any]:
        maybe_tqdm = partial(tqdm, desc='Iterating batches') if verbose else identity

        dataloader = dataset

        eval_ce_losses, logits, out_label_ids, true_seg_boundaries, batch_keys = zip(*list(maybe_tqdm(
            map(
                partial(
                    predict_batch_in_windows_TextTokenClassification,
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
            "true_seg_boundaries": true_seg_boundaries,
            "utt_id": batch_keys,
        }
        if compute_metrics:
            results.update({
                "sklearn_metrics": compute_sklearn_metrics(out_label_list, preds_list),
                "seqeval_metrics": compute_seqeval_metrics(out_label_list, preds_list),
                # We show the metrics obtained with Zhao-Kawahara code which computes them differently
                # (apparently the segment insertion errors are not counted)
                "ORIGINAL_zhao_kawahara_metrics": compute_original_zhao_kawahara_metrics(
                    true_turns=out_label_list, pred_turns=preds_list),
                "zhao_kawahara_metrics": compute_zhao_kawahara_metrics_speech(true_labels=out_label_list, pred_labels=preds_list, true_seg_boundaries=true_seg_boundaries, label_scheme=label_scheme)
            })
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

        return results


def predict_batch_in_windows_TextTokenClassification(
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

    use_xlnet_memory = False #(config.model_type == 'xlnet' and propagate_context and config.mem_len is not None and config.mem_len > 0)

    if window_len is None:
        windows = [batch]
    else:
        window_shift = window_len - window_overlap
        #windows = (
        #    [t[:, i: i + window_len].contiguous().to(device) for t in batch]
        #    for i in range(0, maxlen, window_shift)
        #)

    ce_loss, logits, labels = [], [], []

    mems = None
    with torch.no_grad():
        utt_logits = []
        batch_keys = batch['utt_id']
        max_len_batch = batch['input_ids'].shape[1] # sequence length 
        for i in range(0, max_len_batch, window_shift):
            #print(i)
            inputs = {}
            #for ip in batch.keys():
            for ip in model.model_input_keys:
                if not 'utt_id' in ip:
                    inputs[ip] = batch[ip][:, i: i + window_len].contiguous().to(device)

            if config.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    inputs['token_type_ids'] if config.model_type in ["bert", "xlnet"] else None
                ) # XLM and RoBERTa don't use segment_ids

            #if use_xlnet_memory:
            #    inputs['mems'] = mems
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

            try:
                out_label_ids = outputs.subsampled_labels.detach().cpu().numpy()
            except:
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            labels.append(out_label_ids)
            if use_xlnet_memory:
                mems = outputs[2]
    # workaround for PyTorch file descriptor leaks:
    # https://github.com/pytorch/pytorch/issues/973
    returns = ce_loss, np.concatenate(logits, axis=1), np.concatenate(labels, axis=1), deepcopy(batch['seg_boundaries'].detach().cpu().numpy()), batch_keys 
    #returns = ce_loss, np.concatenate(logits, axis=1), deepcopy(batch[3].detach().cpu().numpy())
    for t in batch:
        del t
    return returns


class TransformerModelSeqClassification:
    @staticmethod
    def from_path(model_path: Path, device: str = 'cpu'):
        pl_model = DialogActTransformer.load_from_checkpoint(str(model_path), map_location=device)
        #model, tokenizer = pl_model.model, pl_model.tokenizer
        model, tokenizer = pl_model.model, None
        return TransformerModel(
            tokenizer=tokenizer,
            model=model,
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
            dataset: Union[DialogActCorpus, DataLoader],
            label_scheme: str,
            batch_size: int = 1,
            window_len: Optional[int] = None,
            window_overlap: Optional[int] = None,
            propagate_context: bool = False,
            compute_metrics: bool = True,
            begin_determines_act: bool = False,
            verbose: bool = True,
            use_joint_coding: bool = False,
            use_turns: bool = False,
    ) -> Dict[str, Any]:
        maybe_tqdm = partial(tqdm, desc='Iterating batches') if verbose else identity

        dataloader = dataset

        eval_ce_losses, logits, out_label_ids, true_seg_boundaries, batch_keys = zip(*list(maybe_tqdm(
            map(
                partial(
                    predict_batch_in_windows_SequenceClassification,
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
        
        assert len(eval_ce_losses) == len(dataloader)
        print(f'It looks like we evaluated on {len(eval_ce_losses)} samples, now proceeding to metric calculation')

        pad_token_label_id = CrossEntropyLoss().ignore_index
        #out_label_ids = pad_list_of_arrays(out_label_ids, value=pad_token_label_id)
        #logits = pad_list_of_arrays(logits, value=0)
        out_label_ids = np.concatenate(out_label_ids, axis=0)
        logits = np.concatenate(logits, axis=0)
        logits = logits.squeeze()
        if self.config.num_labels > 1:
            preds = np.argmax(logits, axis=-1)
            label_map = {int(k): v for k, v in self.config.id2label.items()}
    
            out_label_list = [[label_map[i]  for i in out_label_ids]]
            preds_list = [[label_map[i] for i in preds]]
    
            eval_ce_losses = list(flatten(eval_ce_losses))
            results = {
                "losses": np.array(eval_ce_losses),
                "loss": np.mean(eval_ce_losses),
                "predictions": preds_list,
                "logits": logits,
                "true_labels": out_label_list,
                "true_seg_boundaries": true_seg_boundaries,
                "utt_id": batch_keys,
            }
            if compute_metrics:
                results.update({
                    "sklearn_metrics": compute_sklearn_metrics(out_label_list, preds_list),
                })

        else:
            preds_list = [list(logits)]
            out_label_list = [list(out_label_ids)]
            eval_ce_losses = list(flatten(eval_ce_losses))
            results = {
                "losses": np.array(eval_ce_losses),
                "loss": np.mean(eval_ce_losses),
                "predictions": preds_list,
                "logits": logits,
                "true_labels": out_label_list,
                "utt_id": batch_keys,
            }
        
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

        return results


def predict_batch_in_windows_SequenceClassification(
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

    use_xlnet_memory = False #(config.model_type == 'xlnet' and propagate_context and config.mem_len is not None and config.mem_len > 0)

    batch_keys = batch[-1]
    print(f'assuming last field in the batch contains key of the utterance so removing it from the batch to be able to convert batch to tensors')
    batch = batch[:-1]
    batch = tuple(t.to(device) for t in batch)

    if window_len is None:
        windows = [batch]
    else:
        maxlen = batch[0].shape[1]
        window_shift = window_len - window_overlap
        #windows = []
        #for i in range(0, maxlen, window_shift):
        #    temp = [t[:, i: i + window_len].contiguous().to(device) for t in batch]
        #    windows.append(temp)

        #windows = (
        #    [t[:, i: i + window_len].contiguous().to(device) if ind!=2 else t.contiguous().to(device) for ind,t in enumerate(batch)]
        #    for i in range(0, maxlen, window_shift)
        #)
        

    ce_loss, logits, labels = [], [], []

    mems = None
    with torch.no_grad():
        #for ind,t in enumerate(batch):
        if 1:
            utt_logits = []
            for i in range(0, maxlen, window_shift):
                window = [t[:, i: i + window_len].contiguous().to(device) if ind!=2 else t.contiguous().to(device) for ind,t in enumerate(batch)]

                #for window in windows:
                # Construct the input according to specific Transformer model
                #inputs = {"input_ids": window[0], "attention_mask": window[1], "labels": window[3]}
                inputs = {"input_ids": window[0], "attention_mask": window[1], "labels": window[2]}
    
                if config.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        window[4] if config.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don't use segment_ids
                #if use_xlnet_memory:
                #    inputs['mems'] = mems
                # Compute
                outputs = model(**inputs)
                batch_ce_loss = outputs[0]
                batch_logits = outputs[1]
                # Consume the outputs according to a specific model
                try:
                    ce_loss.extend(batch_ce_loss.detach().cpu().numpy())
                except:
                    ce_loss.append(batch_ce_loss.detach().cpu().numpy())
                utt_logits.append(batch_logits.detach().cpu().numpy())
    
                try:
                    out_label_ids = outputs.subsampled_labels.detach().cpu().numpy()
                except:
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                labels.append(out_label_ids)
                if use_xlnet_memory:
                    mems = outputs[2]
            if len(utt_logits) > 1:
                logits.append([[np.mean(np.concatenate(utt_logits, axis=0), axis=0)]])
            else:
                logits.append(utt_logits)
           
    logits = np.vstack(logits)
    # workaround for PyTorch file descriptor leaks:
    # https://github.com/pytorch/pytorch/issues/973
    returns = ce_loss, logits, deepcopy(batch[2].detach().cpu().numpy()), deepcopy(batch[3].detach().cpu().numpy()), batch_keys 
    for t in batch:
        del t
    return returns


class TransformerMultimodalSeqClassificationInference:
    @staticmethod
    def from_path(model_path: Path, device: str = 'cpu'):
        pl_model = DialogActTransformer.load_from_checkpoint(str(model_path), map_location=device)
        #model, tokenizer = pl_model.model, pl_model.tokenizer
        model, tokenizer = pl_model.model, None
        return TransformerModel(
            tokenizer=tokenizer,
            model=model,
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
            dataset: Union[DialogActCorpus, DataLoader],
            label_scheme: str,
            batch_size: int = 1,
            window_len: Optional[int] = None,
            window_overlap: Optional[int] = None,
            propagate_context: bool = False,
            compute_metrics: bool = True,
            begin_determines_act: bool = False,
            verbose: bool = True,
            use_joint_coding: bool = False,
            use_turns: bool = False,
            max_len_text: int = 512,
            max_len_speech: int = None
    ) -> Dict[str, Any]:
        maybe_tqdm = partial(tqdm, desc='Iterating batches') if verbose else identity

        dataloader = dataset

        eval_ce_losses, logits, out_label_ids, true_seg_boundaries, batch_keys = zip(*list(maybe_tqdm(
            map(
                partial(
                    predict_batch_in_windows_Multimodal_SequenceClassification,
                    model=self.model,
                    config=self.config,
                    window_len=window_len,
                    window_overlap=window_overlap,
                    propagate_context=propagate_context,
                    device=self.device,
                    max_len_text=max_len_text,
                    max_len_speech=max_len_speech,
                ),
                dataloader
            ),
        )))

        pad_token_label_id = CrossEntropyLoss().ignore_index
        #out_label_ids = pad_list_of_arrays(out_label_ids, value=pad_token_label_id)
        #logits = pad_list_of_arrays(logits, value=0)
        out_label_ids = np.concatenate(out_label_ids, axis=0)
        logits = np.concatenate(logits, axis=0)
        logits = logits.squeeze()
        preds = np.argmax(logits, axis=-1)

        label_map = {int(k): v for k, v in self.config.id2label.items()}

        out_label_list = [[label_map[i]  for i in out_label_ids]]
        preds_list = [[label_map[i] for i in preds]]

        eval_ce_losses = list(flatten(eval_ce_losses))
        results = {
            "losses": np.array(eval_ce_losses),
            "loss": np.mean(eval_ce_losses),
            "predictions": preds_list,
            "logits": logits,
            "true_labels": out_label_list,
            "true_seg_boundaries": true_seg_boundaries,
            "utt_id": batch_keys,
        }
        if compute_metrics:
            results.update({
                "sklearn_metrics": compute_sklearn_metrics(out_label_list, preds_list),
            })
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

        return results


def predict_batch_in_windows_Multimodal_SequenceClassification(
        batch: Tuple[torch.Tensor],
        model,
        config,
        window_len: Optional[int] = None,
        window_overlap: Optional[int] = None,
        propagate_context: bool = True,
        device: str = 'cpu',
        max_len_text: int = 512,
        max_len_speech: int = None,
):
    if window_overlap is not None:
        raise ValueError("Overlapping windows processing not implemented.")
    else:
        window_overlap = 0

    use_xlnet_memory = False #(config.model_type == 'xlnet' and propagate_context and config.mem_len is not None and config.mem_len > 0)

    batch_keys = batch['speech_utt_id']
    assert batch['speech_utt_id'] == batch['text_utt_id']


    if window_len is None:
        windows = [batch]
    else:
        window_shift = window_len - window_overlap

    ce_loss, logits, labels = [], [], []
    mems = None
    #import pdb; pdb.set_trace()
    with torch.no_grad():
        #for ind,t in enumerate(batch):
        if 1:
            utt_logits = []
            max_len_batch = batch['text_ip'].shape[1] # sequence length 
            for i in range(0, max_len_batch, window_shift):
                #print(i)
                inputs = {}
                for ip in batch.keys():
                    if not 'utt_id' in ip:
                        if 'text' in ip:
                            if 'labels' in ip:
                                inputs[ip] = batch[ip].contiguous().to(device)
                            else:
                                inputs[ip] = batch[ip][:, i: i + window_len].contiguous().to(device)
                        else:
                            ## currently the speech model is ResNet which can handle any sequence length so not shortening for inference
                            inputs[ip] = batch[ip].contiguous().to(device)

                # Compute
                #for ip in inputs.keys():
                #    print(ip, inputs[ip].shape, batch[ip].shape)
                #import pdb; pdb.set_trace()
                outputs = model(**inputs)
                batch_ce_loss = outputs[0]
                batch_logits = outputs[1]
                # Consume the outputs according to a specific model
                try:
                    ce_loss.extend(batch_ce_loss.detach().cpu().numpy())
                except:
                    ce_loss.append(batch_ce_loss.detach().cpu().numpy())
                utt_logits.append(batch_logits.detach().cpu().numpy())
                try:
                    out_label_ids = outputs.subsampled_labels.detach().cpu().numpy()
                except:
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
                labels.append(out_label_ids)
                if use_xlnet_memory:
                    mems = outputs[2]

            if len(utt_logits) > 1:
                logits.append([[np.mean(np.concatenate(utt_logits, axis=0), axis=0)]])
            else:
                logits.append(utt_logits)
           
    logits = np.vstack(logits)
    # workaround for PyTorch file descriptor leaks:
    # https://github.com/pytorch/pytorch/issues/973
    returns = ce_loss, logits, deepcopy(batch['labels'].detach().cpu().numpy()), deepcopy(batch['text_seg_boundaries'].detach().cpu().numpy()), batch_keys
    for t in batch:
        del t
    return returns


class TransformerMultimodalMultiLossSeqClassificationInference:
    @staticmethod
    def from_path(model_path: Path, device: str = 'cpu'):
        pl_model = DialogActTransformer.load_from_checkpoint(str(model_path), map_location=device)
        #model, tokenizer = pl_model.model, pl_model.tokenizer
        model, tokenizer = pl_model.model, None
        return TransformerModel(
            tokenizer=tokenizer,
            model=model,
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
            dataset: Union[DialogActCorpus, DataLoader],
            label_scheme: str,
            batch_size: int = 1,
            window_len: Optional[int] = None,
            window_overlap: Optional[int] = None,
            propagate_context: bool = False,
            compute_metrics: bool = True,
            begin_determines_act: bool = False,
            verbose: bool = True,
            use_joint_coding: bool = False,
            use_turns: bool = False,
            max_len_text: int = 512,
            max_len_speech: int = None
    ) -> Dict[str, Any]:
        maybe_tqdm = partial(tqdm, desc='Iterating batches') if verbose else identity

        dataloader = dataset

        eval_ce_losses, logits, out_label_ids, true_seg_boundaries, batch_keys = zip(*list(maybe_tqdm(
            map(
                partial(
                    predict_batch_in_windows_MultimodalMultiLoss_SequenceClassification,
                    model=self.model,
                    config=self.config,
                    window_len=window_len,
                    window_overlap=window_overlap,
                    propagate_context=propagate_context,
                    device=self.device,
                    max_len_text=max_len_text,
                    max_len_speech=max_len_speech,
                ),
                dataloader
            ),
        )))

        num_op_ind = 2
        new_logits = [[] for op_ind in range(num_op_ind)]
        eval_ce_losses_new = [[] for op_ind in range(num_op_ind)]
        out_label_ids_new = [[] for op_ind in range(num_op_ind)]
        for op_ind in range(num_op_ind):
            new_logits[op_ind] = [logits[ind][op_ind] for ind in range(len(logits))]
            eval_ce_losses_new[op_ind] = [eval_ce_losses[ind][op_ind] for ind in range(len(eval_ce_losses))]
            out_label_ids_new[op_ind] = [out_label_ids[ind][op_ind] for ind in range(len(out_label_ids))]
    
        #for op_ind in range(num_op_ind):
        #    new_logits[op_ind] = np.vstack(new_logits[op_ind]).squeeze()
        #    eval_ce_losses_new[op_ind] = np.vstack(eval_ce_losses_new[op_ind]).squeeze()
        #    out_label_ids_new[op_ind] = np.vstack(out_label_ids_new[op_ind]).squeeze()

        eval_ce_losses, logits, out_label_ids = eval_ce_losses_new, new_logits, out_label_ids_new

        label_map = []
        def obtain_map(label, label_map):
            return label_map[label]
    
        diagnosis_label_map = self.config.id2label  ## TODO: Assuming the first op type matches with self.config.label2id which usually is the case
        label_map.append(lambda x:obtain_map(str(x), diagnosis_label_map))
        label_map.append(lambda x:x)  # identity function for MMSE targets

        results = {}
        op_ind = 0
        if 1:            
            pad_token_label_id = CrossEntropyLoss().ignore_index
            out_label_list_per_op = np.concatenate(out_label_ids[op_ind], axis=0)
            out_label_list_per_op = out_label_list_per_op.squeeze()
            logits_per_op = np.concatenate(logits[op_ind], axis=0)
            logits_per_op = logits_per_op.squeeze()
            preds_per_op = np.argmax(logits_per_op, axis=-1)

            out_label_list_per_op = [[label_map[op_ind](i)  for i in out_label_list_per_op]]
            preds_list_per_op = [[label_map[op_ind](i) for i in preds_per_op]]
            if compute_metrics:
                sklearn_metrics = compute_sklearn_metrics(out_label_list_per_op, preds_list_per_op)

            eval_ce_losses_per_op = list(flatten(eval_ce_losses[op_ind]))
            results.update({
                "losses": np.array(eval_ce_losses_per_op),
                "loss": np.mean(eval_ce_losses_per_op),
                "predictions": preds_list_per_op,
                "logits": logits_per_op,
                "true_labels": out_label_list_per_op,
                "true_seg_boundaries": true_seg_boundaries,
                "utt_id": batch_keys,
            })

        op_ind = 1
        results.update({'losses_op2': np.array(out_label_ids[op_ind])})
        results.update({'mean_loss_op2':  np.mean(out_label_ids[op_ind])})
        results.update({'predictions_op2': np.vstack(logits[op_ind]).squeeze()})
        results.update({'true_labels_op2': np.vstack(out_label_ids[op_ind]).squeeze()})

        return results


def predict_batch_in_windows_MultimodalMultiLoss_SequenceClassification(
        batch: Tuple[torch.Tensor],
        model,
        config,
        window_len: Optional[int] = None,
        window_overlap: Optional[int] = None,
        propagate_context: bool = True,
        device: str = 'cpu',
        max_len_text: int = 512,
        max_len_speech: int = None,
):
    if window_overlap is not None:
        raise ValueError("Overlapping windows processing not implemented.")
    else:
        window_overlap = 0

    batch_keys = batch['speech_utt_id']
    assert batch['speech_utt_id'] == batch['text_utt_id']


    if window_len is None:
        windows = [batch]
    else:
        window_shift = window_len - window_overlap

    num_op_ind = 2
    ce_loss = [[] for op_ind in range(num_op_ind)]
    batch_logits = [[] for op_ind in range(num_op_ind)]
    labels = [[] for op_ind in range(num_op_ind)]
    mems = None
    #import pdb; pdb.set_trace()
    with torch.no_grad():
        #for ind,t in enumerate(batch):
        if 1:
            utt_logits = [[] for op_ind in range(num_op_ind)]
            max_len_batch = batch['text_ip'].shape[1] # sequence length 
            # TODO: max_len_batch is set to window_shift
            max_len_batch = window_shift 
            for i in range(0, max_len_batch, window_shift):
                #print(i)
                inputs = {}
                for ip in batch.keys():
                    if not 'utt_id' in ip:
                        if 'text' in ip:
                            if 'labels' in ip:
                                inputs[ip] = [batch[ip][op_ind].contiguous().to(device) for op_ind in range(num_op_ind)]
                            else:
                                inputs[ip] = batch[ip][:, i: i + window_len].contiguous().to(device)
                        else:
                            ## TODO: currently the speech model is ResNet which can handle any sequence length so not shortening for inference
                            if 'labels' in ip:
                                inputs[ip] = [batch[ip][op_ind].contiguous().to(device) for op_ind in range(num_op_ind)]
                            else:
                                inputs[ip] = batch[ip].contiguous().to(device)

                # Compute
                #for ip in inputs.keys():
                #    if not 'labels' in ip:
                #        print(ip, inputs[ip].shape, batch[ip].shape)
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                batch_ce_loss = [tmp_eval_loss[op_ind].detach().cpu().numpy() for op_ind in range(num_op_ind)]
                
                logits = [logits[op_ind].detach().cpu().numpy() for op_ind in range(num_op_ind)]
                for op_ind in range(num_op_ind):
                    utt_logits[op_ind].append(logits[op_ind])

                # Consume the outputs according to a specific model
                try:
                    for op_ind in range(num_op_ind):
                        ce_loss[op_ind].extend(batch_ce_loss[op_ind])
                except:
                    for op_ind in range(num_op_ind):
                        ce_loss[op_ind].append(batch_ce_loss[op_ind])

                try:
                    out_label_ids = [outputs.subsampled_labels[op_ind].detach().cpu().numpy() for op_ind in range(num_op_ind)]
                except:
                    out_label_ids = [inputs['labels'][op_ind].detach().cpu().numpy() for op_ind in range(num_op_ind)]
                for op_ind in range(num_op_ind):
                    labels[op_ind].append(out_label_ids[op_ind])

            for op_ind in range(num_op_ind):
                if len(utt_logits[op_ind]) > 1:
                    batch_logits[op_ind].append([[np.mean(np.concatenate(utt_logits[op_ind], axis=0), axis=0)]])
                else:
                    batch_logits[op_ind].append(utt_logits[op_ind])
           
    for op_ind in range(num_op_ind):
        batch_logits[op_ind] = np.vstack(batch_logits[op_ind])

    # workaround for PyTorch file descriptor leaks:
    # https://github.com/pytorch/pytorch/issues/973
    returns = ce_loss, batch_logits, labels, None, batch_keys
    for t in batch:
        del t
    return returns


