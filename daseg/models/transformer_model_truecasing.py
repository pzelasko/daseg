import json
from copy import deepcopy
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Union
import warnings

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


class TransformerTextModelMultiLossTokenClassification:
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

        # ave_aggregated_losses, aggregated_logits, aggregated_labels, None, batch_keys
        eval_ce_losses, logits, out_label_ids, true_seg_boundaries, batch_keys, originalword2subtokens, last_layer_outputs = zip(*list(maybe_tqdm(
            map(
                partial(
                    predict_batch_in_windows_TextTokenClassification_MultiLoss,
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

        num_op_types = len(self.config.loss_hyperparameters['loss_weights'])
        pad_token_label_id = CrossEntropyLoss().ignore_index

        results = {}
        for op_ind in range(num_op_types):
            new_logits = [logits[ind][op_ind] for ind in range(len(logits))]
            new_last_layer_outputs = [np.squeeze(last_layer_outputs[ind][op_ind]) for ind in range(len(last_layer_outputs))]

            new_eval_ce_losses = [eval_ce_losses[ind][op_ind] for ind in range(len(eval_ce_losses))]
            new_out_label_ids = [out_label_ids[ind][op_ind] for ind in range(len(out_label_ids))]

            new_out_label_ids = pad_list_of_arrays(new_out_label_ids, value=pad_token_label_id)
            new_out_label_ids = np.concatenate(new_out_label_ids, axis=0)

            new_logits = pad_list_of_arrays(new_logits, value=0)
            new_logits = np.concatenate(new_logits, axis=0)
            preds = np.argmax(new_logits, axis=2)

            label_map = {int(k): v for k, v in self.config.label_maps_id2label[op_ind].items()}

            out_label_list: List[List[str]] = [[] for _ in range(new_out_label_ids.shape[0])]
            preds_list: List[List[str]] = [[] for _ in range(new_out_label_ids.shape[0])]

            filtered_new_last_layer_outputs = [] # stores only relevant indices. new_last_layer_outputs contains padding vectors too
            for i in range(new_out_label_ids.shape[0]): # loop over utterances
                active_indices = []
                for j in range(new_out_label_ids.shape[1]): # loop over each word/frame in each sequence
                    if new_out_label_ids[i, j] != pad_token_label_id:
                        out_label_list[i].append(label_map[new_out_label_ids[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])
                        active_indices.append(j)
                filtered_new_last_layer_outputs.append(new_last_layer_outputs[i][active_indices])


            temp_results = {
                "losses": np.array(new_eval_ce_losses),
                "loss": np.mean(new_eval_ce_losses),
                "predictions": preds_list,
                "logits": new_logits,
                "true_labels": out_label_list,
                "true_seg_boundaries": true_seg_boundaries,
                "utt_id": batch_keys,
                "last_layer_outputs": filtered_new_last_layer_outputs,
            }
            temp_results = {i+'_op'+str(op_ind):j for i,j in temp_results.items()}
            temp_results.update({"sklearn_metrics"+'_op'+str(op_ind): compute_sklearn_metrics(out_label_list, preds_list)})
            results.update(temp_results)

        results.update({'originalword2subtokens':originalword2subtokens})
        return results


def predict_batch_in_windows_TextTokenClassification_MultiLoss(
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

    ce_loss, logits, labels, last_layer_outputs = [], [], [], []

    #import pdb; pdb.set_trace()
    num_op_types = len(config.loss_hyperparameters['loss_weights'])
    mems = None
    with torch.no_grad():
        utt_logits = []
        batch_keys = batch['utt_id']
        originalword2subtokens = batch['originalword2subtokens']
        del batch['utt_id']
        del batch['originalword2subtokens']

        max_len_batch = batch['input_ids'].shape[1] # sequence length 
        for i in range(0, max_len_batch, window_shift):
            #print(i)
            inputs = {}
            for ip in model.model_input_keys:
                if ip == 'labels':
                    #import pdb; pdb.set_trace()
                    inputs[ip] = [batch[ip][op_ind][:, i: i + window_len].contiguous().to(device) 
                                            for op_ind in range(num_op_types)]

                elif not 'utt_id' in ip:
                    inputs[ip] = batch[ip][:, i: i + window_len].contiguous().to(device)

            if config.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    inputs['token_type_ids'] if config.model_type in ["bert", "xlnet"] else None
                ) # XLM and RoBERTa don't use segment_ids

            #if use_xlnet_memory:
            #    inputs['mems'] = mems
            # Compute
            outputs = model(**inputs)

            batch_ce_loss = outputs.loss
            batch_ce_loss = [temp.detach().cpu().numpy() for temp in batch_ce_loss]
            ce_loss.append(batch_ce_loss)

            batch_logits = outputs.logits
            batch_logits = [temp.detach().cpu().numpy() for temp in batch_logits]
            logits.append(batch_logits)

            out_label_ids = [temp.detach().cpu().numpy() for temp in inputs['labels']]
            labels.append(out_label_ids)

            batch_last_layer_outputs = outputs.last_layer_output
            batch_last_layer_outputs = [temp.detach().cpu().numpy() for temp in batch_last_layer_outputs]
            last_layer_outputs.append(batch_last_layer_outputs)        

            if use_xlnet_memory:
                mems = outputs[2]
    # workaround for PyTorch file descriptor leaks:
    # https://github.com/pytorch/pytorch/issues/973

    ############ need to aggregate logits for each output. FOr each forward pass you get op like this [logits1, logits2] for corresponding label sets #######################
    aggregated_logits = []
    aggregated_labels = []
    aggregated_losses = []
    aggregated_last_layer_outputs = []
    for op_ind in range(num_op_types):
        aggregated_logits.append([])
        aggregated_labels.append([])
        aggregated_losses.append([])
        aggregated_last_layer_outputs.append([])
        for forward_pass_ind in range(len(logits)):
            aggregated_logits[op_ind] += [logits[forward_pass_ind][op_ind]]
            aggregated_labels[op_ind] += [labels[forward_pass_ind][op_ind]]
            aggregated_losses[op_ind] += [ce_loss[forward_pass_ind][op_ind]]
            aggregated_last_layer_outputs[op_ind] += [last_layer_outputs[forward_pass_ind][op_ind]]

    ave_aggregated_losses = []
    for op_ind in range(num_op_types):    
        ## assumes only one utt in batch. Here we take average of segment losses
        ave_aggregated_losses.append(np.mean(aggregated_losses[0]))

    
    aggregated_logits = [np.concatenate(temp, axis=1) for temp in aggregated_logits]
    aggregated_labels = [np.concatenate(temp, axis=1) for temp in aggregated_labels]
    aggregated_last_layer_outputs = [np.concatenate(temp, axis=1) for temp in aggregated_last_layer_outputs]

    returns = ave_aggregated_losses, aggregated_logits, aggregated_labels, None, batch_keys, originalword2subtokens, aggregated_last_layer_outputs
    #returns = ce_loss, np.concatenate(logits, axis=1), np.concatenate(labels, axis=1), deepcopy(batch['seg_boundaries'].detach().cpu().numpy()), batch_keys 
    for t in batch:
        del t
    return returns


class TransformerTextModelMultiLossSeqClassificationTopicSeg:
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

        # ave_aggregated_losses, aggregated_logits, aggregated_labels, None, batch_keys
        eval_ce_losses, logits, out_label_ids, true_seg_boundaries, batch_keys, originalword2subtokens, last_layer_outputs = zip(*list(maybe_tqdm(
            map(
                partial(
                    predict_batch_in_windows_TextTokenClassification_MultiLoss_TopicSeg,
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

        num_op_types = len(self.config.loss_hyperparameters['loss_weights'])
        pad_token_label_id = CrossEntropyLoss().ignore_index

        results = {}
        for op_ind in range(num_op_types):
            new_logits = [logits[ind][op_ind] for ind in range(len(logits))]
            new_last_layer_outputs = [np.squeeze(last_layer_outputs[ind][op_ind]) for ind in range(len(last_layer_outputs))]

            new_eval_ce_losses = [eval_ce_losses[ind][op_ind] for ind in range(len(eval_ce_losses))]
            new_out_label_ids = [out_label_ids[ind][op_ind] for ind in range(len(out_label_ids))]

            new_out_label_ids = pad_list_of_arrays(new_out_label_ids, value=pad_token_label_id)
            new_out_label_ids = np.concatenate(new_out_label_ids, axis=0)

            new_logits = pad_list_of_arrays(new_logits, value=1)
            new_logits = np.concatenate(new_logits, axis=0)
            preds = np.argmax(new_logits, axis=2)

            label_map = {int(k): v for k, v in self.config.label_maps_id2label[op_ind].items()}

            out_label_list: List[List[str]] = [[] for _ in range(new_out_label_ids.shape[0])]
            preds_list: List[List[str]] = [[] for _ in range(new_out_label_ids.shape[0])]

            filtered_new_last_layer_outputs = [] # stores only relevant indices. new_last_layer_outputs contains padding vectors too
            for i in range(new_out_label_ids.shape[0]): # loop over utterances
                active_indices = []
                for j in range(new_out_label_ids.shape[1]): # loop over each word/frame in each sequence
                    if new_out_label_ids[i, j] != pad_token_label_id:
                        out_label_list[i].append(label_map[new_out_label_ids[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])
                        active_indices.append(j)
                filtered_new_last_layer_outputs.append(new_last_layer_outputs[i][active_indices])


            temp_results = {
                "losses": np.array(new_eval_ce_losses),
                "loss": np.mean(new_eval_ce_losses),
                "predictions": preds_list,
                "logits": new_logits,
                "true_labels": out_label_list,
                "true_seg_boundaries": true_seg_boundaries,
                "utt_id": batch_keys,
                "last_layer_outputs": filtered_new_last_layer_outputs,
            }
            temp_results = {i+'_op'+str(op_ind):j for i,j in temp_results.items()}
            temp_results.update({"sklearn_metrics"+'_op'+str(op_ind): compute_sklearn_metrics(out_label_list, preds_list)})
            results.update(temp_results)

        results.update({'originalword2subtokens':originalword2subtokens})
        return results


def predict_batch_in_windows_TextTokenClassification_MultiLoss_TopicSeg(
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

    ce_loss, logits, labels, last_layer_outputs = [], [], [], []

    warnings.warn("Warning........ Set window_shift")
    window_shift = 50

    #import pdb; pdb.set_trace()
    num_op_types = len(config.loss_hyperparameters['loss_weights'])
    mems = None
    with torch.no_grad():
        utt_logits = []
        batch_keys = batch['utt_id']
        originalword2subtokens = batch['originalword2subtokens']
        del batch['utt_id']
        del batch['originalword2subtokens']

        max_len_batch = batch['input_ids'].shape[1] # sequence length 
        for i in range(0, max_len_batch, window_shift):
            #print(i)
            inputs = {}
            for ip in model.model_input_keys:
                if ip == 'labels':
                    #import pdb; pdb.set_trace()
                    inputs[ip] = [batch[ip][op_ind][:, i: i + window_len].contiguous().to(device) 
                                            for op_ind in range(num_op_types)]
                    inputs[ip] = [map_batch_labels_for_TopicSegSeqLevelClassif(inputs[ip][op_ind], config.label_maps_id2label[op_ind]) 
                                            for op_ind in range(num_op_types)]
                    #import pdb; pdb.set_trace()

                elif not 'utt_id' in ip:
                    inputs[ip] = batch[ip][:, i: i + window_len].contiguous().to(device)

            if config.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    inputs['token_type_ids'] if config.model_type in ["bert", "xlnet"] else None
                ) # XLM and RoBERTa don't use segment_ids

            #if use_xlnet_memory:
            #    inputs['mems'] = mems
            # Compute
            outputs = model(**inputs)

            batch_ce_loss = outputs.loss
            batch_ce_loss = [temp.detach().cpu().numpy() for temp in batch_ce_loss]
            ce_loss.append(batch_ce_loss)

            batch_logits = outputs.logits
            batch_logits = [[temp.detach().cpu().numpy()] for temp in batch_logits]
            logits.append(batch_logits)

            out_label_ids = [[temp.detach().cpu().numpy()] for temp in inputs['labels']]
            labels.append(out_label_ids)

            batch_last_layer_outputs = outputs.last_layer_output
            batch_last_layer_outputs = [temp.detach().cpu().numpy() for temp in batch_last_layer_outputs]
            last_layer_outputs.append(batch_last_layer_outputs)        

            if use_xlnet_memory:
                mems = outputs[2]
    # workaround for PyTorch file descriptor leaks:
    # https://github.com/pytorch/pytorch/issues/973

    ############ need to aggregate logits for each output. FOr each forward pass you get op like this [logits1, logits2] for corresponding label sets #######################
    aggregated_logits = []
    aggregated_labels = []
    aggregated_losses = []
    aggregated_last_layer_outputs = []
    for op_ind in range(num_op_types):
        aggregated_logits.append([])
        aggregated_labels.append([])
        aggregated_losses.append([])
        aggregated_last_layer_outputs.append([])
        for forward_pass_ind in range(len(logits)):
            aggregated_logits[op_ind] += [logits[forward_pass_ind][op_ind]]
            aggregated_labels[op_ind] += [labels[forward_pass_ind][op_ind]]
            aggregated_losses[op_ind] += [ce_loss[forward_pass_ind][op_ind]]
            aggregated_last_layer_outputs[op_ind] += [last_layer_outputs[forward_pass_ind][op_ind]]

    ave_aggregated_losses = []
    for op_ind in range(num_op_types):    
        ## assumes only one utt in batch. Here we take average of segment losses
        ave_aggregated_losses.append(np.mean(aggregated_losses[0]))

    aggregated_logits = [np.concatenate(temp, axis=1) for temp in aggregated_logits]
    aggregated_labels = [np.concatenate(temp, axis=1) for temp in aggregated_labels]
    aggregated_last_layer_outputs = [np.concatenate(temp, axis=1) for temp in aggregated_last_layer_outputs]

    returns = ave_aggregated_losses, aggregated_logits, aggregated_labels, None, batch_keys, originalword2subtokens, aggregated_last_layer_outputs
    #returns = ce_loss, np.concatenate(logits, axis=1), np.concatenate(labels, axis=1), deepcopy(batch['seg_boundaries'].detach().cpu().numpy()), batch_keys 
    for t in batch:
        del t
    return returns


def map_batch_labels_for_TopicSegSeqLevelClassif(batch_label, label_map):
    '''Since we pass a subsegment each time thorugh BERT model with a target of either boundary or not, we need to map toke-level labels to segment level labels.
       This code basically assigns only one label for the sequence of labels input
    '''
    label_map = {int(k): v for k, v in label_map.items()}
    boundary_class_ind = [class_ind for class_ind,class_name in label_map.items() if class_name == 'Boundary']
    boundary_class_ind = boundary_class_ind[0]
    non_boundary_class_ind = [class_ind for class_ind,class_name in label_map.items() if class_name != 'Boundary']
    if len(non_boundary_class_ind) != 1:
        #import pdb; pdb.set_trace()
        raise ValueError(f'number of non-boundary clases are more than one so there could be an error in data labels')
    else:
        non_boundary_class_ind = non_boundary_class_ind[0]

    new_batch_label = []
    for utt_label in batch_label:
        if boundary_class_ind in utt_label:
            new_batch_label.append(boundary_class_ind)
        else:
            new_batch_label.append(non_boundary_class_ind)

    #import pdb; pdb.set_trace()
    new_batch_label = torch.Tensor(new_batch_label)
    new_batch_label = new_batch_label.type(torch.LongTensor)
    new_batch_label = new_batch_label.to(device=batch_label.device)
    return new_batch_label


