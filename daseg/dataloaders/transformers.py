import warnings
from functools import partial
from itertools import chain
from typing import Iterable, Optional, List

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler, Dataset
from transformers import PreTrainedTokenizer

from daseg import DialogActCorpus, Call
from daseg.utils_ner import InputExample, convert_examples_to_features


def as_windows(call: Call, max_length: int, tokenizer: PreTrainedTokenizer, use_joint_coding: bool) -> Iterable[Call]:
    if not use_joint_coding:
        warnings.warn('Call windows are not available when joint coding is turned off. Some calls will be truncated.')
        return [call]
    window = []
    cur_len = 0
    for segment in call:
        n_segment_tokens = len(list(chain.from_iterable(tokenizer.tokenize(w) for w in segment.text.split())))
        cur_len += n_segment_tokens
        if cur_len > max_length:
            if not window:
                raise ValueError("Max sequence length is too low - a segment longer than this value was found.")
            yield Call(window)
            window = []
            cur_len = n_segment_tokens
        window.append(segment)
    if window:
        yield Call(window)


def to_dataset(
        corpus: DialogActCorpus,
        tokenizer: PreTrainedTokenizer,
        model_type: str,
        labels: Iterable[str],
        max_seq_length: Optional[int] = None,
        windows_if_exceeds_max_length: bool = False,
        use_joint_coding: bool = True,
        use_turns: bool = False
) -> TensorDataset:
    ner_examples = []
    for idx, call in enumerate(corpus.calls):
        if use_turns:
            for turn_idx, (speaker, turn) in enumerate(call.turns):
                words, tags = turn.words_with_tags(use_joint_coding=use_joint_coding, add_turn_token=False)
                ner_examples.append(InputExample(guid=1000 * idx + turn_idx, words=words, labels=tags))
        else:
            if max_seq_length is not None and windows_if_exceeds_max_length:
                call_parts = as_windows(
                    call=call,
                    max_length=max_seq_length,
                    tokenizer=tokenizer,
                    use_joint_coding=use_joint_coding
                )
            else:
                call_parts = [call]
            for call_part in call_parts:
                words, tags = call_part.words_with_tags(add_turn_token=True, use_joint_coding=use_joint_coding)
                ner_examples.append(InputExample(guid=idx, words=words, labels=tags))

    if max_seq_length is None:
        max_seq_length = 99999999999999999

    # determine max seq length
    max_tok_count = 0
    for (ex_index, example) in enumerate(ner_examples):
        tok_count = 0
        for word in example.words:
            word_tokens = tokenizer.tokenize(word)
            tok_count += len(word_tokens)
        max_tok_count = max(max_tok_count, tok_count)

    sep_token_extra = bool(model_type in ["roberta"]),
    special_tokens_count = 3 if sep_token_extra else 2
    max_tok_count += special_tokens_count
    max_seq_length = min(max_tok_count, max_seq_length)
    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    # The following lines are basically a copy-paste of Transformer's NER code
    # TODO: It could be modified to create "ragged" batches for faster CPU inference
    ner_features = convert_examples_to_features(
        examples=ner_examples,
        label_list=labels,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        cls_token_at_end=bool(model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=sep_token_extra,
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=pad_token,
        pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
        pad_token_label_id=CrossEntropyLoss().ignore_index,
    )

    all_input_ids = torch.tensor([f.input_ids for f in ner_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in ner_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in ner_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in ner_features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    dataset.pad_token = pad_token
    return dataset


def to_dataloader(dataset: Dataset, padding_at_start: bool, batch_size: int, train: bool = True) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        sampler=RandomSampler(dataset) if train else SequentialSampler(dataset),
        batch_size=batch_size,
        collate_fn=partial(truncate_padding_collate_fn, padding_at_start=padding_at_start),
        pin_memory=True,
        num_workers=4
    )


def to_transformers_train_dataloader(
        corpus: DialogActCorpus,
        tokenizer: PreTrainedTokenizer,
        model_type: str,
        batch_size: int,
        labels: Iterable[str],
        max_seq_length: Optional[int] = None,
        use_joint_coding: bool = True,
        use_turns: bool = False,
        windows_if_exceeds_max_length: bool = False,
) -> DataLoader:
    dataset = to_dataset(
        corpus=corpus,
        tokenizer=tokenizer,
        model_type=model_type,
        labels=labels,
        max_seq_length=max_seq_length,
        use_joint_coding=use_joint_coding,
        use_turns=use_turns,
        windows_if_exceeds_max_length=windows_if_exceeds_max_length
    )
    return to_dataloader(dataset, batch_size=batch_size, train=True, padding_at_start=model_type == 'xlnet')


def to_transformers_eval_dataloader(
        corpus: DialogActCorpus,
        tokenizer: PreTrainedTokenizer,
        model_type: str,
        batch_size: int,
        labels: Iterable[str],
        max_seq_length: Optional[int] = None,
        use_joint_coding: bool = True,
        use_turns: bool = False
) -> DataLoader:
    """
    Convert the DA dataset into a PyTorch DataLoader for inference.
    :param tokenizer: Transformers pre-trained tokenizer object
    :param max_seq_length: The actual sequence length will be min(max_seq_length, <actual sequence lengths>)
    :param model_type: string describing Transformers model type (e.g. xlnet, xlmroberta, bert, ...)
    :param batch_size: self-explanatory
    :param labels: you might run into problems if this is a subset of larger dataset which doesn't cover every label
        - use this arg to supply the full list of labels
    :return: PyTorch DataLoader
    """
    dataset = to_dataset(
        corpus=corpus,
        tokenizer=tokenizer,
        model_type=model_type,
        labels=labels,
        max_seq_length=max_seq_length,
        use_joint_coding=use_joint_coding,
        use_turns=use_turns
    )
    return to_dataloader(dataset, batch_size=batch_size, train=False, padding_at_start=model_type == 'xlnet')


def truncate_padding_collate_fn(batch: List[List[torch.Tensor]], padding_at_start: bool = False, add_ilen: bool = True, sort: bool = True):
    batch = sorted(batch, key=lambda tensors: (tensors[3] != -100).to(torch.int32).sum(), reverse=True)
    redundant_padding = max(mask.sum() for _, mask, _, _ in batch)
    n_tensors = len(batch[0])
    concat_tensors = (torch.cat([sample[i].unsqueeze(0) for sample in batch]) for i in range(n_tensors))
    if padding_at_start:
        return [t[:, -redundant_padding:] for t in concat_tensors]
    truncated = [t[:, :redundant_padding] for t in concat_tensors]
    if add_ilen:
        # Here we add extra tensor that states the input lens
        truncated.append(
            truncated[1].sum(dim=1)
        )
    return truncated


def pad_list_of_arrays(arrays: List[np.ndarray], value: float) -> List[np.ndarray]:
    max_out_len = max(x.shape[1] for x in arrays)
    return [pad_array(t, target_len=max_out_len, value=value) for t in arrays]


def pad_array(arr: np.ndarray, target_len: int, value: float):
    if arr.shape[1] == target_len:
        return arr
    pad_shape = list(arr.shape)
    pad_shape[1] = target_len - arr.shape[1]
    return np.concatenate([arr, np.ones(pad_shape) * value], axis=1)
