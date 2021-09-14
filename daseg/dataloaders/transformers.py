import warnings
from functools import partial
from itertools import chain
from typing import Iterable, Optional, List
from pathlib import Path

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler, Dataset
from transformers import PreTrainedTokenizer

from daseg import DialogActCorpus, Call
from daseg.utils_ner import InputExample, convert_examples_to_features
import pandas as pd

def as_windows(call: Call, max_length: int, tokenizer: PreTrainedTokenizer, use_joint_coding: bool) -> Iterable[Call]:
    '''It's a generator that yields segments of length max_length from the input transcript
    '''
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
         

def to_speech_dataloader(
        data_dir: list,
        batch_size: int,
        max_sequence_length: int,
        frame_len: float,
        target_label_encoder,
        train_mode: str,
        segmentation_type: str,
        concat_aug: int, 
        test_file: str):

    from daseg.dataloader_speech import collate_fn
    from daseg.dataloader_speech import get_dataset, ConcatDataset_SidebySide, ConcatDataset_SidebySide_EqualizingLength

    padding_value_features = 0
    mask_padding_with_zero = 1
    padding_value_mask = 0 if mask_padding_with_zero else 1
    
    if len(data_dir) > 1:
        print(f'multiple data_dir is not supported for tasks other than EmoSpot')
        sys.exit()
    else:
        data_dir = data_dir[0]

    data_loaders = {}
    feat_dim = None
    if (train_mode == 'TE') or (train_mode == 'T'):
        train_cfn = partial(collate_fn, max_len=max_sequence_length, split='train',
                                target_label_encoder=target_label_encoder,
                                concat_aug=concat_aug, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len, 
                                segmentation_type=segmentation_type)
        dev_cfn = partial(collate_fn, max_len=max_sequence_length, split='dev',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type)
        ## while getting the datasets, it's important to put max_len=None for atleast frame-based classification
        ## because you make the labels in the collate_fn
        train_ds = get_dataset(data_dir, data_csv=data_dir+'/' + 'train' + '.tsv', max_len=None)
        train_ds = [train_ds]
        train_ds = ConcatDataset_SidebySide(*train_ds)

        dev_ds = get_dataset(data_dir, data_csv=data_dir+'/' + 'dev' + '.tsv', max_len=None)
        print(f'train dataset length is {len(train_ds)}')
        print(f'dev dataset length is {len(dev_ds)}')

        train_sampler = RandomSampler(train_ds)
        dev_sampler = SequentialSampler(dev_ds)
        train_dataloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size, collate_fn=train_cfn, drop_last=True)
        dev_dataloader = DataLoader(dev_ds, sampler=dev_sampler, batch_size=1, collate_fn=dev_cfn)
        data_loaders['train'] = train_dataloader
        data_loaders['dev'] = dev_dataloader
        ## get feat_dim
        batch = next(iter(train_dataloader))
        feat_dim = batch[0].shape[-1] 

    if (train_mode == 'TE') or (train_mode == 'E'):
        test_cfn = partial(collate_fn, max_len=None, split='test',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type)
        if test_file == 'test.tsv':
            test_file = data_dir+ '/' + test_file
        print(f'loading {test_file} for evaluating the model')
        test_ds = get_dataset(data_dir, data_csv=test_file, max_len=None)
        test_sampler = SequentialSampler(test_ds)
        test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=1, collate_fn=test_cfn)
    
        data_loaders['test'] = test_dataloader
        print(f'test dataset length is {len(test_ds)}')
        ## get feat_dim
        batch = next(iter(test_dataloader))
        feat_dim_test = batch[0].shape[-1] 
        if feat_dim is not None:
            assert feat_dim_test == feat_dim
        else:
            feat_dim = feat_dim_test

    for split in ['train', 'dev', 'test']:
        if not split in data_loaders:
            data_loaders[split] = None

        else:
            print(f'\n printing few samples from the split {split} for debugging purposes \n')
            for step,data in enumerate(data_loaders[split]):
                print(data[0].shape, data[1].shape, data[2].shape)
                print(data)
                if step > 3:
                    break


    #import pdb; pdb.set_trace()
    #print('train')
    #for step,data in enumerate(train_dataloader):
    #    print(data[0].shape, data[1].shape, data[2].shape)
    #    #break
    #print('dev')
    #for step,data in enumerate(dev_dataloader):
    #    print(data[0].shape, data[1].shape, data[2].shape)
    #    #break
    #print('test')
    #for step,data in enumerate(test_dataloader):
    #    print(data[0].shape, data[1].shape, data[2].shape)
    #    break
 
    return data_loaders, feat_dim


def to_speech_dataloader_SeqClassification(
        data_dir: list,
        batch_size: int,
        max_sequence_length: int,
        frame_len: float,
        target_label_encoder,
        train_mode: str,
        segmentation_type: str,
        concat_aug: int, 
        test_file: str):
        
    from daseg.dataloader_speech_SeqClassification import collate_fn_SeqClassification as collate_fn
    from daseg.dataloader_speech_SeqClassification import get_dataset, ConcatDataset_SidebySide, ConcatDataset_SidebySide_EqualizingLength

    padding_value_features = 0
    mask_padding_with_zero = 1
    padding_value_mask = 0 if mask_padding_with_zero else 1
    
    if len(data_dir) > 1:
        print(f'multiple data_dir is not supported for tasks other than EmoSpot')
        sys.exit()
    else:
        data_dir = data_dir[0]

    data_loaders = {}
    feat_dim = None
    if (train_mode == 'TE') or (train_mode == 'T'):
        train_cfn = partial(collate_fn, max_len=max_sequence_length, split='train',
                                target_label_encoder=target_label_encoder,
                                concat_aug=concat_aug, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len, 
                                segmentation_type=segmentation_type)
        dev_cfn = partial(collate_fn, max_len=max_sequence_length, split='dev',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type)
        ## while getting the datasets, it's important to put max_len=None for atleast frame-based classification
        ## because you make the labels in the collate_fn
        train_ds = get_dataset(data_dir, data_csv=data_dir+'/' + 'train' + '.tsv', max_len=None)
        train_ds = [train_ds]
        train_ds = ConcatDataset_SidebySide(*train_ds)

        dev_ds = get_dataset(data_dir, data_csv=data_dir+'/' + 'dev' + '.tsv', max_len=None)
        print(f'train dataset length is {len(train_ds)}')
        print(f'dev dataset length is {len(dev_ds)}')

        train_sampler = RandomSampler(train_ds)
        dev_sampler = SequentialSampler(dev_ds)
        train_dataloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size, collate_fn=train_cfn, drop_last=True)
        dev_dataloader = DataLoader(dev_ds, sampler=dev_sampler, batch_size=1, collate_fn=dev_cfn)
        data_loaders['train'] = train_dataloader
        data_loaders['dev'] = dev_dataloader
        ## get feat_dim
        batch = next(iter(train_dataloader))
        feat_dim = batch[0].shape[-1] 

    if (train_mode == 'TE') or (train_mode == 'E'):
        test_cfn = partial(collate_fn, max_len=None, split='test',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type)
        if test_file == 'test.tsv':
            test_file = data_dir+ '/' + test_file
        print(f'loading {test_file} for evaluating the model')
        test_ds = get_dataset(data_dir, data_csv=test_file, max_len=None)
        test_sampler = SequentialSampler(test_ds)
        test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=1, collate_fn=test_cfn)
    
        data_loaders['test'] = test_dataloader
        print(f'test dataset length is {len(test_ds)}')
        ## get feat_dim
        batch = next(iter(test_dataloader))
        feat_dim_test = batch[0].shape[-1] 
        if feat_dim is not None:
            assert feat_dim_test == feat_dim
        else:
            feat_dim = feat_dim_test

    for split in ['train', 'dev', 'test']:
        if not split in data_loaders:
            data_loaders[split] = None
        

    #import pdb; pdb.set_trace()
    #print('train')
    #for step,data in enumerate(train_dataloader):
    #    print(data[0].shape, data[1].shape, data[2].shape)
    #    #break
    #print('dev')
    #for step,data in enumerate(dev_dataloader):
    #    print(data[0].shape, data[1].shape, data[2].shape)
    #    #break
    #print('test')
    #for step,data in enumerate(test_dataloader):
    #    print(data[0].shape, data[1].shape, data[2].shape)
    #    break
 
    return data_loaders, feat_dim

def EmoSpot_speech_dataloader(
        data_dir: list,
        batch_size: int,
        max_sequence_length: int,
        frame_len: float,
        target_label_encoder,
        train_mode: str,
        segmentation_type: str,
        concat_aug: int):

    from daseg.dataloader_speech import collate_fn
    from daseg.dataloader_speech import get_dataset, ConcatDataset_SidebySide, collate_fn_EmoSpot, ConcatDataset_SidebySide_EqualizingLength

   
    padding_value_features = 0
    mask_padding_with_zero = 1
    padding_value_mask = 0 if mask_padding_with_zero else 1
         
    data_loaders = {}
    feat_dim = None
    if (train_mode == 'TE') or (train_mode == 'T'):
        train_cfn = partial(collate_fn_EmoSpot, max_len=max_sequence_length, split='train',
                                target_label_encoder=target_label_encoder,
                                concat_aug=concat_aug, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len, 
                                segmentation_type=segmentation_type)
        dev_cfn = partial(collate_fn, max_len=max_sequence_length, split='dev',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type)
        ## while getting the datasets, it's important to put max_len=None for atleast frame-based classification
        ## because you make the labels in the collate_fn

        train_ds = []
        for data_dir_temp in data_dir:
            train_ds.append(get_dataset(data_dir_temp, data_csv=data_dir_temp+'/' + 'train' + '.tsv', max_len=None))

        train_ds = ConcatDataset_SidebySide_EqualizingLength(*train_ds)

        data_dir_temp = data_dir[0]
        dev_ds = get_dataset(data_dir_temp, data_csv=data_dir_temp+'/' + 'dev' + '.tsv', max_len=None)
        print(f'train dataset length is {len(train_ds)}')
        print(f'dev dataset length is {len(dev_ds)}')

        train_sampler = RandomSampler(train_ds)
        dev_sampler = SequentialSampler(dev_ds)
        train_dataloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size, collate_fn=train_cfn, drop_last=True)
        dev_dataloader = DataLoader(dev_ds, sampler=dev_sampler, batch_size=1, collate_fn=dev_cfn)
        data_loaders['train'] = train_dataloader
        data_loaders['dev'] = dev_dataloader
        ## get feat_dim
        batch = next(iter(train_dataloader))
        feat_dim = batch[0].shape[-1] 

    if (train_mode == 'TE') or (train_mode == 'E'):
        test_cfn = partial(collate_fn, max_len=None, split='test',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type)
        data_dir_temp = data_dir[0]
        test_ds = get_dataset(data_dir_temp, data_csv=data_dir_temp+'/' + 'test' + '.tsv', max_len=None)
        test_sampler = SequentialSampler(test_ds)
        test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=1, collate_fn=test_cfn)
    
        data_loaders['test'] = test_dataloader
        print(f'test dataset length is {len(test_ds)}')
        ## get feat_dim
        batch = next(iter(test_dataloader))
        feat_dim_test = batch[0].shape[-1] 
        if feat_dim is not None:
            assert feat_dim_test == feat_dim
        else:
            feat_dim = feat_dim_test


    for split in ['train', 'dev', 'test']:
        if not split in data_loaders:
            data_loaders[split] = None

    batch = next(iter(train_dataloader))
    feat_dim = batch[0].shape[-1] 

    #import pdb; pdb.set_trace()
    #for step,data in enumerate(train_dataloader):
    #    #print(data[0].shape, data[1].shape, data[2].shape)
    #    print(step)
    #    #break
    #import pdb; pdb.set_trace()
    #for step,data in enumerate(dev_dataloader):
    #    print(data[0].shape, data[1].shape, data[2].shape)
    #    #break
  
    return data_loaders, feat_dim


def to_text_dataloader(
        data_dir: list,
        batch_size: int,
        max_sequence_length: int,
        frame_len: float,
        target_label_encoder,
        train_mode: str,
        segmentation_type: str,
        concat_aug: int, 
        test_file: str, 
        tokenizer):

    from daseg.dataloader_text import ConcatDataset_SidebySide, ConcatDataset_SidebySide_EqualizingLength
    from daseg.dataloader_text import collate_fn_text, get_dataset_text

    padding_value_features = 0
    mask_padding_with_zero = 1
    padding_value_mask = 0 if mask_padding_with_zero else 1
    
    if len(data_dir) > 1:
        print(f'multiple data_dir is not supported for tasks other than EmoSpot')
        sys.exit()
    else:
        data_dir = data_dir[0]

    data_loaders = {}
    feat_dim = None
    if (train_mode == 'TE') or (train_mode == 'T'):
        train_cfn = partial(collate_fn_text, max_len=max_sequence_length, split='train',
                                target_label_encoder=target_label_encoder,
                                concat_aug=concat_aug, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len, 
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        dev_cfn = partial(collate_fn_text, max_len=max_sequence_length, split='dev',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        ## while getting the datasets, it's important to put max_len=None for atleast frame-based classification
        ## because you make the labels in the collate_fn
        train_ds = get_dataset_text(data_dir, data_csv=data_dir+'/' + 'train' + '.tsv', max_len=None)
        train_ds = [train_ds]
        train_ds = ConcatDataset_SidebySide(*train_ds)

        dev_ds = get_dataset_text(data_dir, data_csv=data_dir+'/' + 'dev' + '.tsv', max_len=None)
        print(f'train dataset length is {len(train_ds)}')
        print(f'dev dataset length is {len(dev_ds)}')

        train_sampler = RandomSampler(train_ds)
        dev_sampler = SequentialSampler(dev_ds)
        train_dataloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size, collate_fn=train_cfn, drop_last=True)
        dev_dataloader = DataLoader(dev_ds, sampler=dev_sampler, batch_size=1, collate_fn=dev_cfn)
        data_loaders['train'] = train_dataloader
        data_loaders['dev'] = dev_dataloader
        ## get feat_dim
        feat_dim = None
        #batch = next(iter(train_dataloader))
        #feat_dim = batch[0].shape[-1] 

    if (train_mode == 'TE') or (train_mode == 'E'):
        test_cfn = partial(collate_fn_text, max_len=None, split='test',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        if test_file == 'test.tsv':
            test_file = data_dir+ '/' + test_file
        print(f'loading {test_file} for evaluating the model')
        test_ds = get_dataset_text(data_dir, data_csv=test_file, max_len=None)
        test_sampler = SequentialSampler(test_ds)
        test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=1, collate_fn=test_cfn)
    
        data_loaders['test'] = test_dataloader
        print(f'test dataset length is {len(test_ds)}')
        ## get feat_dim
        batch = next(iter(test_dataloader))
        feat_dim_test = batch[0].shape[-1] 
        feat_dim = None
        #if feat_dim is not None:
        #    assert feat_dim_test == feat_dim
        #else:
        #    feat_dim = feat_dim_test

    #import pdb; pdb.set_trace()
    for split in ['train', 'dev', 'test']:
        if not split in data_loaders:
            data_loaders[split] = None
        
        else:
            print(f'\n printing few samples from the split {split} for debugging purposes \n')
            for step,data in enumerate(data_loaders[split]):
                print(data[0].shape, data[1].shape, data[2].shape)
                print(data)
                if step > 3:
                    break

    return data_loaders, feat_dim


def to_text_TrueCasingTokenClassif_dataloader(
        data_dir: list,
        batch_size: int,
        max_sequence_length: int,
        frame_len: float,
        target_label_encoder,
        train_mode: str,
        segmentation_type: str,
        concat_aug: int, 
        test_file: str, 
        tokenizer):

    from daseg.dataloader_text_TrueCasing import ConcatDataset_SidebySide, ConcatDataset_SidebySide_EqualizingLength, collate_fn_text_TrueCasing, get_dataset_text_TrueCasing

    padding_value_features = 0
    mask_padding_with_zero = 1
    padding_value_mask = 0 if mask_padding_with_zero else 1
    
    if len(data_dir) > 1:
        print(f'multiple data_dir is not supported for tasks other than EmoSpot')
        sys.exit()
    else:
        data_dir = data_dir[0]

    data_loaders = {}
    feat_dim = None
    if (train_mode == 'TE') or (train_mode == 'T'):
        train_cfn = partial(collate_fn_text_TrueCasing, max_len=max_sequence_length, split='train',
                                target_label_encoder=target_label_encoder,
                                concat_aug=concat_aug, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len, 
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        dev_cfn = partial(collate_fn_text_TrueCasing, max_len=max_sequence_length, split='dev',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        ## while getting the datasets, it's important to put max_len=None for atleast frame-based classification
        ## because you make the labels in the collate_fn
        train_ds = get_dataset_text_TrueCasing(data_dir, data_csv=data_dir+'/' + 'train' + '.tsv', max_len=None)
        train_ds = [train_ds]
        train_ds = ConcatDataset_SidebySide(*train_ds)

        dev_ds = get_dataset_text_TrueCasing(data_dir, data_csv=data_dir+'/' + 'dev' + '.tsv', max_len=None)
        print(f'train dataset length is {len(train_ds)}')
        print(f'dev dataset length is {len(dev_ds)}')

        train_sampler = RandomSampler(train_ds)
        dev_sampler = SequentialSampler(dev_ds)
        train_dataloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size, collate_fn=train_cfn, drop_last=True)
        dev_dataloader = DataLoader(dev_ds, sampler=dev_sampler, batch_size=1, collate_fn=dev_cfn)
        data_loaders['train'] = train_dataloader
        data_loaders['dev'] = dev_dataloader
        ## get feat_dim
        feat_dim = None
        #batch = next(iter(train_dataloader))
        #feat_dim = batch[0].shape[-1] 

    if (train_mode == 'TE') or (train_mode == 'E'):
        test_cfn = partial(collate_fn_text_TrueCasing, max_len=None, split='test',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        if test_file == 'test.tsv':
            test_file = data_dir+ '/' + test_file
        print(f'loading {test_file} for evaluating the model')
        test_ds = get_dataset_text_TrueCasing(data_dir, data_csv=test_file, max_len=None)
        test_sampler = SequentialSampler(test_ds)
        test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=1, collate_fn=test_cfn)
    
        data_loaders['test'] = test_dataloader
        print(f'test dataset length is {len(test_ds)}')
        ## get feat_dim
        batch = next(iter(test_dataloader))
        feat_dim = None

    for split in ['train', 'dev', 'test']:
        if not split in data_loaders:
            data_loaders[split] = None
        
        else:
            pass
            print(f'\n printing few samples from the split {split} for debugging purposes \n')
            for step,data in enumerate(data_loaders[split]):
                print(data)
                if step > 3:
                    break

    return data_loaders, feat_dim


def to_text_PunctuationTokenClassif_dataloader(
        data_dir: list,
        batch_size: int,
        max_sequence_length: int,
        frame_len: float,
        target_label_encoder,
        train_mode: str,
        segmentation_type: str,
        concat_aug: int, 
        test_file: str, 
        tokenizer):

    from daseg.dataloader_text_TrueCasing import ConcatDataset_SidebySide, ConcatDataset_SidebySide_EqualizingLength, collate_fn_text_Punctuation, get_dataset_text_TrueCasing

    padding_value_features = 0
    mask_padding_with_zero = 1
    padding_value_mask = 0 if mask_padding_with_zero else 1
    
    if len(data_dir) > 1:
        print(f'multiple data_dir is not supported for tasks other than EmoSpot')
        sys.exit()
    else:
        data_dir = data_dir[0]

    data_loaders = {}
    feat_dim = None
    if (train_mode == 'TE') or (train_mode == 'T'):
        train_cfn = partial(collate_fn_text_Punctuation, max_len=max_sequence_length, split='train',
                                target_label_encoder=target_label_encoder,
                                concat_aug=concat_aug, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len, 
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        dev_cfn = partial(collate_fn_text_Punctuation, max_len=max_sequence_length, split='dev',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        ## while getting the datasets, it's important to put max_len=None for atleast frame-based classification
        ## because you make the labels in the collate_fn
        train_ds = get_dataset_text_TrueCasing(data_dir, data_csv=data_dir+'/' + 'train' + '.tsv', max_len=None)
        train_ds = [train_ds]
        train_ds = ConcatDataset_SidebySide(*train_ds)

        dev_ds = get_dataset_text_TrueCasing(data_dir, data_csv=data_dir+'/' + 'dev' + '.tsv', max_len=None)
        print(f'train dataset length is {len(train_ds)}')
        print(f'dev dataset length is {len(dev_ds)}')

        train_sampler = RandomSampler(train_ds)
        dev_sampler = SequentialSampler(dev_ds)
        train_dataloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size, collate_fn=train_cfn, drop_last=True)
        dev_dataloader = DataLoader(dev_ds, sampler=dev_sampler, batch_size=1, collate_fn=dev_cfn)
        data_loaders['train'] = train_dataloader
        data_loaders['dev'] = dev_dataloader
        ## get feat_dim
        feat_dim = None
        #batch = next(iter(train_dataloader))
        #feat_dim = batch[0].shape[-1] 

    if (train_mode == 'TE') or (train_mode == 'E'):
        test_cfn = partial(collate_fn_text_Punctuation, max_len=None, split='test',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        if test_file == 'test.tsv':
            test_file = data_dir+ '/' + test_file
        print(f'loading {test_file} for evaluating the model')
        test_ds = get_dataset_text_TrueCasing(data_dir, data_csv=test_file, max_len=None)
        test_sampler = SequentialSampler(test_ds)
        test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=1, collate_fn=test_cfn)
    
        data_loaders['test'] = test_dataloader
        print(f'test dataset length is {len(test_ds)}')
        ## get feat_dim
        batch = next(iter(test_dataloader))
        feat_dim = None

    for split in ['train', 'dev', 'test']:
        if not split in data_loaders:
            data_loaders[split] = None
        
        else:
            pass
            print(f'\n printing few samples from the split {split} for debugging purposes \n')
            for step,data in enumerate(data_loaders[split]):
                print(data)
                if step > 3:
                    break

    return data_loaders, feat_dim



def to_text_TrueCasingPunctuationTokenClassif_dataloader(
        data_dir: list,
        batch_size: int,
        max_sequence_length: int,
        frame_len: float,
        target_label_encoder,
        train_mode: str,
        segmentation_type: str,
        concat_aug: int, 
        test_file: str, 
        tokenizer):

    from daseg.dataloader_text_TrueCasingPunctuation import ConcatDataset_SidebySide, ConcatDataset_SidebySide_EqualizingLength, collate_fn_text_TrueCasingPunctuation, get_dataset_text_TrueCasing

    padding_value_features = 0
    mask_padding_with_zero = 1
    padding_value_mask = 0 if mask_padding_with_zero else 1
    
    if len(data_dir) > 1:
        print(f'multiple data_dir is not supported for tasks other than EmoSpot')
        sys.exit()
    else:
        data_dir = data_dir[0]

    data_loaders = {}
    feat_dim = None
    if (train_mode == 'TE') or (train_mode == 'T'):
        train_cfn = partial(collate_fn_text_TrueCasingPunctuation, max_len=max_sequence_length, split='train',
                                target_label_encoder=target_label_encoder,
                                concat_aug=concat_aug, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len, 
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        dev_cfn = partial(collate_fn_text_TrueCasingPunctuation, max_len=max_sequence_length, split='dev',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        ## while getting the datasets, it's important to put max_len=None for atleast frame-based classification
        ## because you make the labels in the collate_fn
        train_ds = get_dataset_text_TrueCasing(data_dir, data_csv=data_dir+'/' + 'train' + '.tsv', max_len=None)
        train_ds = [train_ds]
        train_ds = ConcatDataset_SidebySide(*train_ds)

        dev_ds = get_dataset_text_TrueCasing(data_dir, data_csv=data_dir+'/' + 'dev' + '.tsv', max_len=None)
        print(f'train dataset length is {len(train_ds)}')
        print(f'dev dataset length is {len(dev_ds)}')

        train_sampler = RandomSampler(train_ds)
        dev_sampler = SequentialSampler(dev_ds)
        train_dataloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size, collate_fn=train_cfn, drop_last=True)
        dev_dataloader = DataLoader(dev_ds, sampler=dev_sampler, batch_size=1, collate_fn=dev_cfn)
        data_loaders['train'] = train_dataloader
        data_loaders['dev'] = dev_dataloader
        ## get feat_dim
        feat_dim = None
        #batch = next(iter(train_dataloader))
        #feat_dim = batch[0].shape[-1] 

    if (train_mode == 'TE') or (train_mode == 'E'):
        test_cfn = partial(collate_fn_text_TrueCasingPunctuation, max_len=None, split='test',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        if test_file == 'test.tsv':
            test_file = data_dir+ '/' + test_file
        print(f'loading {test_file} for evaluating the model')
        test_ds = get_dataset_text_TrueCasing(data_dir, data_csv=test_file, max_len=None)
        test_sampler = SequentialSampler(test_ds)
        test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=1, collate_fn=test_cfn)
    
        data_loaders['test'] = test_dataloader
        print(f'test dataset length is {len(test_ds)}')
        ## get feat_dim
        batch = next(iter(test_dataloader))
        feat_dim = None

    for split in ['train', 'dev', 'test']:
        if not split in data_loaders:
            data_loaders[split] = None
        
        else:
            pass
            print(f'\n printing few samples from the split {split} for debugging purposes \n')
            for step,data in enumerate(data_loaders[split]):
                print(data)
                if step > 5:
                    break

    return data_loaders, feat_dim


def to_text_TrueCasingPunctuationTokenClassif_Morethan2Tasks_dataloader(
        data_dir: list,
        batch_size: int,
        max_sequence_length: int,
        frame_len: float,
        target_label_encoder,
        train_mode: str,
        segmentation_type: str,
        concat_aug: int, 
        test_file: str, 
        tokenizer):

    from daseg.dataloader_text_TrueCasingPunctuation import ConcatDataset_SidebySide, ConcatDataset_SidebySide_EqualizingLength, collate_fn_text_TrueCasingPunctuation_Morethan2Tasks, get_dataset_text_TrueCasing

    padding_value_features = 0
    mask_padding_with_zero = 1
    padding_value_mask = 0 if mask_padding_with_zero else 1
    
    if len(data_dir) > 1:
        raise ValueError(f'multiple data_dir is not supported for tasks other than EmoSpot')
    else:
        data_dir = data_dir[0]

    data_loaders = {}
    feat_dim = None
    if (train_mode == 'TE') or (train_mode == 'T'):
        train_cfn = partial(collate_fn_text_TrueCasingPunctuation_Morethan2Tasks, max_len=max_sequence_length, split='train',
                                target_label_encoder=target_label_encoder,
                                concat_aug=concat_aug, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len, 
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        dev_cfn = partial(collate_fn_text_TrueCasingPunctuation_Morethan2Tasks, max_len=max_sequence_length, split='dev',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        ## while getting the datasets, it's important to put max_len=None for atleast frame-based classification
        ## because you make the labels in the collate_fn
        train_ds = get_dataset_text_TrueCasing(data_dir, data_csv=data_dir+'/' + 'train' + '.tsv', max_len=None)
        train_ds = [train_ds]
        train_ds = ConcatDataset_SidebySide(*train_ds)

        dev_ds = get_dataset_text_TrueCasing(data_dir, data_csv=data_dir+'/' + 'dev' + '.tsv', max_len=None)
        print(f'train dataset length is {len(train_ds)}')
        print(f'dev dataset length is {len(dev_ds)}')

        train_sampler = RandomSampler(train_ds)
        dev_sampler = SequentialSampler(dev_ds)
        train_dataloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size, collate_fn=train_cfn, drop_last=True)
        dev_dataloader = DataLoader(dev_ds, sampler=dev_sampler, batch_size=1, collate_fn=dev_cfn)
        data_loaders['train'] = train_dataloader
        data_loaders['dev'] = dev_dataloader
        ## get feat_dim
        feat_dim = None
        #batch = next(iter(train_dataloader))
        #feat_dim = batch[0].shape[-1] 

    if (train_mode == 'TE') or (train_mode == 'E'):
        test_cfn = partial(collate_fn_text_TrueCasingPunctuation_Morethan2Tasks, max_len=None, split='test',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        if test_file == 'test.tsv':
            test_file = data_dir+ '/' + test_file
        print(f'loading {test_file} for evaluating the model')
        test_ds = get_dataset_text_TrueCasing(data_dir, data_csv=test_file, max_len=None)
        test_sampler = SequentialSampler(test_ds)
        test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=1, collate_fn=test_cfn)
    
        data_loaders['test'] = test_dataloader
        print(f'test dataset length is {len(test_ds)}')

    
    for split in ['train', 'dev', 'test']:
        if not split in data_loaders:
            data_loaders[split] = None
        
        else:
            print(f'\n printing few samples from the split {split} for debugging purposes \n')
            for step,data in enumerate(data_loaders[split]):
                print(data)
                if step > 5:
                    break
                #print(step)
                #pass

    return data_loaders, feat_dim


def to_text_TopicSegSeqLevelClassif_dataloader(
        data_dir: list,
        batch_size: int,
        max_sequence_length: int,
        frame_len: float,
        target_label_encoder,
        train_mode: str,
        segmentation_type: str,
        concat_aug: int, 
        test_file: str, 
        tokenizer):

    from daseg.dataloader_text_TrueCasingPunctuation import ConcatDataset_SidebySide, ConcatDataset_SidebySide_EqualizingLength, collate_fn_text_TopicSegSeqLevelClassif, get_dataset_text_TrueCasing

    padding_value_features = 0
    mask_padding_with_zero = 1
    padding_value_mask = 0 if mask_padding_with_zero else 1
    
    if len(data_dir) > 1:
        raise ValueError(f'multiple data_dir is not supported for tasks other than EmoSpot')
    else:
        data_dir = data_dir[0]

    data_loaders = {}
    feat_dim = None
    if (train_mode == 'TE') or (train_mode == 'T'):
        train_cfn = partial(collate_fn_text_TopicSegSeqLevelClassif, max_len=max_sequence_length, split='train',
                                target_label_encoder=target_label_encoder,
                                concat_aug=concat_aug, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len, 
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        dev_cfn = partial(collate_fn_text_TopicSegSeqLevelClassif, max_len=max_sequence_length, split='dev',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        ## while getting the datasets, it's important to put max_len=None for atleast frame-based classification
        ## because you make the labels in the collate_fn
        train_ds = get_dataset_text_TrueCasing(data_dir, data_csv=data_dir+'/' + 'train' + '.tsv', max_len=None)
        train_ds = [train_ds]
        train_ds = ConcatDataset_SidebySide(*train_ds)

        dev_ds = get_dataset_text_TrueCasing(data_dir, data_csv=data_dir+'/' + 'dev' + '.tsv', max_len=None)
        print(f'train dataset length is {len(train_ds)}')
        print(f'dev dataset length is {len(dev_ds)}')

        train_sampler = RandomSampler(train_ds)
        dev_sampler = SequentialSampler(dev_ds)
        train_dataloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size, collate_fn=train_cfn, drop_last=True)
        dev_dataloader = DataLoader(dev_ds, sampler=dev_sampler, batch_size=1, collate_fn=dev_cfn)
        data_loaders['train'] = train_dataloader
        data_loaders['dev'] = dev_dataloader
        ## get feat_dim
        feat_dim = None
        #batch = next(iter(train_dataloader))
        #feat_dim = batch[0].shape[-1] 

    if (train_mode == 'TE') or (train_mode == 'E'):
        test_cfn = partial(collate_fn_text_TopicSegSeqLevelClassif, max_len=None, split='test',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer)
        if test_file == 'test.tsv':
            test_file = data_dir+ '/' + test_file
        print(f'loading {test_file} for evaluating the model')
        test_ds = get_dataset_text_TrueCasing(data_dir, data_csv=test_file, max_len=None)
        test_sampler = SequentialSampler(test_ds)
        test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=1, collate_fn=test_cfn)
    
        data_loaders['test'] = test_dataloader
        print(f'test dataset length is {len(test_ds)}')

    
    for split in ['train', 'dev', 'test']:
        if not split in data_loaders:
            data_loaders[split] = None
        
        else:
            print(f'\n printing few samples from the split {split} for debugging purposes \n')
            for step,data in enumerate(data_loaders[split]):
                print(data)
                if step > 5:
                    break
                #print(step)
                #pass

    return data_loaders, feat_dim




def to_multimodal_dataloader_SeqClassification(
        data_dir: list,
        batch_size: int,
        max_sequence_length: int,
        frame_len: float,
        target_label_encoder,
        train_mode: str,
        segmentation_type: str,
        concat_aug: int, 
        test_file: str, 
        tokenizer,
        max_len_text: int):

    from daseg.dataloader_multimodal import ConcatDataset_SidebySide, ConcatDataset_SidebySide_EqualizingLength
    from daseg.dataloader_multimodal import collate_fn_multimodal as collate_fn
    from daseg.dataloader_multimodal import get_dataset_multimodal as get_dataset

    padding_value_features = 0
    mask_padding_with_zero = 1
    padding_value_mask = 0 if mask_padding_with_zero else 1
    
    if len(data_dir) > 1:
        print(f'multiple data_dir is not supported for tasks other than EmoSpot')
        sys.exit()
    else:
        data_dir = data_dir[0]

    data_loaders = {}
    feat_dim = None
    if (train_mode == 'TE') or (train_mode == 'T'):
        train_cfn = partial(collate_fn, max_len=max_sequence_length, split='train',
                                target_label_encoder=target_label_encoder,
                                concat_aug=concat_aug, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len, 
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer, max_len_text=max_len_text)
        dev_cfn = partial(collate_fn, max_len=max_sequence_length, split='dev',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer, max_len_text=max_len_text)
        ## while getting the datasets, it's important to put max_len=None for atleast frame-based classification
        ## because you make the labels in the collate_fn
        train_ds = get_dataset(data_dir, data_csv=data_dir+'/' + 'train' + '.tsv', max_len=None)
        train_ds = [train_ds]
        train_ds = ConcatDataset_SidebySide(*train_ds)

        dev_ds = get_dataset(data_dir, data_csv=data_dir+'/' + 'dev' + '.tsv', max_len=None)
        print(f'train dataset length is {len(train_ds)}')
        print(f'dev dataset length is {len(dev_ds)}')

        train_sampler = RandomSampler(train_ds)
        dev_sampler = SequentialSampler(dev_ds)
        train_dataloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size, collate_fn=train_cfn, drop_last=True)
        dev_dataloader = DataLoader(dev_ds, sampler=dev_sampler, batch_size=1, collate_fn=dev_cfn)
        data_loaders['train'] = train_dataloader
        data_loaders['dev'] = dev_dataloader
        ## get feat_dim
        feat_dim = None
        #batch = next(iter(train_dataloader))
        #feat_dim = batch[0].shape[-1] 

    if (train_mode == 'TE') or (train_mode == 'E'):
        test_cfn = partial(collate_fn, max_len=None, split='test',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer, max_len_text=max_len_text)
        if test_file == 'test.tsv':
            test_file = data_dir+ '/' + test_file
        print(f'loading {test_file} for evaluating the model')
        test_ds = get_dataset(data_dir, data_csv=test_file, max_len=None)
        test_sampler = SequentialSampler(test_ds)
        test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=1, collate_fn=test_cfn)
    
        data_loaders['test'] = test_dataloader
        print(f'test dataset length is {len(test_ds)}')
        ## get feat_dim
        #batch = next(iter(test_dataloader))
        #feat_dim_test = batch[0].shape[-1] 
        feat_dim = None
        #if feat_dim is not None:
        #    assert feat_dim_test == feat_dim
        #else:
        #    feat_dim = feat_dim_test

    for split in ['train', 'dev', 'test']:
        if not split in data_loaders:
            data_loaders[split] = None
        

    #import pdb; pdb.set_trace()
    #print('train')
    #for step,data in enumerate(train_dataloader):
    #    print(data[0].shape, data[1].shape, data[2].shape)
    #    #break
    #print('dev')
    #for step,data in enumerate(dev_dataloader):
    #    print(data[0].shape, data[1].shape, data[2].shape)
    #    #break
    #print('test')
    #for step,data in enumerate(test_dataloader):
    #    print(data[0].shape, data[1].shape, data[2].shape)
    #    break
 
    return data_loaders, feat_dim


def to_multimodal_multiloss_dataloader_SeqClassification(
        data_dir: list,
        batch_size: int,
        max_sequence_length: int,
        frame_len: float,
        target_label_encoder,
        train_mode: str,
        segmentation_type: str,
        concat_aug: int, 
        test_file: str, 
        tokenizer,
        max_len_text: int):

    from daseg.dataloader_multimodal_multiloss import ConcatDataset_SidebySide, ConcatDataset_SidebySide_EqualizingLength
    from daseg.dataloader_multimodal_multiloss import collate_fn_multimodal_multiloss as collate_fn
    from daseg.dataloader_multimodal_multiloss import get_dataset_multimodal as get_dataset
    
    extra_labels_path = '/export/b15/rpapagari/Tianzi_work/ADReSSo_NoVAD_IS2021_dataset/datainfo_diagnosis_cv10_8k.txt'
    extra_labels = pd.read_csv(extra_labels_path, sep=',')
    extra_labels = extra_labels[['utt_id', 'label', 'mmse']]
    extra_labels = extra_labels.drop_duplicates()
    extra_labels['mmse'] = extra_labels['mmse']/30
    extra_labels.set_index('utt_id', inplace=True)
    
    padding_value_features = 0
    mask_padding_with_zero = 1
    padding_value_mask = 0 if mask_padding_with_zero else 1
    
    if len(data_dir) > 1:
        print(f'multiple data_dir is not supported for tasks other than EmoSpot')
        sys.exit()
    else:
        data_dir = data_dir[0]

    data_loaders = {}
    feat_dim = None
    if (train_mode == 'TE') or (train_mode == 'T'):
        train_cfn = partial(collate_fn, max_len=max_sequence_length, split='train',
                                target_label_encoder=target_label_encoder,
                                concat_aug=concat_aug, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len, 
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer, max_len_text=max_len_text, 
                                extra_labels=extra_labels)
        dev_cfn = partial(collate_fn, max_len=max_sequence_length, split='dev',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer, max_len_text=max_len_text, 
                                extra_labels=extra_labels)
        ## while getting the datasets, it's important to put max_len=None for atleast frame-based classification
        ## because you make the labels in the collate_fn
        train_ds = get_dataset(data_dir, data_csv=data_dir+'/' + 'train' + '.tsv', max_len=None)
        train_ds = [train_ds]
        train_ds = ConcatDataset_SidebySide(*train_ds)

        dev_ds = get_dataset(data_dir, data_csv=data_dir+'/' + 'dev' + '.tsv', max_len=None)
        print(f'train dataset length is {len(train_ds)}')
        print(f'dev dataset length is {len(dev_ds)}')

        train_sampler = RandomSampler(train_ds)
        dev_sampler = SequentialSampler(dev_ds)
        train_dataloader = DataLoader(train_ds, sampler=train_sampler, batch_size=batch_size, collate_fn=train_cfn, drop_last=True)
        dev_dataloader = DataLoader(dev_ds, sampler=dev_sampler, batch_size=1, collate_fn=dev_cfn)
        data_loaders['train'] = train_dataloader
        data_loaders['dev'] = dev_dataloader
        ## get feat_dim
        feat_dim = None
        #batch = next(iter(train_dataloader))
        #feat_dim = batch[0].shape[-1] 

    if (train_mode == 'TE') or (train_mode == 'E'):
        test_cfn = partial(collate_fn, max_len=None, split='test',
                                target_label_encoder=target_label_encoder,
                                concat_aug=-1, 
                                padding_value_features=padding_value_features, 
                                padding_value_mask=padding_value_mask, 
                                padding_value_labels=CrossEntropyLoss().ignore_index, 
                                frame_len=frame_len,
                                segmentation_type=segmentation_type,
                                tokenizer=tokenizer, max_len_text=max_len_text,
                                extra_labels=extra_labels)
        if test_file == 'test.tsv':
            test_file = data_dir+ '/' + test_file
        print(f'loading {test_file} for evaluating the model')
        test_ds = get_dataset(data_dir, data_csv=test_file, max_len=None)
        test_sampler = SequentialSampler(test_ds)
        test_dataloader = DataLoader(test_ds, sampler=test_sampler, batch_size=1, collate_fn=test_cfn)
    
        data_loaders['test'] = test_dataloader
        print(f'test dataset length is {len(test_ds)}')
        ## get feat_dim
        #batch = next(iter(test_dataloader))
        #feat_dim_test = batch[0].shape[-1] 
        feat_dim = None
        #if feat_dim is not None:
        #    assert feat_dim_test == feat_dim
        #else:
        #    feat_dim = feat_dim_test

    for split in ['train', 'dev', 'test']:
        if not split in data_loaders:
            data_loaders[split] = None
        

    #import pdb; pdb.set_trace()
    #print('train')
    #for step,data in enumerate(train_dataloader):
    #    print(data[0].shape, data[1].shape, data[2].shape)
    #    #break
    #print('dev')
    #for step,data in enumerate(dev_dataloader):
    #    print(data[0].shape, data[1].shape, data[2].shape)
    #    #break
    #print('test')
    #for step,data in enumerate(test_dataloader):
    #    print(data[0].shape, data[1].shape, data[2].shape)
    #    break
 
    return data_loaders, feat_dim


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


def truncate_padding_collate_fn(batch: List[List[torch.Tensor]], padding_at_start: bool = False):
    redundant_padding = max(mask.sum() for _, mask, _, _ in batch)
    n_tensors = len(batch[0])
    concat_tensors = (torch.cat([sample[i].unsqueeze(0) for sample in batch]) for i in range(n_tensors))
    if padding_at_start:
        return [t[:, -redundant_padding:] for t in concat_tensors]
    return [t[:, :redundant_padding] for t in concat_tensors]


def pad_list_of_arrays(arrays: List[np.ndarray], value: float) -> List[np.ndarray]:
    set_of_size_lengths = set(len(x.shape) for x in arrays)
    if len(set_of_size_lengths) != 1:
        raise ValueError(f'there is some inconsistency in array sizes, please check')

    try:
        max_out_len = max(x.shape[1] for x in arrays)
    except:
        max_out_len = max(x.shape[0] for x in arrays)
    return [pad_array(t, target_len=max_out_len, value=value) for t in arrays]


def pad_array(arr: np.ndarray, target_len: int, value: float):
    if arr.shape[1] == target_len:
        return arr
    pad_shape = list(arr.shape)
    pad_shape[1] = target_len - arr.shape[1]
    return np.concatenate([arr, np.ones(pad_shape) * value], axis=1)
