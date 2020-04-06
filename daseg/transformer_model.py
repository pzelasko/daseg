import json
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

from daseg.data import SwdaDataset, to_transformers_ner_dataset

__all__ = ['TransformerModel']


class TransformerModel:
    def __init__(self, model_dir: Path, device: str = 'cpu'):
        self.model_dir = model_dir
        self.config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            **json.load(open(Path(model_dir) / 'tokenizer_config.json'))
        )
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device).eval()

        # TODO: We are optionally using a CRF just for constraints on possible transitions between tags
        #       We might eventually train the CRF transition probs along with the model;
        #       Currently, they have equal an likelihood mass.
        id2label = self.model.config.id2label
        self.crf = ConditionalRandomField(
            num_tags=len(id2label),
            constraints=allowed_transitions(
                constraint_type='IOB1',
                labels=id2label
            )
        )
        self.crf.transitions = torch.nn.Parameter(torch.ones(self.crf.transitions.shape), requires_grad=False)
        self.crf.start_transitions = torch.nn.Parameter(torch.ones(self.crf.start_transitions.shape),
                                                        requires_grad=False)
        self.crf.end_transitions = torch.nn.Parameter(torch.ones(self.crf.end_transitions.shape), requires_grad=False)

    def predict(
            self,
            dataset: SwdaDataset,
            batch_size: int = 1,
            forced_max_len: Optional[int] = None,
            crf_decoding: bool = False
    ) -> Dict[str, Any]:
        max_len = forced_max_len if forced_max_len is not None else 2 * max(len(c.words()) for c in dataset.calls)
        # TODO: max len should be more bc of tokenization, for now 2 * is a heuristic...
        labels = list(self.config.label2id.keys())

        dataloader = dataset.to_transformers_ner_format(
            tokenizer=self.tokenizer,
            max_seq_length=max_len,
            model_type=self.config.model_type,
            batch_size=batch_size,
            labels=self.config.label2id.keys()
        )

        eval_losses, logits, out_label_ids = zip(*list(tqdm(
            map(
                partial(predict_batch, model=self.model, config=self.config),
                dataloader
            ),
            desc=f'Predicting dialog acts (batches of {batch_size})',
            leave=False
        )))

        out_label_ids = np.concatenate(out_label_ids, axis=0)
        logits = np.concatenate(logits, axis=0)
        if crf_decoding:
            preds, lls = zip(*self.crf.viterbi_tags(torch.from_numpy(logits)))
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

        results = {
            "losses": np.array(eval_losses),
            "predictions": preds_list,
            "logits": logits,
            "true_labels": out_label_list,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
            "accuracy": accuracy_score(out_label_list, preds_list),
            "dataset": predictions_to_dataset(dataset, preds_list)
        }

        return results


def predict_batch(batch, model, config):
    batch = tuple(t.to('cpu') for t in batch)
    with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don't use segment_ids
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
    return tmp_eval_loss, logits.detach().cpu().numpy(), inputs["labels"].detach().cpu().numpy()


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
