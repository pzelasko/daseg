import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

from daseg.data import SwdaDataset


class TransformerModel:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            **json.load(open(Path(model_dir) / 'tokenizer_config.json'))
        )
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).to('cpu').eval()

    def predict(self, dataset: SwdaDataset) -> Dict[str, Any]:
        max_len = max(len(c.words()) for c in dataset.calls)
        # TODO: max len should be more bc of tokenization
        labels = list(self.config.label2id.keys())

        dataloader = dataset.to_transformers_ner_format(
            tokenizer=self.tokenizer,
            max_seq_length=max_len + 5,  # because Transformers will add some special tokens
            model_type=self.config.model_type,
            batch_size=1,
            labels=self.config.label2id.keys()
        )

        eval_losses = []
        preds = None
        out_label_ids = None
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = tuple(t.to('cpu') for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if self.config.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if self.config.model_type in ["bert", "xlnet"] else None
                    )  # XLM and RoBERTa don't use segment_ids
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_losses.append(tmp_eval_loss.item())
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=2)

        pad_token_label_id = CrossEntropyLoss().ignore_index
        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "losses": np.array(eval_losses),
            "predictions": preds_list,
            "true_labels": out_label_list,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
            "accuracy": accuracy_score(out_label_list, preds_list),
        }

        return results
