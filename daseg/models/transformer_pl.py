from pathlib import Path
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, AutoConfig, AutoModelForTokenClassification, AutoTokenizer, \
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, get_linear_schedule_with_warmup

from daseg.data import NEW_TURN
from daseg.dataloaders.transformers import pad_array
from daseg.losses.crf import CRFLoss
from daseg.metrics import as_tensors, compute_sklearn_metrics


class DialogActTransformer(pl.LightningModule):
    def __init__(
            self,
            labels: List[str],
            model_name_or_path: str,
            pretrained: bool = True,
            crf: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
            id2label={str(i): label for i, label in enumerate(self.labels)},
            label2id=self.label2id
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens({'additional_special_tokens': [NEW_TURN]})
        if pretrained:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name_or_path,
                from_tf='.ckpt' in model_name_or_path,
                config=self.config
            )
        else:
            model_class = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING[type(self.config)]
            self.model = model_class(self.config)
        self.model.resize_token_embeddings(len(self.tokenizer))
        if crf:
            self.crf = CRFLoss([l for l in self.labels if l != 'O' and not l.startswith('I-')], self.label2id)
        else:
            self.crf = None

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_num):
        "Compute loss and log."
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if self.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        outputs = self(**inputs)
        ce_loss, logits = outputs[:2]
        if self.crf is not None:
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            labels, ilens = batch[3], batch[4]
            crf_loss = -self.crf(log_probs, ilens, labels)
            ce_loss = 0.1 * ce_loss
            loss = crf_loss + ce_loss
            logs = {"loss": loss, 'crf_loss': crf_loss, 'ce_loss': ce_loss}
        else:
            logs = {"loss": loss}
        progdict = logs.copy()
        progdict.pop('loss')
        return {"loss": loss, "log": logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_nb):
        "Compute validation"

        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[2] if self.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        outputs = self(**inputs)
        loss, logits = outputs[:2]
        if self.crf is not None:
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            labels, ilens = batch[3], batch[4]
            loss = -self.crf(log_probs, ilens, labels)
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"]
        return {"val_loss": loss, "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def _eval_end(self, outputs):
        "Evaluation called for both Val and Test"
        outputs = pad_outputs(outputs)
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        preds = np.argmax(preds, axis=2)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        label_map = {i: label for i, label in enumerate(self.labels)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        sklearn_metrics = as_tensors(compute_sklearn_metrics(out_label_list, preds_list, compute_common_I=False))
        results = {
            "val_loss": val_loss_mean,
            **sklearn_metrics
        }

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs):
        # when stable
        ret, preds, targets = self._eval_end(outputs)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs):
        # updating to test_epoch_end instead of deprecated test_end
        ret, predictions, targets = self._eval_end(outputs)

        # Converting to the dict required by pl
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/master/\
        # pytorch_lightning/trainer/logging.py#L139
        logs = ret["log"]
        # `val_loss` is the key returned by `self._eval_end()` but actually refers to `test_loss`
        return {"avg_test_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def compute_and_set_total_steps(
            self,
            dataloader: DataLoader,
            gradient_accumulation_steps: int,
            num_epochs: int
    ):
        self.total_steps = len(dataloader) // gradient_accumulation_steps * num_epochs

    def get_lr_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=250, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.opt = AdamW(
                optimizer_grouped_parameters, 
                lr=5e-5, 
                eps=1e-8
        )
        self.scheduler = self.get_lr_scheduler()

        return [self.opt], [self.scheduler]

    def set_output_dir(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    #@pl.utilities.rank_zero_only
    #def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #    try:
    #        save_path = self.output_dir / "best_tfmr"
    #    except:
    #        traceback.print_exc()
    #        warnings.warn("on_save_checkpoint: can't store extra artifacts, "
    #                      "set_output_dir() was not called on the model.")
    #    else:
    #        self.model.config.save_step = self.step_count
    #        self.model.save_pretrained(save_path)
    #        self.tokenizer.save_pretrained(save_path)


def pad_outputs(outputs: Dict) -> Dict:
    max_out_len = max(x["pred"].shape[1] for x in outputs)
    for x in outputs:
        for k in ['pred', 'target']:
            if isinstance(x[k], torch.Tensor):
                x[k] = x[k].cpu().numpy()
        x["pred"] = pad_array(x["pred"], target_len=max_out_len, value=0)
        x["target"] = pad_array(x["target"], target_len=max_out_len, value=CrossEntropyLoss().ignore_index)
    return outputs
