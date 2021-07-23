import traceback
import warnings
from pathlib import Path
from typing import Dict, Any, List

from more_itertools import flatten
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification, AdamW, \
    get_linear_schedule_with_warmup, LongformerForTokenClassification, LongformerTokenizer, \
    LongformerConfig

from daseg.dataloaders.transformers import pad_list_of_arrays
from transformers.modeling_outputs import TokenClassifierOutput
from daseg.data import NEW_TURN
from daseg.dataloaders.transformers import pad_array
from daseg.metrics import as_tensors
from daseg.metrics import compute_sklearn_metrics, compute_seqeval_metrics, compute_zhao_kawahara_metrics, \
    compute_original_zhao_kawahara_metrics, compute_zhao_kawahara_metrics_speech

from daseg.loss_fun import LabelSmoothingCrossEntropy
from daseg.models.transformer_pl import DialogActTransformer
from daseg.models.x_vector import _XvectorModel_Regression

import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, inputs_embeds_dim, hidden_size=256, bidirectional=True, num_layers=6, loss_fun=None, num_labels=None):
        super().__init__()
        self.num_labels = num_labels
        self.bilstm = nn.LSTM(input_size=inputs_embeds_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first = True)
        if bidirectional:
            self.lstm_out_dim = 2*hidden_size
        else:
            self.lstm_out_dim = hidden_size
        self.fc1 = nn.Linear(self.lstm_out_dim, self.lstm_out_dim)
        self.fc2 = nn.Linear(self.lstm_out_dim, self.num_labels)

        self.loss_fun = loss_fun

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if not input_ids is None:
            raise ValueError(f'BiLSTM model is not yet implemented for discrete/text input')
        
        outputs = self.bilstm(inputs_embeds)
        outputs = outputs[0]
        outputs = F.relu(outputs)
        fc1_op = self.fc1(outputs)
        fc1_op = F.relu(fc1_op)

        logits = self.fc2(fc1_op)
        loss_fct = self.loss_fun # CrossEntropyLoss()
        loss = self.compute_loss(loss_fct, logits, attention_mask, labels, self.num_labels)
        tb_loss = torch.Tensor((loss,))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
            all_losses=tb_loss
        )

    def compute_loss(self, loss_fct, logits, attention_mask, labels, num_labels):
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.reshape(-1) == 1
                active_logits = logits.view(-1, num_labels)
                active_labels = torch.where(
                    active_loss, labels.reshape(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, num_labels), labels.reshape(-1))
        return loss



class BiLSTM_pl(pl.LightningModule):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0):
        super().__init__()
        self.save_hyperparameters()
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)        
        self.model = BiLSTM(inputs_embeds_dim, loss_fun=self.loss_fun, num_labels=self.num_labels)

    @staticmethod
    def from_path(model_path: Path, device: str = 'cpu'):
        pl_model = BiLSTM_pl.load_from_checkpoint(str(model_path), map_location=device)
        model = pl_model.model
        return model

    def forward(self, **inputs):
        return self.model(**inputs)

    def predict(self, dataset, batch_size=1, label_scheme=None, compute_metrics=True):
        out_label_ids = []
        logits = []
        eval_ce_losses = []
        true_seg_boundaries = []        
        self.model = self.model.eval()
        batch_keys = []
        with torch.no_grad():
            for step,batch in enumerate(dataset):
                utt_id = batch[-1]
                batch_keys.append(utt_id)
                inputs = {"inputs_embeds": batch[0], "attention_mask": batch[1], "labels": batch[2], "token_type_ids":batch[4]}
                utt_true_seg_boundaries = batch[3].detach().cpu().numpy()
                true_seg_boundaries.append(utt_true_seg_boundaries)
                model_op = self.model(**inputs)
                utt_loss = model_op.loss.detach().cpu().numpy()
                utt_logits = model_op.logits.detach().cpu().numpy()
                eval_ce_losses.append(utt_loss)
                try:
                    utt_labels = model_op.subsampled_labels.detach().cpu().numpy()
                except:
                    utt_labels = inputs['labels']
                out_label_ids.append(utt_labels)
                logits.append(utt_logits)
        results = {} 
        pad_token_label_id = CrossEntropyLoss().ignore_index
        out_label_ids = pad_list_of_arrays(out_label_ids, value=pad_token_label_id)
        logits = pad_list_of_arrays(logits, value=0)
        out_label_ids = np.concatenate(out_label_ids, axis=0)
        logits = np.concatenate(logits, axis=0)
        preds = np.argmax(logits, axis=2)

        label_map = self.label_map
        out_label_list: List[List[str]] = [[] for _ in range(out_label_ids.shape[0])]
        preds_list: List[List[str]] = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        eval_ce_losses = list(flatten(np.vstack(eval_ce_losses)))
        results = {
            "losses": np.array(eval_ce_losses),
            "loss": np.mean(eval_ce_losses),
            "predictions": preds_list,
            "logits": logits,
            "true_labels": out_label_list,
            "true_seg_boundaries": true_seg_boundaries,
            "utt_id": batch_keys
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

        return results


    def training_step(self, batch, batch_num):
        "Compute loss and log."
        inputs = {"inputs_embeds": batch[0], "attention_mask": batch[1], "labels": batch[2], "token_type_ids":batch[4]}
        #if self.config.model_type != "distilbert":
        #    inputs["token_type_ids"] = (
        #        batch[2] if self.config.model_type in ["bert", "xlnet"] else None
        #    )  # XLM and RoBERTa don"t use token_type_ids

        outputs = self(**inputs)
        loss = outputs[0]
        all_losses = outputs.all_losses
        if not all_losses is None:
            tensorboard_logs = {'loss'+str(ind):loss_value for ind,loss_value in enumerate(all_losses)}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        "Compute validation"
        inputs = {"inputs_embeds": batch[0], "attention_mask": batch[1], "labels": batch[2], "token_type_ids":batch[4]}
        #if self.config.model_type != "distilbert":
        #    inputs["token_type_ids"] = (
        #        batch[2] if self.config.model_type in ["bert", "xlnet"] else None
        #    )  # XLM and RoBERTa don"t use token_type_ids
        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        all_losses = outputs.all_losses
        preds = logits.detach().cpu().numpy()
        try:
            out_label_ids = outputs.subsampled_labels.detach().cpu().numpy()
        except:
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        tensorboard_logs = {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}
        if not all_losses is None:
            tensorboard_logs_losses = {'val_loss'+str(ind):loss_value for ind,loss_value in enumerate(all_losses)}
            tensorboard_logs.update(tensorboard_logs_losses)
        return tensorboard_logs

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

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters())
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optim, factor=0.9, patience=3, verbose=True,
        )
        return [optim], [sched]

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



class ResNet34(nn.Module):
    def __init__(self, inputs_embeds_dim, loss_fun=None, num_labels=None, pretrained_model_path=None, pretrained=False):
        super().__init__()
        self.num_labels = num_labels
        self.x_vector_model = _XvectorModel_Regression(pretrained, pretrained_model_path, output_channels=num_labels, input_dim=inputs_embeds_dim)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, self.num_labels)

        self.loss_fun = loss_fun

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if not input_ids is None:
            raise ValueError(f'ResNet model is not yet implemented for discrete/text input')

        ###### for x-vector model ##########
        x_vector_model_op = self.x_vector_model.model.forward_encoder(inputs_embeds)

        subsampling_factor = 8
        attention_mask = attention_mask[:, ::subsampling_factor]
        labels = labels[:, ::subsampling_factor]
        token_type_ids = token_type_ids[:, ::subsampling_factor]
        assert attention_mask.shape[-1] == x_vector_model_op.shape[1]

        outputs = x_vector_model_op
        outputs = F.relu(outputs)
        fc1_op = self.fc1(outputs)
        fc1_op = F.relu(fc1_op)

        logits = self.fc2(fc1_op)
        loss_fct = self.loss_fun # CrossEntropyLoss()
        loss = self.compute_loss(loss_fct, logits, attention_mask, labels, self.num_labels)
        tb_loss = torch.Tensor((loss,))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
            all_losses=tb_loss,
            subsampled_labels=labels
        )

    def compute_loss(self, loss_fct, logits, attention_mask, labels, num_labels):
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.reshape(-1) == 1
                active_logits = logits.view(-1, num_labels)
                active_labels = torch.where(
                    active_loss, labels.reshape(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, num_labels), labels.reshape(-1))
        return loss


class ResNet34_pl(BiLSTM_pl):
    def __init__(self, target_label_encoder, model_name_or_path: str, 
                        inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, 
                        emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)        
        self.model = ResNet34(inputs_embeds_dim, loss_fun=self.loss_fun, 
                                num_labels=self.num_labels, 
                                pretrained_model_path=pretrained_model_path, 
                                pretrained=pre_trained_model)


class ResNet34_SeqClassification(nn.Module):
    def __init__(self, inputs_embeds_dim, loss_fun=None, num_labels=None, pretrained_model_path=None, pretrained=False):
        super().__init__()
        self.num_labels = num_labels
        self.x_vector_model = _XvectorModel_Regression(pretrained, pretrained_model_path, output_channels=num_labels, input_dim=inputs_embeds_dim)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, self.num_labels)

        self.loss_fun = loss_fun

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if not input_ids is None:
            raise ValueError(f'ResNet model is not yet implemented for discrete/text input')

        ####### for x-vector model ##########
        #x_vector_model_op = self.x_vector_model.model.forward_encoder(inputs_embeds)

        #subsampling_factor = 8
        #attention_mask = attention_mask[:, ::subsampling_factor]
        #labels = labels[:, ::subsampling_factor]
        #token_type_ids = token_type_ids[:, ::subsampling_factor]
        #assert attention_mask.shape[-1] == x_vector_model_op.shape[1]

        #outputs = x_vector_model_op
        #outputs = F.relu(outputs)
        #fc1_op = self.fc1(outputs)
        #fc1_op = F.relu(fc1_op)
        #logits = self.fc2(fc1_op)

        logits = self.x_vector_model(inputs_embeds, labels=None)
        loss_fct = self.loss_fun # CrossEntropyLoss()
        loss = self.compute_loss(loss_fct, logits, attention_mask, labels, self.num_labels)
        tb_loss = torch.Tensor((loss,))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
            all_losses=tb_loss,
            subsampled_labels=labels
        )

    def compute_loss(self, loss_fct, logits, attention_mask, labels, num_labels):
        loss = None
        if labels is not None:
            if num_labels == 1:
                #  We are doing regression
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return loss


class ResNet34_SeqClassification_pl(pl.LightningModule):
    def __init__(self, target_label_encoder, model_name_or_path: str, 
                        inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, 
                        emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        if not target_label_encoder is None:
            self.target_label_encoder = target_label_encoder
            self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
            self.labels = self.target_label_encoder.classes_
            self.num_labels = len(self.labels)
            self.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        else:
            ## usually for regression tasks 
            self.label_map = None
            self.num_labels = 1
            self.loss_fun = nn.MSELoss()
            #self.loss_fun = nn.L1Loss()
        #import pdb; pdb.set_trace()
        self.model = ResNet34_SeqClassification(inputs_embeds_dim, loss_fun=self.loss_fun, 
                                num_labels=self.num_labels, 
                                pretrained_model_path=pretrained_model_path, 
                                pretrained=pre_trained_model)

    @staticmethod
    def from_path(model_path: Path, device: str = 'cpu'):
        pl_model = ResNet34_SeqClassification_pl.load_from_checkpoint(str(model_path), map_location=device)
        model = pl_model.model
        return model

    def forward(self, **inputs):
        return self.model(**inputs)

    def predict(self, dataset, batch_size=1, label_scheme=None, compute_metrics=True):
        out_label_ids = []
        logits = []
        eval_ce_losses = []
        true_seg_boundaries = []        
        self.model = self.model.eval()
        batch_keys = []
        with torch.no_grad():
            for step,batch in enumerate(dataset):
               
                utt_id = batch[-1]
                batch_keys.append(utt_id)
                inputs = {"inputs_embeds": batch[0], "attention_mask": batch[1], "labels": batch[2], "token_type_ids":batch[4]}
                utt_true_seg_boundaries = batch[3].detach().cpu().numpy()
                true_seg_boundaries.append(utt_true_seg_boundaries)
                model_op = self.model(**inputs)
                utt_loss = model_op.loss.detach().cpu().numpy()
                utt_logits = model_op.logits.detach().cpu().numpy()
                eval_ce_losses.append(utt_loss)
                try:
                    utt_labels = model_op.subsampled_labels.detach().cpu().numpy()
                except:
                    utt_labels = inputs['labels']
                out_label_ids.append(utt_labels)
                logits.append(utt_logits)
        results = {} 
        pad_token_label_id = CrossEntropyLoss().ignore_index
        #out_label_ids = pad_list_of_arrays(out_label_ids, value=pad_token_label_id)
        #logits = pad_list_of_arrays(logits, value=0)
        out_label_ids = np.concatenate(out_label_ids, axis=0)
        logits = np.concatenate(logits, axis=0)

        eval_ce_losses = list(flatten(np.vstack(eval_ce_losses)))
        if self.num_labels > 1:
            preds = np.argmax(logits, axis=-1)
            label_map = self.label_map
            out_label_list = [[label_map[i]  for i in out_label_ids]]
            preds_list = [[label_map[i] for i in preds]]
        else:
            out_label_list = [out_label_ids]
            preds_list = [logits]
        results = {
            "losses": np.array(eval_ce_losses),
            "loss": np.mean(eval_ce_losses),
            "predictions": preds_list,
            "logits": logits,
            "true_labels": out_label_list,
            "true_seg_boundaries": true_seg_boundaries,
            "utt_id": batch_keys
        }
        if (self.num_labels > 1) and compute_metrics:
            results.update({
                "sklearn_metrics": compute_sklearn_metrics(out_label_list, preds_list),
                "seqeval_metrics": compute_seqeval_metrics(out_label_list, preds_list),
                # We show the metrics obtained with Zhao-Kawahara code which computes them differently
                # (apparently the segment insertion errors are not counted)
                "ORIGINAL_zhao_kawahara_metrics": compute_original_zhao_kawahara_metrics(
                    true_turns=out_label_list, pred_turns=preds_list),
                "zhao_kawahara_metrics": compute_zhao_kawahara_metrics_speech(true_labels=out_label_list, pred_labels=preds_list, true_seg_boundaries=true_seg_boundaries, label_scheme=label_scheme)
            })

        return results


    def training_step(self, batch, batch_num):
        "Compute loss and log."
        inputs = {"inputs_embeds": batch[0], "attention_mask": batch[1], "labels": batch[2], "token_type_ids":batch[4]}
        #if self.config.model_type != "distilbert":
        #    inputs["token_type_ids"] = (
        #        batch[2] if self.config.model_type in ["bert", "xlnet"] else None
        #    )  # XLM and RoBERTa don"t use token_type_ids

        outputs = self(**inputs)
        loss = outputs[0]
        all_losses = outputs.all_losses
        if not all_losses is None:
            tensorboard_logs = {'loss'+str(ind):loss_value for ind,loss_value in enumerate(all_losses)}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        "Compute validation"
        inputs = {"inputs_embeds": batch[0], "attention_mask": batch[1], "labels": batch[2], "token_type_ids":batch[4]}
        #if self.config.model_type != "distilbert":
        #    inputs["token_type_ids"] = (
        #        batch[2] if self.config.model_type in ["bert", "xlnet"] else None
        #    )  # XLM and RoBERTa don"t use token_type_ids
        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        all_losses = outputs.all_losses
        preds = logits.detach().cpu().numpy()
        try:
            out_label_ids = outputs.subsampled_labels.detach().cpu().numpy()
        except:
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        tensorboard_logs = {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}
        if not all_losses is None:
            tensorboard_logs_losses = {'val_loss'+str(ind):loss_value for ind,loss_value in enumerate(all_losses)}
            tensorboard_logs.update(tensorboard_logs_losses)
        return tensorboard_logs

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def _eval_end(self, outputs):
        "Evaluation called for both Val and Test"
        #outputs = pad_outputs(outputs)
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        if  self.num_labels > 1:
            preds = np.argmax(preds, axis=-1)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        #label_map = {i: label for i, label in enumerate(self.labels)}
        #out_label_list = [[label_map[i]  for i in out_label_ids]]
        #preds_list = [[label_map[i] for i in preds]]
        out_label_list = [list(out_label_ids)]
        preds_list = [list(preds)]
        if  self.num_labels == 1:
            results = {'val_loss': val_loss_mean}
        else:
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

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.model.parameters())
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optim, factor=0.9, patience=3, verbose=True,
        )
        return [optim], [sched]

    def set_output_dir(self, output_dir: Path):
        self.output_dir = Path(output_dir)


class BiLSTM_SeqClassification_pl(ResNet34_SeqClassification_pl):
    def __init__(self, target_label_encoder, model_name_or_path: str, 
                        inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, 
                        emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None, normalize_ip_feats=False):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        if not target_label_encoder is None:
            self.target_label_encoder = target_label_encoder
            self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
            self.labels = self.target_label_encoder.classes_
            self.num_labels = len(self.labels)
            self.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        else:
            ## usually for regression tasks 
            self.label_map = None
            self.num_labels = 1
            self.loss_fun = nn.MSELoss()
            #self.loss_fun = nn.L1Loss()
        #import pdb; pdb.set_trace()
        self.num_layers = 2 # 6
        self.normalize_ip_feats = normalize_ip_feats
        self.model = BiLSTM_SeqClassification(inputs_embeds_dim, loss_fun=self.loss_fun, 
                                num_labels=self.num_labels, normalize_ip_feats=self.normalize_ip_feats, num_layers=self.num_layers)


class BiLSTM_SeqClassification(nn.Module):
    def __init__(self, inputs_embeds_dim, hidden_size=256, bidirectional=True, num_layers=6, loss_fun=None, num_labels=None, normalize_ip_feats=False):
        super().__init__()
        self.num_labels = num_labels
        self.normalize_ip_feats = normalize_ip_feats

        if self.normalize_ip_feats:
            self.norm_layer = nn.BatchNorm1d(inputs_embeds_dim)

        self.bilstm = nn.LSTM(input_size=inputs_embeds_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first = True)
        self.hidden_size = hidden_size
        if bidirectional:
            self.lstm_out_dim = 2*self.hidden_size
        else:
            self.lstm_out_dim = self.hidden_size
        self.fc1 = nn.Linear(self.lstm_out_dim, self.lstm_out_dim)
        self.fc2 = nn.Linear(self.lstm_out_dim, self.num_labels)

        self.loss_fun = loss_fun

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if not input_ids is None:
            raise ValueError(f'BiLSTM model is not yet implemented for discrete/text input')
        
        if self.normalize_ip_feats:
            inputs_embeds = inputs_embeds.permute(0, 2, 1)
            inputs_embeds = self.norm_layer(inputs_embeds)
            inputs_embeds = inputs_embeds.permute(0, 2, 1)
        outputs = self.bilstm(inputs_embeds)
        final_state = outputs[0]
        outputs = torch.mean(final_state, axis=1)
        #outputs = torch.cat([final_state[:, -1, :self.hidden_size], final_state[:, -1, self.hidden_size:]], -1)
        outputs = F.relu(outputs)
        fc1_op = self.fc1(outputs)
        fc1_op = F.relu(fc1_op)

        logits = self.fc2(fc1_op)
        loss_fct = self.loss_fun # CrossEntropyLoss()
        loss = self.compute_loss(loss_fct, logits, attention_mask, labels, self.num_labels)
        tb_loss = torch.Tensor((loss,))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
            all_losses=tb_loss
        )

    def compute_loss(self, loss_fct, logits, attention_mask, labels, num_labels):
        loss = None
        if labels is not None:
            if num_labels == 1:
                #  We are doing regression
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return loss


def pad_outputs(outputs: Dict) -> Dict:
    max_out_len = max(x["pred"].shape[1] for x in outputs)
    for x in outputs:
        x["pred"] = pad_array(x["pred"], target_len=max_out_len, value=0)
        x["target"] = pad_array(x["target"], target_len=max_out_len, value=CrossEntropyLoss().ignore_index)
    return outputs


