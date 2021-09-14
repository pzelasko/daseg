import traceback
import warnings
from pathlib import Path
from typing import Dict, Any, List

from copy import deepcopy
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification, AdamW, \
    get_linear_schedule_with_warmup,  LongformerTokenizer, \
    LongformerConfig, AutoModelForSequenceClassification
from transformers.modeling_outputs import TokenClassifierOutput

from daseg.models.longformer_model import LongformerForTokenClassification, LongformerForTokenClassificationEmoSpot, XFormerForTokenClassificationEmoSpot, XFormerPoolSegments, XFormerCNNOPPoolSegments, LongformerForSequenceClassification, XFormerForSeqClassification, LongformerModel, LongformerPreTrainedModel, LongformerLayer, LongformerEncoder
from daseg.models.modeling_bert import BertForSequenceClassification, BertEncoder, BertModel, BertLayer, BertPooler, BertPreTrainedModel

from daseg.data import NEW_TURN
from daseg.dataloaders.transformers import pad_array
from daseg.metrics import as_tensors, compute_sklearn_metrics
from daseg.loss_fun import LabelSmoothingCrossEntropy
from daseg.models.x_vector import _XvectorModel_Regression
import torch.nn.functional as F


###########  text seq classification
class TrueCasingTransformer(pl.LightningModule):
    def __init__(self, labels, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, max_sequence_length: int, label_smoothing_alpha=0):
        super().__init__()
        self.save_hyperparameters()
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.labels = labels #list(target_label_encoder.classes_)
        self.num_labels = len(self.labels)
        #self.id2label = {str(i): label for i, label in enumerate(self.labels)},
        #self.label2id = {label: i for i, label in enumerate(self.labels)},
        #self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        #self.label2id = {label:int(ind) for ind,label in self.id2label.items()}
        model_name_or_path = 'allenai/longformer-base-4096' 
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
            #id2label=self.id2label,
            #label2id=self.label2id,
            id2label={str(i): label for i, label in enumerate(self.labels)},
            label2id={label: i for i, label in enumerate(self.labels)},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        #self.model = AutoModelForTokenClassification.from_pretrained(
        self.model = LongformerForTokenClassification.from_pretrained(
            model_name_or_path,
            from_tf='.ckpt' in model_name_or_path,
            config=self.config)
        self.tokenizer.add_special_tokens({'additional_special_tokens': [NEW_TURN]})
        self.model.resize_token_embeddings(len(self.tokenizer))    
        self.model_input_keys = ['input_ids', 'attention_mask', 'labels', 'token_type_ids']
   

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_num):
        "Compute loss and log."

        #inputs = {"inputs_embeds": batch[0], "attention_mask": batch[1], "labels": batch[2], "token_type_ids":batch[4]}
        inputs = {}
        for i in self.model_input_keys:
            inputs[i] = batch[i]

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch["token_type_ids"] if self.config.model_type in ["bert", "xlnet"] else None
            )

        outputs = self(**inputs)
        loss = outputs[0]
        all_losses = outputs.all_losses
        if not all_losses is None:
            tensorboard_logs = {'loss'+str(ind):loss_value for ind,loss_value in enumerate(all_losses)}
        else:
            tensorboard_logs = {"loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        "Compute validation"
        inputs = {}
        for i in self.model_input_keys:
            inputs[i] = batch[i]

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch["token_type_ids"] if self.config.model_type in ["bert", "xlnet"] else None
            )
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

    def get_lr_scheduler(self):
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=200, num_training_steps=self.total_steps
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def set_output_dir(self, output_dir: Path):
        self.output_dir = Path(output_dir)


class TrueCasingPunctuationTransformer(pl.LightningModule):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, max_sequence_length: int, label_smoothing_alpha=0, loss_weights=[1, 1]):
        super().__init__()
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.id2label = []
        self.label2id = []
        for label_block_name in target_label_encoder.keys():
            self.labels = list(target_label_encoder[label_block_name].classes_)
            self.num_labels = len(self.labels)
            id2label_temp = {str(target_label_encoder[label_block_name].transform([label])[0]):label for label in self.labels}
            label2id_temp = {label:int(ind) for ind,label in id2label_temp.items()}
            self.id2label += [id2label_temp]
            self.label2id += [label2id_temp]

        ###### label_block_name_for_model_config is just to get the config        
        label_block_name_for_model_config = list(target_label_encoder.keys())[0]
        self.labels = list(target_label_encoder[label_block_name_for_model_config].classes_)
        self.num_labels = len(self.labels)
        model_name_or_path = 'allenai/longformer-base-4096' 
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
            id2label=self.id2label[0],
            label2id=self.label2id[0],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        ########## Specific to multi tasking  ########################
        self.config.label_maps_id2label = self.id2label
        self.config.label_maps_label2id = self.label2id

        self.config.loss_hyperparameters = {}
        self.config.loss_hyperparameters['num_labels'] = [len(target_label_encoder[label_block_name].classes_)  for label_block_name in target_label_encoder.keys()]
        print(f'num_labels at the output are {self.config.loss_hyperparameters["num_labels"]}')
        #self.config.loss_hyperparameters['num_labels'] = [len(target_label_encoder['label'].classes_), len(target_label_encoder['label2'].classes_)]
        self.config.loss_hyperparameters['loss_types'] = [LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha) for _ in target_label_encoder.keys()]
        #                                            LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)]

        self.config.loss_hyperparameters['loss_weights'] = loss_weights
        print(f'loss_weights are { self.config.loss_hyperparameters["loss_weights"]}')
        
        ##############################################################

        #import pdb; pdb.set_trace()    
        self.model = LongformerForTokenClassificationTrueCasingPunctuation(
            model_name_or_path,
            from_tf='.ckpt' in model_name_or_path,
            config=self.config)

        self.tokenizer.add_special_tokens({'additional_special_tokens': [NEW_TURN]})
        self.model.resize_token_embeddings(len(self.tokenizer))    
        self.model_input_keys = ['input_ids', 'attention_mask', 'labels', 'token_type_ids']
   

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_num):
        "Compute loss and log."
        inputs = {}
        for i in self.model_input_keys:
            inputs[i] = batch[i]

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch["token_type_ids"] if self.config.model_type in ["bert", "xlnet"] else None
            )

        outputs = self(**inputs)
        #import pdb; pdb.set_trace()
        loss_per_op = outputs.loss
        wts = self.config.loss_hyperparameters['loss_weights']
        loss = 0
        for ind in range(len(wts)):
            loss += loss_per_op[ind] * wts[ind]
        tensorboard_logs = {"loss": loss}
        return {"loss": loss, "log": tensorboard_logs}        

    def validation_step(self, batch, batch_nb):
        "Compute validation"
        inputs = {}
        for i in self.model_input_keys:
            inputs[i] = batch[i]

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch["token_type_ids"] if self.config.model_type in ["bert", "xlnet"] else None
            )
        outputs = self(**inputs)       
        #import pdb; pdb.set_trace()

        #tmp_eval_loss, logits = outputs[:2]
        tmp_eval_loss = outputs.loss
        logits = outputs.logits
        tmp_eval_loss = [tmp_eval_loss[i].detach().cpu().numpy() for i in range(len(logits))]
        preds = [logits[i].detach().cpu().numpy() for i in range(len(logits))]
        out_label_ids = [batch["labels"][i].detach().cpu().numpy() for i in range(len(logits))]
        return {"val_loss": tmp_eval_loss, "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def _eval_end(self, outputs):
        "Evaluation called for both Val and Test"
        num_op_types = len(self.config.loss_hyperparameters['num_labels'])
        outputs = pad_outputs_MultiLoss(outputs, num_op_types)

        results = {}
        ret = {}
        #import pdb; pdb.set_trace()
        for op_ind in range(num_op_types):
            val_loss_mean = np.stack([x["val_loss"][op_ind] for x in outputs]).mean()
            val_loss_mean = torch.as_tensor(val_loss_mean)
            preds = np.concatenate([x["pred"][op_ind] for x in outputs], axis=0)
            preds = np.argmax(preds, axis=-1)
            out_label_ids = np.concatenate([x["target"][op_ind] for x in outputs], axis=0)
            #out_label_list = [list(out_label_ids)]
            #preds_list = [list(preds)]

            out_label_list = [[] for _ in range(out_label_ids.shape[0])]
            preds_list = [[] for _ in range(out_label_ids.shape[0])]
    
            ## remove lobel locations not used (i.e., which have -100)
            for i in range(out_label_ids.shape[0]):
                for j in range(out_label_ids.shape[1]):
                    if out_label_ids[i, j] != self.pad_token_label_id:
                        out_label_list[i].append(out_label_ids[i][j])
                        preds_list[i].append(preds[i][j])

            if isinstance(self.config.loss_hyperparameters['loss_types'][op_ind], LabelSmoothingCrossEntropy):
                sklearn_metrics = as_tensors(compute_sklearn_metrics(out_label_list, preds_list, compute_common_I=False))
                sklearn_metrics.update({'val_loss':val_loss_mean})
                results = {i+'_op'+str(op_ind):j for i,j in sklearn_metrics.items()}
                ret.update({k: v for k, v in results.items()})
                #ret["log"] = results
            else:
                raise ValueError(f'Not implemented for targets other than categorical targets')

        #import pdb; pdb.set_trace()
        wts = self.config.loss_hyperparameters['loss_weights']
        val_loss = 0
        ret["log"] = {}
        for op_ind in range(len(wts)):
            val_loss += ret['val_loss'+'_op'+str(op_ind)] * wts[op_ind]
            ret["log"]['val_loss'+'_op'+str(op_ind)] = ret['val_loss'+'_op'+str(op_ind)]
            ret["log"]['micro_f1'+'_op'+str(op_ind)] = ret['micro_f1'+'_op'+str(op_ind)]
            ret["log"]['macro_f1'+'_op'+str(op_ind)] = ret['macro_f1'+'_op'+str(op_ind)]
            
        ret["log"]['val_loss'] = val_loss

        for key in ['macro_f1', 'micro_f1']:
            perf_list = [ret['log'][i] for i in ret.keys() if i.startswith(key)]
            perf_ave = sum(perf_list)/len(perf_list)
            ret["log"][key+'_ave'] = perf_ave

        return ret

    def validation_epoch_end(self, outputs):
        # when stable
        ret = self._eval_end(outputs)
        logs = ret["log"]
        print(logs)
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs):
        # updating to test_epoch_end instead of deprecated test_end
        ret = self._eval_end(outputs)

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
            self.opt, num_warmup_steps=200, num_training_steps=self.total_steps
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def set_output_dir(self, output_dir: Path):
        self.output_dir = Path(output_dir)


class LongformerForTokenClassificationTrueCasingPunctuation(LongformerPreTrainedModel):
    def __init__(self, model_name_or_path, from_tf, config):
        super().__init__(config)

        self.config = config
        #self.longformer = LongformerModel(config, add_pooling_layer=False)

        self.longformer = LongformerModel.from_pretrained(
            model_name_or_path,
            from_tf='.ckpt' in model_name_or_path,
            config=self.config, add_pooling_layer=False)

        self.dropout = nn.ModuleList([])
        self.classifier = nn.ModuleList([])
        for ind, label_count in enumerate(self.config.loss_hyperparameters['num_labels']):
            self.dropout.append(nn.Dropout(config.hidden_dropout_prob))
            self.classifier.append(nn.Linear(config.hidden_size, label_count))

        #self.init_weights() ## TODO: Do we need this?

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
        **kwargs
    ):                
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #print(inputs_embeds.shape, attention_mask.shape, labels.shape)
        #import pdb; pdb.set_trace()
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = []
        total_loss = []
        for ind,label_count in enumerate(self.config.loss_hyperparameters['num_labels']):
            logits_perlabeltype = self.dropout[ind](sequence_output)
            logits_perlabeltype = self.classifier[ind](logits_perlabeltype)

            loss_fct = self.config.loss_hyperparameters['loss_types'][ind]
            loss_perlabeltype = self.compute_loss(loss_fct, logits_perlabeltype, attention_mask, labels[ind], label_count)
            logits.append(logits_perlabeltype)
            total_loss.append(loss_perlabeltype)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            all_losses=total_loss
        )
    
    def compute_loss(self, loss_fct, logits, attention_mask, labels, num_labels):
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
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


def pad_outputs(outputs: Dict) -> Dict:
    max_out_len = max(x["pred"].shape[1] for x in outputs)
    for x in outputs:
        x["pred"] = pad_array(x["pred"], target_len=max_out_len, value=0)
        x["target"] = pad_array(x["target"], target_len=max_out_len, value=CrossEntropyLoss().ignore_index)
    return outputs


def pad_outputs_MultiLoss(outputs: Dict, num_op_types) -> Dict:
    for op_ind in range(num_op_types):
        max_out_len = max(x["pred"][op_ind].shape[1] for x in outputs)
        for x in outputs:
            x["pred"][op_ind] = pad_array(x["pred"][op_ind], target_len=max_out_len, value=0)
            x["target"][op_ind] = pad_array(x["target"][op_ind], target_len=max_out_len, value=CrossEntropyLoss().ignore_index)
    return outputs


class TrueCasingPunctuationBERT(TrueCasingPunctuationTransformer):
    ''' This class is to be able to train any HuggingFace (HF) model instead of just longformer
    '''
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, max_sequence_length: int, label_smoothing_alpha=0, loss_weights=[1, 1], hf_model_name=None):
        super(pl.LightningModule, self).__init__()

        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.id2label = []
        self.label2id = []
        for label_block_name in target_label_encoder.keys():
            self.labels = list(target_label_encoder[label_block_name].classes_)
            self.num_labels = len(self.labels)
            id2label_temp = {str(target_label_encoder[label_block_name].transform([label])[0]):label for label in self.labels}
            label2id_temp = {label:int(ind) for ind,label in id2label_temp.items()}
            self.id2label += [id2label_temp]
            self.label2id += [label2id_temp]

        ###### label_block_name_for_model_config is just to get the config        
        label_block_name_for_model_config = list(target_label_encoder.keys())[0]
        self.labels = list(target_label_encoder[label_block_name_for_model_config].classes_)
        self.num_labels = len(self.labels)
        #hf_model = 'allenai/longformer-base-4096' 
        #import pdb; pdb.set_trace()
        self.config = AutoConfig.from_pretrained(
            hf_model_name,
            num_labels=self.num_labels,
            id2label=self.id2label[0],
            label2id=self.label2id[0],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

        ########## Specific to multi tasking  ########################
        self.config.label_maps_id2label = self.id2label
        self.config.label_maps_label2id = self.label2id

        self.config.loss_hyperparameters = {}
        self.config.loss_hyperparameters['num_labels'] = [len(target_label_encoder[label_block_name].classes_)  for label_block_name in target_label_encoder.keys()]
        print(f'num_labels at the output are {self.config.loss_hyperparameters["num_labels"]}')
        #self.config.loss_hyperparameters['num_labels'] = [len(target_label_encoder['label'].classes_), len(target_label_encoder['label2'].classes_)]
        self.config.loss_hyperparameters['loss_types'] = [LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha) for _ in target_label_encoder.keys()]
        #                                            LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)]

        self.config.loss_hyperparameters['loss_weights'] = loss_weights
        print(f'loss_weights are { self.config.loss_hyperparameters["loss_weights"]}')
        
        ##############################################################

        #import pdb; pdb.set_trace()    
        self.model = BERTForTokenClassificationTrueCasingPunctuation(
            hf_model_name,
            from_tf='.ckpt' in hf_model_name,
            config=self.config)

        #self.tokenizer.add_special_tokens({'additional_special_tokens': [NEW_TURN]})
        #self.model.resize_token_embeddings(len(self.tokenizer))    
        self.model_input_keys = ['input_ids', 'attention_mask', 'labels', 'token_type_ids']


class BERTForTokenClassificationTrueCasingPunctuation(BertPreTrainedModel):
    def __init__(self, hf_model_name, from_tf, config):
        super().__init__(config)

        self.config = config

        self.bert = BertModel.from_pretrained(
            hf_model_name,
            from_tf=from_tf,
            config=self.config, add_pooling_layer=False)

        self.dropout = nn.ModuleList([])
        self.classifier = nn.ModuleList([])
        for ind, label_count in enumerate(self.config.loss_hyperparameters['num_labels']):
            self.dropout.append(nn.Dropout(config.hidden_dropout_prob))
            self.classifier.append(nn.Linear(config.hidden_size, label_count))

        #self.init_weights() ## TODO: Do we need this?

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):                
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #print(inputs_embeds.shape, attention_mask.shape, labels.shape)
        #import pdb; pdb.set_trace()
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = []
        total_loss = []
        for ind,label_count in enumerate(self.config.loss_hyperparameters['num_labels']):
            logits_perlabeltype = self.dropout[ind](sequence_output)
            logits_perlabeltype = self.classifier[ind](logits_perlabeltype)

            loss_fct = self.config.loss_hyperparameters['loss_types'][ind]
            loss_perlabeltype = self.compute_loss(loss_fct, logits_perlabeltype, attention_mask, labels[ind], label_count)
            logits.append(logits_perlabeltype)
            total_loss.append(loss_perlabeltype)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        #import pdb; pdb.set_trace()
        return TokenClassifierOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            last_layer_output=[outputs.last_hidden_state, outputs.last_hidden_state],  ##  last_layer_output is formed as a list and contains one tensor per task. Here both tasks have final shared output
            attentions=outputs.attentions,
            all_losses=total_loss
        )
    
    def compute_loss(self, loss_fct, logits, attention_mask, labels, num_labels):
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
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


class TopicSegSeqLevelClassifBERT_pl(pl.LightningModule):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, max_sequence_length: int, label_smoothing_alpha=0, loss_weights=[1, 1], hf_model_name=None, classification_type='SeqClassif'):
        super(pl.LightningModule, self).__init__()

        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.id2label = []
        self.label2id = []
        for label_block_name in target_label_encoder.keys():
            self.labels = list(target_label_encoder[label_block_name].classes_)
            self.num_labels = len(self.labels)
            id2label_temp = {str(target_label_encoder[label_block_name].transform([label])[0]):label for label in self.labels}
            label2id_temp = {label:int(ind) for ind,label in id2label_temp.items()}
            self.id2label += [id2label_temp]
            self.label2id += [label2id_temp]

        ###### label_block_name_for_model_config is just to get the config        
        label_block_name_for_model_config = list(target_label_encoder.keys())[0]
        self.labels = list(target_label_encoder[label_block_name_for_model_config].classes_)
        self.num_labels = len(self.labels)
        #hf_model = 'allenai/longformer-base-4096' 
        #import pdb; pdb.set_trace()
        self.config = AutoConfig.from_pretrained(
            hf_model_name,
            num_labels=self.num_labels,
            id2label=self.id2label[0],
            label2id=self.label2id[0],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

        ########## Specific to multi tasking  ########################
        self.config.label_maps_id2label = self.id2label
        self.config.label_maps_label2id = self.label2id

        self.config.loss_hyperparameters = {}
        self.config.loss_hyperparameters['num_labels'] = [len(target_label_encoder[label_block_name].classes_)  for label_block_name in target_label_encoder.keys()]
        print(f'num_labels at the output are {self.config.loss_hyperparameters["num_labels"]}')
        #self.config.loss_hyperparameters['num_labels'] = [len(target_label_encoder['label'].classes_), len(target_label_encoder['label2'].classes_)]
        self.config.loss_hyperparameters['loss_types'] = [LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha) for _ in target_label_encoder.keys()]
        #                                            LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)]

        self.config.loss_hyperparameters['loss_weights'] = loss_weights
        print(f'loss_weights are { self.config.loss_hyperparameters["loss_weights"]}')
        
        ##############################################################

        self.config.classification_type = [classification_type for _ in target_label_encoder.keys()] # SeqClassif means we are doing sequence level classification.
        # for frame-level or token-level classification use "SeqTagging"
        ## currently assumes all the tasks are of same type ( either SeqClassif or SeqTagging )

        #import pdb; pdb.set_trace()    
        self.model = TopicSegSeqLevelClassifBERT(
            hf_model_name,
            from_tf='.ckpt' in hf_model_name,
            config=self.config)

        #self.tokenizer.add_special_tokens({'additional_special_tokens': [NEW_TURN]})
        #self.model.resize_token_embeddings(len(self.tokenizer))    
        self.model_input_keys = ['input_ids', 'attention_mask', 'labels', 'token_type_ids']

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_num):
        "Compute loss and log."
        inputs = {}
        for i in self.model_input_keys:
            inputs[i] = batch[i]

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch["token_type_ids"] if self.config.model_type in ["bert", "xlnet"] else None
            )

        outputs = self(**inputs)
        #import pdb; pdb.set_trace()
        loss_per_op = outputs.loss
        wts = self.config.loss_hyperparameters['loss_weights']
        loss = 0
        for ind in range(len(wts)):
            loss += loss_per_op[ind] * wts[ind]
        tensorboard_logs = {"loss": loss}
        return {"loss": loss, "log": tensorboard_logs}        

    def validation_step(self, batch, batch_nb):
        "Compute validation"
        inputs = {}
        for i in self.model_input_keys:
            inputs[i] = batch[i]

        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch["token_type_ids"] if self.config.model_type in ["bert", "xlnet"] else None
            )
        outputs = self(**inputs)       
        #import pdb; pdb.set_trace()

        #tmp_eval_loss, logits = outputs[:2]
        tmp_eval_loss = outputs.loss
        logits = outputs.logits
        tmp_eval_loss = [tmp_eval_loss[i].detach().cpu().numpy() for i in range(len(logits))]
        preds = [logits[i].detach().cpu().numpy() for i in range(len(logits))]
        out_label_ids = [batch["labels"][i].detach().cpu().numpy() for i in range(len(logits))]
        return {"val_loss": tmp_eval_loss, "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def _eval_end(self, outputs):
        "Evaluation called for both Val and Test"
        num_op_types = len(self.config.loss_hyperparameters['num_labels'])
        if len(set(self.config.classification_type)) > 1:
            raise ValueError(f'implementation supports only one type classification either SeqTagging or SeqClassif but not both at the same time')

        if self.config.classification_type[0] == 'SeqTagging':
            outputs = pad_outputs_MultiLoss(outputs, num_op_types)

        results = {}
        ret = {}
        #import pdb; pdb.set_trace()
        for op_ind in range(num_op_types):
            val_loss_mean = np.stack([x["val_loss"][op_ind] for x in outputs]).mean()
            val_loss_mean = torch.as_tensor(val_loss_mean)
            preds = np.concatenate([x["pred"][op_ind] for x in outputs], axis=0)
            preds = np.argmax(preds, axis=-1)
            out_label_ids = np.concatenate([x["target"][op_ind] for x in outputs], axis=0)
 
            if self.config.classification_type[0] == 'SeqTagging': 
               out_label_list = [[] for _ in range(out_label_ids.shape[0])]
               preds_list = [[] for _ in range(out_label_ids.shape[0])]
       
               ## remove lobel locations not used (i.e., which have -100)
               for i in range(out_label_ids.shape[0]):
                   for j in range(out_label_ids.shape[1]):
                       if out_label_ids[i, j] != self.pad_token_label_id:
                           out_label_list[i].append(out_label_ids[i][j])
                           preds_list[i].append(preds[i][j])
            else:
                out_label_list = [list(out_label_ids)]
                preds_list = [list(preds)] # used


            if isinstance(self.config.loss_hyperparameters['loss_types'][op_ind], LabelSmoothingCrossEntropy):
                sklearn_metrics = as_tensors(compute_sklearn_metrics(out_label_list, preds_list, compute_common_I=False))
                sklearn_metrics.update({'val_loss':val_loss_mean})
                results = {i+'_op'+str(op_ind):j for i,j in sklearn_metrics.items()}
                ret.update({k: v for k, v in results.items()})
                #ret["log"] = results
            else:
                raise ValueError(f'Not implemented for targets other than categorical targets')

        #import pdb; pdb.set_trace()
        wts = self.config.loss_hyperparameters['loss_weights']
        val_loss = 0
        ret["log"] = {}
        for op_ind in range(len(wts)):
            val_loss += ret['val_loss'+'_op'+str(op_ind)] * wts[op_ind]
            ret["log"]['val_loss'+'_op'+str(op_ind)] = ret['val_loss'+'_op'+str(op_ind)]
            ret["log"]['micro_f1'+'_op'+str(op_ind)] = ret['micro_f1'+'_op'+str(op_ind)]
            ret["log"]['macro_f1'+'_op'+str(op_ind)] = ret['macro_f1'+'_op'+str(op_ind)]
            
        ret["log"]['val_loss'] = val_loss

        for key in ['macro_f1', 'micro_f1']:
            perf_list = [ret['log'][i] for i in ret.keys() if i.startswith(key)]
            perf_ave = sum(perf_list)/len(perf_list)
            ret["log"][key+'_ave'] = perf_ave

        return ret

    def validation_epoch_end(self, outputs):
        # when stable
        ret = self._eval_end(outputs)
        logs = ret["log"]
        print(logs)
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

    def test_epoch_end(self, outputs):
        # updating to test_epoch_end instead of deprecated test_end
        ret = self._eval_end(outputs)

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
            self.opt, num_warmup_steps=200, num_training_steps=self.total_steps
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def set_output_dir(self, output_dir: Path):
        self.output_dir = Path(output_dir)


class TopicSegSeqLevelClassifBERT(BertPreTrainedModel):
    def __init__(self, hf_model_name, from_tf, config):
        super().__init__(config)

        self.config = config

        self.bert = BertModel.from_pretrained(
            hf_model_name,
            from_tf=from_tf,
            config=self.config, add_pooling_layer=True)

        self.dropout = nn.ModuleList([])
        self.classifier = nn.ModuleList([])
        for ind, label_count in enumerate(self.config.loss_hyperparameters['num_labels']):
            self.dropout.append(nn.Dropout(config.hidden_dropout_prob))
            self.classifier.append(nn.Linear(config.hidden_size, label_count))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):                
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #print(inputs_embeds.shape, attention_mask.shape, labels.shape)
        #import pdb; pdb.set_trace()
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.classification_type[0] == 'SeqClassif':
            sequence_output = outputs[1] # outputs[0] would be token-level output and outputs[1] is pooled output
        elif self.config.classification_type[0] == 'SeqTagging':
            sequence_output = outputs[0]
        else:
            raise ValueError(f'not implemented for other types of classification, given self.config.classification_type is {self.config.classification_type}')
        
        logits = []
        total_loss = []
        for ind,label_count in enumerate(self.config.loss_hyperparameters['num_labels']):
            logits_perlabeltype = self.dropout[ind](sequence_output)
            logits_perlabeltype = self.classifier[ind](logits_perlabeltype)

            loss_fct = self.config.loss_hyperparameters['loss_types'][ind]
            ## probably you need to use different compute_loss function to  accommodate sequence level classification
            if self.config.classification_type[ind] == 'SeqClassif':
                loss_perlabeltype = self.compute_loss_SeqClassif(loss_fct, logits_perlabeltype, attention_mask, labels[ind], label_count)
            elif self.config.classification_type[0] == 'SeqTagging':
                loss_perlabeltype = self.compute_loss_SeqTagging(loss_fct, logits_perlabeltype, attention_mask, labels[ind], label_count)
            else:
                raise ValueError(f'not implemented for other types of classification, given self.config.classification_type is {self.config.classification_type}')
                
            logits.append(logits_perlabeltype)
            total_loss.append(loss_perlabeltype)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        #import pdb; pdb.set_trace()
        return TokenClassifierOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            last_layer_output=[outputs.last_hidden_state, outputs.last_hidden_state],  ##  last_layer_output is formed as a list and contains one tensor per task. Here both tasks have final shared output
            attentions=outputs.attentions,
            all_losses=total_loss
        )
    
    def compute_loss_SeqClassif(self, loss_fct, logits, attention_mask, labels, num_labels):
        loss = None
        if labels is not None:
            if num_labels == 1:
                #  We are doing regression
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return loss

    def compute_loss_SeqTagging(self, loss_fct, logits, attention_mask, labels, num_labels):
        loss = None
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


