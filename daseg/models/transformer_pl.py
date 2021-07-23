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

from daseg.models.longformer_model import LongformerForTokenClassification, LongformerForTokenClassificationEmoSpot, XFormerForTokenClassificationEmoSpot, XFormerPoolSegments, XFormerCNNOPPoolSegments, LongformerForSequenceClassification, XFormerForSeqClassification, XFormerForTokenClassificationAddSpeakerEmb, XFormerForTokenClassificationConcatSpeakerEmb, XFormerForTokenClassificationAddAveSpeakerEmb, XFormerForTokenClassificationConcatAveSpeakerEmb, XFormerForTokenClassificationConcatAveParamSpeakerEmb, XFormerForTokenClassificationConcatAveSpeakerEmbSpkrDiarize, XFormerForTokenClassificationAddAveSpeakerEmbSpkrDiarize, XFormerForTokenClassificationConcatTokenTypeEmb

from daseg.models.modeling_bert import BertForSequenceClassification, BertEncoder, BertModel, BertLayer, BertPooler, BertForFrameClassification

from daseg.data import NEW_TURN
from daseg.dataloaders.transformers import pad_array
from daseg.metrics import as_tensors, compute_sklearn_metrics
from daseg.loss_fun import LabelSmoothingCrossEntropy
from daseg.models.x_vector import _XvectorModel_Regression


class DialogActTransformer(pl.LightningModule):
    def __init__(self, labels, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0):
        super().__init__()
        self.save_hyperparameters()
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.labels = labels #list(target_label_encoder.classes_)
        self.num_labels = len(self.labels)
        #self.id2label = {str(i): label for i, label in enumerate(self.labels)},
        #self.label2id = {label: i for i, label in enumerate(self.labels)},
        #self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        #self.label2id = {label:int(ind) for ind,label in self.id2label.items()}
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
            #id2label=self.id2label,
            #label2id=self.label2id,
            id2label={str(i): label for i, label in enumerate(self.labels)},
            label2id={label: i for i, label in enumerate(self.labels)},
        )
        #self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if pre_trained_model:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_name_or_path,
                from_tf='.ckpt' in model_name_or_path,
                config=self.config)
            self.model.resize_token_embeddings(len(self.tokenizer))    
        else:
            self.config.inputs_embeds_dim = inputs_embeds_dim
            self.config.vocab_size = None
            self.config.max_position_embeddings = max_sequence_length + 2
            self.config.type_vocab_size = 2
            self.config.emospotloss_wt = emospotloss_wt
            self.config.emospot_concat = emospot_concat
            print(f'label_smoothing_alpha is {label_smoothing_alpha}')
            self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
            self.model = LongformerForTokenClassificationEmoSpot(config=self.config) 
            #import pdb; pdb.set_trace()
    

    def forward(self, **inputs):
        return self.model(**inputs)

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


#class ERCTransformerBERT(pl.LightningModule):
#    def __init__(self, labels, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0):
#        super().__init__()
#        self.save_hyperparameters()
#        self.pad_token_label_id = CrossEntropyLoss().ignore_index
#        self.labels = labels #list(target_label_encoder.classes_)
#        self.num_labels = len(self.labels)
#        #self.id2label = {str(i): label for i, label in enumerate(self.labels)},
#        #self.label2id = {label: i for i, label in enumerate(self.labels)},
#        #self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
#        #self.label2id = {label:int(ind) for ind,label in self.id2label.items()}
#        self.config = AutoConfig.from_pretrained(
#            model_name_or_path,
#            num_labels=self.num_labels,
#            #id2label=self.id2label,
#            #label2id=self.label2id,
#            id2label={str(i): label for i, label in enumerate(self.labels)},
#            label2id={label: i for i, label in enumerate(self.labels)},
#        )
#        #self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#        #model_name_or_path = 
#        if pre_trained_model:
#            self.model = AutoModelForTokenClassification.from_pretrained(
#                model_name_or_path,
#                from_tf='.ckpt' in model_name_or_path,
#                config=self.config)
#            self.model.resize_token_embeddings(len(self.tokenizer))    
#        else:
#            self.config.inputs_embeds_dim = inputs_embeds_dim
#            self.config.vocab_size = None
#            self.config.max_position_embeddings = max_sequence_length + 2
#            self.config.type_vocab_size = 2
#            self.config.emospotloss_wt = emospotloss_wt
#            self.config.emospot_concat = emospot_concat
#            print(f'label_smoothing_alpha is {label_smoothing_alpha}')
#            self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
#            self.model = BertForFrameClassification(config=self.config) 
#            #import pdb; pdb.set_trace()
   
 
class ERCTransformerBERT(DialogActTransformer):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}

        self.config = AutoConfig.from_pretrained(
            'bert-base-uncased',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        ### CONFIG CHANGE TO MAKE TRANSFORMERS SCRIPTS WORK FOR SPEECH INPUT  ############
        #self.config.speech_feats_ip_dim = inputs_embeds_dim # x-vector input
        self.config.inputs_embeds_dim = inputs_embeds_dim #128 # x-vector model op and longformer input
        self.config.vocab_size = None
        self.config.max_position_embeddings = max_sequence_length # int(np.ceil(max_sequence_length/8)) + 2
        print(f'BERT max_sequence_length is set to {self.config.max_position_embeddings}')
        attention_window_context = max_sequence_length # min(512, int(np.ceil(max_sequence_length/8)))
        self.config.attention_window = [attention_window_context for _ in range(self.config.num_hidden_layers)]
        print(f'setting self.config.attention_window to {self.config.attention_window}')
        self.config.type_vocab_size = 30 #2 #30 #2
        self.config.emospotloss_wt = emospotloss_wt
        self.config.emospot_concat = emospot_concat
        print(f'label_smoothing_alpha is {label_smoothing_alpha}')
        self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        if pre_trained_model:
            self.config.pretrained = True
            self.config.pretrained_model_path = pretrained_model_path #'/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5'
        else:
            self.config.pretrained = False
            self.config.pretrained_model_path = None
        self.model = BertForFrameClassification(config=self.config)


class XFormer(DialogActTransformer):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}

        self.config = AutoConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        ### CONFIG CHANGE TO MAKE TRANSFORMERS SCRIPTS WORK FOR SPEECH INPUT  ############
        self.config.speech_feats_ip_dim = inputs_embeds_dim # x-vector input
        self.config.inputs_embeds_dim = 128 # x-vector model op and longformer input
        self.config.vocab_size = None
        self.config.max_position_embeddings = int(np.ceil(max_sequence_length/8)) + 2
        print(f'longformer max_sequence_length is set to {self.config.max_position_embeddings}')
        attention_window_context = min(512, int(np.ceil(max_sequence_length/8)))
        self.config.attention_window = [attention_window_context for _ in range(self.config.num_hidden_layers)]
        print(f'setting self.config.attention_window to {self.config.attention_window}')
        self.config.type_vocab_size = 30 #2 #30 #2
        self.config.emospotloss_wt = emospotloss_wt
        self.config.emospot_concat = emospot_concat
        print(f'label_smoothing_alpha is {label_smoothing_alpha}')
        self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        if pre_trained_model:
            self.config.pretrained = True
            self.config.pretrained_model_path = pretrained_model_path #'/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5'
        else:
            self.config.pretrained = False
            self.config.pretrained_model_path = None
        self.model = XFormerForTokenClassificationEmoSpot(config=self.config)


class XFormerAddSpeakerEmb(DialogActTransformer):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}

        self.config = AutoConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        ### CONFIG CHANGE TO MAKE TRANSFORMERS SCRIPTS WORK FOR SPEECH INPUT  ############
        self.config.speech_feats_ip_dim = inputs_embeds_dim # x-vector input
        self.config.inputs_embeds_dim = 128 # x-vector model op and longformer input
        self.config.vocab_size = None
        self.config.max_position_embeddings = int(np.ceil(max_sequence_length/8)) + 2
        print(f'longformer max_sequence_length is set to {self.config.max_position_embeddings}')
        attention_window_context = min(512, int(np.ceil(max_sequence_length/8)))
        self.config.attention_window = [attention_window_context for _ in range(self.config.num_hidden_layers)]
        print(f'setting self.config.attention_window to {self.config.attention_window}')
        self.config.type_vocab_size = 30 #2 #30 #2
        self.config.emospotloss_wt = emospotloss_wt
        self.config.emospot_concat = emospot_concat
        print(f'label_smoothing_alpha is {label_smoothing_alpha}')
        self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        if pre_trained_model:
            self.config.pretrained = True
            self.config.pretrained_model_path = pretrained_model_path #'/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5'
        else:
            self.config.pretrained = False
            self.config.pretrained_model_path = None
        self.model = XFormerForTokenClassificationAddSpeakerEmb(config=self.config)


class XFormerAddAveSpeakerEmb(DialogActTransformer):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}

        self.config = AutoConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        ### CONFIG CHANGE TO MAKE TRANSFORMERS SCRIPTS WORK FOR SPEECH INPUT  ############
        self.config.speech_feats_ip_dim = inputs_embeds_dim # x-vector input
        self.config.inputs_embeds_dim = 128 # x-vector model op and longformer input
        self.config.vocab_size = None
        self.config.max_position_embeddings = int(np.ceil(max_sequence_length/8)) + 2
        print(f'longformer max_sequence_length is set to {self.config.max_position_embeddings}')
        attention_window_context = min(512, int(np.ceil(max_sequence_length/8)))
        self.config.attention_window = [attention_window_context for _ in range(self.config.num_hidden_layers)]
        print(f'setting self.config.attention_window to {self.config.attention_window}')
        self.config.type_vocab_size = 30 #2 #30 #2
        self.config.emospotloss_wt = emospotloss_wt
        self.config.emospot_concat = emospot_concat
        print(f'label_smoothing_alpha is {label_smoothing_alpha}')
        self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        if pre_trained_model:
            self.config.pretrained = True
            self.config.pretrained_model_path = pretrained_model_path #'/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5'
        else:
            self.config.pretrained = False
            self.config.pretrained_model_path = None
        self.model = XFormerForTokenClassificationAddAveSpeakerEmb(config=self.config)


class XFormerConcatSpeakerEmb(DialogActTransformer):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}

        self.config = AutoConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        ### CONFIG CHANGE TO MAKE TRANSFORMERS SCRIPTS WORK FOR SPEECH INPUT  ############
        self.config.speech_feats_ip_dim = inputs_embeds_dim # x-vector input
        self.config.inputs_embeds_dim = 2*128 # x-vector model op and longformer input. It is 2 times 128 because we concatenate speaker rep to x-vector model op
        self.config.vocab_size = None
        self.config.max_position_embeddings = int(np.ceil(max_sequence_length/8)) + 2
        print(f'longformer max_sequence_length is set to {self.config.max_position_embeddings}')
        attention_window_context = min(512, int(np.ceil(max_sequence_length/8)))
        self.config.attention_window = [attention_window_context for _ in range(self.config.num_hidden_layers)]
        print(f'setting self.config.attention_window to {self.config.attention_window}')
        self.config.type_vocab_size = 30 #2 #30 #2
        self.config.emospotloss_wt = emospotloss_wt
        self.config.emospot_concat = emospot_concat
        print(f'label_smoothing_alpha is {label_smoothing_alpha}')
        self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        if pre_trained_model:
            self.config.pretrained = True
            self.config.pretrained_model_path = pretrained_model_path #'/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5'
        else:
            self.config.pretrained = False
            self.config.pretrained_model_path = None
        self.model = XFormerForTokenClassificationConcatSpeakerEmb(config=self.config)


class XFormerConcatAveSpeakerEmb(DialogActTransformer):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}

        self.config = AutoConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        ### CONFIG CHANGE TO MAKE TRANSFORMERS SCRIPTS WORK FOR SPEECH INPUT  ############
        self.config.speech_feats_ip_dim = inputs_embeds_dim # x-vector input
        self.config.inputs_embeds_dim = 2*128 # x-vector model op and longformer input. It is 2 times 128 because we concatenate speaker rep to x-vector model op
        self.config.vocab_size = None
        self.config.max_position_embeddings = int(np.ceil(max_sequence_length/8)) + 2
        print(f'longformer max_sequence_length is set to {self.config.max_position_embeddings}')
        attention_window_context = min(512, int(np.ceil(max_sequence_length/8)))
        self.config.attention_window = [attention_window_context for _ in range(self.config.num_hidden_layers)]
        print(f'setting self.config.attention_window to {self.config.attention_window}')
        self.config.type_vocab_size = 30 #2 #30 #2
        self.config.emospotloss_wt = emospotloss_wt
        self.config.emospot_concat = emospot_concat
        print(f'label_smoothing_alpha is {label_smoothing_alpha}')
        self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        if pre_trained_model:
            self.config.pretrained = True
            self.config.pretrained_model_path = pretrained_model_path #'/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5'
        else:
            self.config.pretrained = False
            self.config.pretrained_model_path = None
        self.model = XFormerForTokenClassificationConcatAveSpeakerEmb(config=self.config)


class XFormerConcatAveSpeakerEmbSpkrDiarize(DialogActTransformer):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}

        self.config = AutoConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        ### CONFIG CHANGE TO MAKE TRANSFORMERS SCRIPTS WORK FOR SPEECH INPUT  ############
        self.config.speech_feats_ip_dim = inputs_embeds_dim # x-vector input
        self.config.inputs_embeds_dim = 2*128 # x-vector model op and longformer input. It is 2 times 128 because we concatenate speaker rep to x-vector model op
        self.config.vocab_size = None
        self.config.max_position_embeddings = int(np.ceil(max_sequence_length/8)) + 2
        print(f'longformer max_sequence_length is set to {self.config.max_position_embeddings}')
        attention_window_context = min(512, int(np.ceil(max_sequence_length/8)))
        self.config.attention_window = [attention_window_context for _ in range(self.config.num_hidden_layers)]
        print(f'setting self.config.attention_window to {self.config.attention_window}')
        self.config.type_vocab_size = 30 #2 #30 #2
        self.config.emospotloss_wt = emospotloss_wt
        self.config.emospot_concat = emospot_concat
        print(f'label_smoothing_alpha is {label_smoothing_alpha}')
        self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        if pre_trained_model:
            self.config.pretrained = True
            self.config.pretrained_model_path = pretrained_model_path #'/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5'
        else:
            self.config.pretrained = False
            self.config.pretrained_model_path = None
        self.model = XFormerForTokenClassificationConcatAveSpeakerEmbSpkrDiarize(config=self.config)


class XFormerAddAveSpeakerEmbSpkrDiarize(DialogActTransformer):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}

        self.config = AutoConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        ### CONFIG CHANGE TO MAKE TRANSFORMERS SCRIPTS WORK FOR SPEECH INPUT  ############
        self.config.speech_feats_ip_dim = inputs_embeds_dim # x-vector input
        self.config.inputs_embeds_dim = 128 
        self.config.vocab_size = None
        self.config.max_position_embeddings = int(np.ceil(max_sequence_length/8)) + 2
        print(f'longformer max_sequence_length is set to {self.config.max_position_embeddings}')
        attention_window_context = min(512, int(np.ceil(max_sequence_length/8)))
        self.config.attention_window = [attention_window_context for _ in range(self.config.num_hidden_layers)]
        print(f'setting self.config.attention_window to {self.config.attention_window}')
        self.config.type_vocab_size = 30 #2 #30 #2
        self.config.emospotloss_wt = emospotloss_wt
        self.config.emospot_concat = emospot_concat
        print(f'label_smoothing_alpha is {label_smoothing_alpha}')
        self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        if pre_trained_model:
            self.config.pretrained = True
            self.config.pretrained_model_path = pretrained_model_path #'/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5'
        else:
            self.config.pretrained = False
            self.config.pretrained_model_path = None
        self.model = XFormerForTokenClassificationAddAveSpeakerEmbSpkrDiarize(config=self.config)


class XFormerConcatTokenTypeEmb(DialogActTransformer):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}

        self.config = AutoConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        ### CONFIG CHANGE TO MAKE TRANSFORMERS SCRIPTS WORK FOR SPEECH INPUT  ############
        self.config.speech_feats_ip_dim = inputs_embeds_dim # x-vector input
        self.config.inputs_embeds_dim = 2*128 
        self.config.vocab_size = None
        self.config.max_position_embeddings = int(np.ceil(max_sequence_length/8)) + 2
        print(f'longformer max_sequence_length is set to {self.config.max_position_embeddings}')
        attention_window_context = min(512, int(np.ceil(max_sequence_length/8)))
        self.config.attention_window = [attention_window_context for _ in range(self.config.num_hidden_layers)]
        print(f'setting self.config.attention_window to {self.config.attention_window}')
        self.config.type_vocab_size = 30 #2 #30 #2
        self.config.emospotloss_wt = emospotloss_wt
        self.config.emospot_concat = emospot_concat
        print(f'label_smoothing_alpha is {label_smoothing_alpha}')
        self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        if pre_trained_model:
            self.config.pretrained = True
            self.config.pretrained_model_path = pretrained_model_path #'/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5'
        else:
            self.config.pretrained = False
            self.config.pretrained_model_path = None
        self.model = XFormerForTokenClassificationConcatTokenTypeEmb(config=self.config)


class XFormerCNNOPPoolSegments_pl(DialogActTransformer):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None, classwts=None):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}

        self.config = AutoConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.config.speech_feats_ip_dim = inputs_embeds_dim # x-vector input
        self.config.inputs_embeds_dim = 128 # x-vector model op and longformer input
        self.config.vocab_size = None
        self.config.max_position_embeddings = int(np.ceil(max_sequence_length/8)) + 2
        print(f'longformer max_sequence_length is set to {self.config.max_position_embeddings}')
        attention_window_context = min(512, int(np.ceil(max_sequence_length/8)))
        self.config.attention_window = [attention_window_context for _ in range(self.config.num_hidden_layers)]
        print(f'setting self.config.attention_window to {self.config.attention_window}')
        self.config.type_vocab_size = 30 #2 #30 #2
        self.config.emospotloss_wt = emospotloss_wt
        self.config.emospot_concat = emospot_concat
        print(f'label_smoothing_alpha is {label_smoothing_alpha}')
        if classwts is None:
            self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        else:
            print(f'\n since classwts are given default crossentropy loss function is used and LabelSmoothingCrossEntropy is discarded. Need to implement classwts in LabelSmoothingCrossEntropy  \n')
            print(f'\n classwts are {classwts} \n ')
            #import pdb; pdb.set_trace()
            #classwts = torch.tensor(classwts)
            self.config.loss_fun = CrossEntropyLoss(weight=classwts)

        if pre_trained_model:
            self.config.pretrained = True
            self.config.pretrained_model_path = pretrained_model_path #'/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5'
        else:
            self.config.pretrained = False
            self.config.pretrained_model_path = None
        self.model = XFormerCNNOPPoolSegments(config=self.config)


class XFormerConcatAveParamSpeakerEmb(DialogActTransformer):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None, avg_window=10):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}

        self.config = AutoConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        ### CONFIG CHANGE TO MAKE TRANSFORMERS SCRIPTS WORK FOR SPEECH INPUT  ############
        self.config.speech_feats_ip_dim = inputs_embeds_dim # x-vector input
        self.config.inputs_embeds_dim = 2*128 # x-vector model op and longformer input. It is 2 times 128 because we concatenate speaker rep to x-vector model op
        self.config.vocab_size = None
        self.config.max_position_embeddings = int(np.ceil(max_sequence_length/8)) + 2
        print(f'longformer max_sequence_length is set to {self.config.max_position_embeddings}')
        attention_window_context = min(512, int(np.ceil(max_sequence_length/8)))
        self.config.attention_window = [attention_window_context for _ in range(self.config.num_hidden_layers)]
        print(f'setting self.config.attention_window to {self.config.attention_window}')
        self.config.type_vocab_size = 30 #2 #30 #2
        self.config.emospotloss_wt = emospotloss_wt
        self.config.emospot_concat = emospot_concat
        print(f'label_smoothing_alpha is {label_smoothing_alpha}')
        self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        if pre_trained_model:
            self.config.pretrained = True
            self.config.pretrained_model_path = pretrained_model_path #'/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5'
        else:
            self.config.pretrained = False
            self.config.pretrained_model_path = None
        self.config.avg_window = avg_window
        self.model = XFormerForTokenClassificationConcatAveParamSpeakerEmb(config=self.config)


class XFormerPoolSegClassification(DialogActTransformer):
     def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None):   
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.model = XFormerPoolSegClassification_def(target_label_encoder, model_name_or_path, inputs_embeds_dim,
                        pre_trained_model, max_sequence_length, emospotloss_wt, emospot_concat, label_smoothing_alpha,
                        pretrained_model_path)
        self.pad_token_label_id = self.model.pad_token_label_id
        self.target_label_encoder = self.model.target_label_encoder
        self.label_map = self.model.label_map

        self.labels = self.model.labels
        self.num_labels = self.model.num_labels
        self.id2label = self.model.id2label
        self.label2id = self.model.label2id
        self.config = self.model.config
        self.config2 = self.model.config2


class XFormerPoolSegClassification_def(nn.Module):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool, 
                        max_sequence_length: int, emospotloss_wt=1.0, emospot_concat=False, label_smoothing_alpha=0, 
                        pretrained_model_path=None):
        super().__init__()
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}

        self.config = AutoConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.config.speech_feats_ip_dim = inputs_embeds_dim # x-vector input
        self.config.inputs_embeds_dim = 128 # x-vector model op and longformer input
        self.config.vocab_size = None
        self.config.max_position_embeddings = int(np.ceil(max_sequence_length/8)) + 2
        print(f'longformer max_sequence_length is set to {self.config.max_position_embeddings}')
        attention_window_context = min(512, int(np.ceil(max_sequence_length/8)))
        self.config.num_hidden_layers = self.config.num_hidden_layers - 2
        self.config.attention_window = [attention_window_context for _ in range(self.config.num_hidden_layers)]
        print(f'setting self.config.attention_window to {self.config.attention_window}')
        self.config.type_vocab_size = 30 #2 #30 #2
        self.config.emospotloss_wt = emospotloss_wt
        self.config.emospot_concat = emospot_concat
        print(f'label_smoothing_alpha is {label_smoothing_alpha}')
        self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        if pre_trained_model:
            self.config.pretrained = True
            self.config.pretrained_model_path = pretrained_model_path #'/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5'
        else:
            self.config.pretrained = False
            self.config.pretrained_model_path = None
        self.model_framelevel = XFormerPoolSegments(config=self.config)
        
        self.config2 = AutoConfig.from_pretrained(
            'allenai/longformer-base-4096',
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.config2.inputs_embeds_dim = self.config.hidden_size
        self.config2.vocab_size = None
        self.config2.max_position_embeddings = int(np.ceil(max_sequence_length/8)) + 2
        attention_window_context = min(512, int(np.ceil(max_sequence_length/8)))
        self.config2.num_hidden_layers = 2
        self.config2.attention_window = [attention_window_context for _ in range(self.config2.num_hidden_layers)]
        self.config2.type_vocab_size = 30 #2 #30 #2
        self.config2.emospotloss_wt = emospotloss_wt
        self.config2.emospot_concat = emospot_concat
        self.config2.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        self.model_seglevel = LongformerForTokenClassificationEmoSpot(config=self.config2)

    def forward(self, **inputs):
        op_framelevel = self.model_framelevel(**inputs)    
        op_seglevel = self.model_seglevel(**op_framelevel)
        return op_seglevel
   

def pad_outputs(outputs: Dict) -> Dict:
    max_out_len = max(x["pred"].shape[1] for x in outputs)
    for x in outputs:
        x["pred"] = pad_array(x["pred"], target_len=max_out_len, value=0)
        x["target"] = pad_array(x["target"], target_len=max_out_len, value=CrossEntropyLoss().ignore_index)
    return outputs


###############   Text #####################

class TransformerTextSeqClassification(pl.LightningModule):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool,
                        max_sequence_length: int, label_smoothing_alpha=0,
                        pretrained_model_path=None, warmup_proportion=0):
        super().__init__()
        self.save_hyperparameters()
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.warmup_proportion = warmup_proportion
        model_name_or_path = 'allenai/longformer-base-4096'

        if not target_label_encoder is None:
            self.target_label_encoder = target_label_encoder
            self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
            self.labels = self.target_label_encoder.classes_
            self.num_labels = len(self.labels)
            self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
            self.label2id = {label:int(ind) for ind,label in self.id2label.items()}
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )

        else:
            self.label_map = None
            self.num_labels = 1
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                num_labels=self.num_labels,
            )


        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.add_special_tokens({'additional_special_tokens': [NEW_TURN]})
        self.model = LongformerForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf='.ckpt' in model_name_or_path,
            config=self.config
        )
        #self.model = BertForSequenceClassification.from_pretrained(
        #self.model = AutoModelForSequenceClassification.from_pretrained(
        #    model_name_or_path,
        #    from_tf='.ckpt' in model_name_or_path,
        #    config=self.config
        #)
 
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_num):
        "Compute loss and log."
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[4] if self.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        
        outputs = self(**inputs)
        loss = outputs[0]
        tensorboard_logs = {"loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        "Compute validation"

        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        if self.config.model_type != "distilbert":
            inputs["token_type_ids"] = (
                batch[4] if self.config.model_type in ["bert", "xlnet"] else None
            )  # XLM and RoBERTa don"t use token_type_ids
        outputs = self(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def _eval_end(self, outputs):
        "Evaluation called for both Val and Test"
        #outputs = pad_outputs(outputs)
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        if self.num_labels > 1:
            preds = np.argmax(preds, axis=-1)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        out_label_list = [list(out_label_ids)]
        preds_list = [list(preds)]

        #label_map = {i: label for i, label in enumerate(self.labels)}
        #out_label_list = [[label_map[i]  for i in out_label_ids]]
        #preds_list = [[label_map[i] for i in preds]]
        if self.num_labels > 1:
            sklearn_metrics = as_tensors(compute_sklearn_metrics(out_label_list, preds_list, compute_common_I=False))
            results = {
                "val_loss": val_loss_mean,
                **sklearn_metrics
            }
        else:
            results = {"val_loss": val_loss_mean}

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
        num_warmup_steps = int(self.total_steps*self.warmup_proportion)
        print(f'\n INFO: num_warmup_steps are {num_warmup_steps}, warmup_proportion is {self.warmup_proportion}, total_steps are {self.total_steps} \n')
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
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


class TransformerTextSeqClassificationBERT(TransformerTextSeqClassification):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool,
                        max_sequence_length: int, label_smoothing_alpha=0,
                        pretrained_model_path=None, warmup_proportion=0, 
                        additional_tokens=None):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.warmup_proportion = warmup_proportion

        if not target_label_encoder is None:
            self.target_label_encoder = target_label_encoder
            self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
            self.labels = self.target_label_encoder.classes_
            self.num_labels = len(self.labels)
            self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
            self.label2id = {label:int(ind) for ind,label in self.id2label.items()}
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
            )
        else:
            self.label_map = None
            self.num_labels = 1
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                num_labels=self.num_labels,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        #self.tokenizer.add_special_tokens({'additional_special_tokens': [NEW_TURN]})

        #additional_tokens = ['sil_l', 'sil_m', 'sil_s', '[noise]', '<unk>']
        if additional_tokens is None:
            self.tokenizer.add_special_tokens({'additional_special_tokens': [NEW_TURN]})
        else:
            additional_tokens += [NEW_TURN]
            print(f'adding additional_tokens to the BERT model and they are {additional_tokens}')
            self.tokenizer.add_special_tokens({'additional_special_tokens': additional_tokens})

        self.model = BertForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf='.ckpt' in model_name_or_path,
            config=self.config
        )
        #import pdb; pdb.set_trace() 
        #self.config.type_vocab_size = 30 ## 5-6 are enough but just putting more in case we need
        self.model.resize_token_embeddings(len(self.tokenizer))
    

class TransformerMultiModalSeqClassification_pl(TransformerTextSeqClassification):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool,
                        max_sequence_length: int, label_smoothing_alpha=0,
                        pretrained_model_path=None, no_cross_att_layers=2, speech_att=False):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}
        model_name_or_path = 'bert-base-uncased'
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        #self.tokenizer.add_special_tokens({'additional_special_tokens': [NEW_TURN]})
        self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        self.config.no_cross_att_layers = no_cross_att_layers
        self.config.speech_att = speech_att # True or False
        self.model = TransformerMultiModalSeqClassification(self.config, pretrained_model_path)

    def training_step(self, batch, batch_num):
        "Compute loss and log."
        #inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        #if self.config.model_type != "distilbert":
        #    inputs["token_type_ids"] = (
        #        batch[4] if self.config.model_type in ["bert", "xlnet"] else None
        #    )  # XLM and RoBERTa don"t use token_type_ids

        #outputs = self(**inputs)
        outputs = self(**batch)
        loss = outputs[0]
        tensorboard_logs = {"loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        "Compute validation"

        #inputs = {"inputs_embeds": batch[0], "attention_mask": batch[1], "labels": batch[2], "token_type_ids":batch[4]}
        #if self.config.model_type != "distilbert":
        #    inputs["token_type_ids"] = (
        #        batch[2] if self.config.model_type in ["bert", "xlnet"] else None
        #    )  # XLM and RoBERTa don"t use token_type_ids
        #outputs = self(**inputs)
        outputs = self(**batch)
        tmp_eval_loss, logits = outputs[:2]
        preds = logits.detach().cpu().numpy()
        out_label_ids = batch["labels"].detach().cpu().numpy()
        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}


class TransformerMultiModalSeqClassification(pl.LightningModule):
    def __init__(self, config_bert, pretrained_model_path):
        super().__init__()
        self.num_labels = config_bert.num_labels
        self.loss_fun = config_bert.loss_fun
        self.config = config_bert

        self.bert = BertModel(config_bert)
        self.x_vector_proj = nn.Linear(128, config_bert.hidden_size)

        self.speech_att = self.config.speech_att
        if self.speech_att:
            config_att_speech = deepcopy(config_bert)
            self.speech_att_layer = BertLayer(config_att_speech)

        config_bert2 = deepcopy(config_bert)
        #config_bert2['num_hidden_layers'] = 2
        config_bert2.is_decoder = True
        config_bert2.add_cross_attention = True
        self.bertlayer_t2s = nn.ModuleList([])
        self.bertlayer_s2t = nn.ModuleList([])
        self.no_cross_att_layers = self.config.no_cross_att_layers
        for i in range(self.no_cross_att_layers):
            self.bertlayer_t2s.append(BertLayer(config_bert2))
            self.bertlayer_s2t.append(BertLayer(config_bert2))

        self.embed_proj = nn.Linear(2704, self.num_labels) ## 2704 = 768*3 + 400
        self.bertpooler_t2s = BertPooler(config_bert2)
        self.bertpooler_s2t = BertPooler(config_bert2)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        #self.model.resize_token_embeddings(len(self.tokenizer))

        self.x_vector_model = _XvectorModel_Regression(pretrained=True, pretrained_model=pretrained_model_path, output_channels=self.num_labels, input_dim=23)

    def forward(self,
        text_ip=None,
        text_attention_mask=None,
        text_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        speech_ip=None,
        speech_attention_mask=None,
        speech_token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        **kwargs
        ):
        ## forward args are copied from BertModel forward function and added labels as final argument
        bert_op = self.bert(
            text_ip,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_hidden_states, text_pooled_op = bert_op['last_hidden_state'], bert_op['pooler_output']

        speech_hidden_states = self.x_vector_model.model.forward_encoder(speech_ip)
        _, x_vectors = self.x_vector_model.forward_decoder(speech_hidden_states)

        subsampling_factor = 8
        speech_attention_mask = speech_attention_mask[:, ::subsampling_factor]
        speech_token_type_ids = speech_token_type_ids[:, ::subsampling_factor]
        speech_hidden_states = self.x_vector_proj(speech_hidden_states)

        ## get extended attention masks for selfattention operaion
        text_attention_mask = self.bert.get_extended_attention_mask(text_attention_mask, 
                                                    text_hidden_states.shape, 
                                                    device=text_hidden_states.device)
        speech_attention_mask = self.bert.get_extended_attention_mask(speech_attention_mask, 
                                                    speech_hidden_states.shape, 
                                                    device=speech_hidden_states.device)
        if self.speech_att:
            speech_hidden_states = self.speech_att_layer(speech_hidden_states, speech_attention_mask)
        
        #print(text_hidden_states.shape, speech_hidden_states.shape)
        #import pdb; pdb.set_trace()
        for layer_ind in range(self.no_cross_att_layers):
            text_hidden_states_copy = text_hidden_states[0] if isinstance(text_hidden_states, tuple) else text_hidden_states
            speech_hidden_states_copy = speech_hidden_states[0] if isinstance(speech_hidden_states,  tuple) else speech_hidden_states

            ## speech sequence --> text sequence
            text_hidden_states = self.bertlayer_s2t[layer_ind](    
                text_hidden_states_copy,
                attention_mask=text_attention_mask,
                encoder_hidden_states=speech_hidden_states_copy,
                encoder_attention_mask=speech_attention_mask,
                )
            ## text sequence --> speech sequence
            speech_hidden_states = self.bertlayer_t2s[layer_ind](    
                speech_hidden_states_copy,
                attention_mask=speech_attention_mask,
                encoder_hidden_states=text_hidden_states_copy,
                encoder_attention_mask=text_attention_mask,
                )
        s2t = self.bertpooler_s2t(text_hidden_states[0])
        t2s = self.bertpooler_t2s(speech_hidden_states[0])
        #t2s = t2s['pooler_output']
        ## op from all modalities
        concat_op = torch.cat([text_pooled_op, x_vectors, s2t, t2s], -1)

        #concat_op = self.dropout(concat_op)

        logits = self.embed_proj(concat_op)

        loss_fct = self.loss_fun # CrossEntropyLoss()
        loss = self.compute_loss(loss_fct, logits, None, labels, self.num_labels)
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


class TransformerMultiModalMultiLossSeqClassification_pl(TransformerTextSeqClassification):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool,
                        max_sequence_length: int, label_smoothing_alpha=0,
                        pretrained_model_path=None, no_cross_att_layers=2, speech_att=False, warmup_proportion=0,
                        loss_weights=[1,1]):
        super(pl.LightningModule, self).__init__()
        self._example_input_array = None
        self.save_hyperparameters()

        
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}
        model_name_or_path = 'bert-base-uncased'
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        #self.tokenizer.add_special_tokens({'additional_special_tokens': [NEW_TURN]})
        self.config.loss_hyperparameters = {}
        #self.config.loss_hyperparameters['output_types'] = 3
        self.config.loss_hyperparameters['num_labels'] = [2, 1] #[2, 1, 2]
        #self.config.loss_hyperparameters['loss_types'] = [LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha), nn.MSELoss(), LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)]
        self.config.loss_hyperparameters['loss_types'] = [LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha), nn.MSELoss()] #, LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)]
        self.config.loss_hyperparameters['loss_weights'] = loss_weights
        self.warmup_proportion = warmup_proportion

        #self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        self.config.no_cross_att_layers = no_cross_att_layers
        self.config.speech_att = speech_att # True or False
        self.model = TransformerMultiModalMultiLossSeqClassification(self.config, pretrained_model_path)

    def training_step(self, batch, batch_num):
        "Compute loss and log."
        outputs = self(**batch)
        loss_per_op = outputs[0]
        wts = self.config.loss_hyperparameters['loss_weights']
        loss = 0
        for ind in range(len(wts)):
            loss += loss_per_op[ind] * wts[ind]

        tensorboard_logs = {"loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        "Compute validation"

        outputs = self(**batch)
        tmp_eval_loss, logits = outputs[:2]
        tmp_eval_loss = [tmp_eval_loss[i].detach().cpu().numpy() for i in range(len(logits))]
        preds = [logits[i].detach().cpu().numpy() for i in range(len(logits))]
        out_label_ids = [batch["labels"][i].detach().cpu().numpy() for i in range(len(logits))]
        #preds = logits.detach().cpu().numpy()
        #out_label_ids = batch["labels"].detach().cpu().numpy()
        return {"val_loss": tmp_eval_loss, "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs):
        "Evaluation called for both Val and Test"
        #val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
   
        #import pdb; pdb.set_trace()
        results = {}
        ret = {}
        num_op_types = len(self.config.loss_hyperparameters['num_labels'])
        # self.config.loss_hyperparameters['loss_types']
        for op_ind in range(num_op_types):
            val_loss_mean = np.stack([x["val_loss"][op_ind] for x in outputs]).mean()
            val_loss_mean = torch.as_tensor(val_loss_mean)
            preds = np.concatenate([x["pred"][op_ind] for x in outputs], axis=0)
            preds = np.argmax(preds, axis=-1)
            out_label_ids = np.concatenate([x["target"][op_ind] for x in outputs], axis=0)
            out_label_list = [list(out_label_ids)]
            preds_list = [list(preds)]
            if isinstance( self.config.loss_hyperparameters['loss_types'][op_ind], LabelSmoothingCrossEntropy):
                sklearn_metrics = as_tensors(compute_sklearn_metrics(out_label_list, preds_list, compute_common_I=False))
                results = {
                    'val_loss': val_loss_mean,
                    **sklearn_metrics
                }
                ret.update({k: v for k, v in results.items()})
                ret["log"] = results
            else:
                ## TODO: Need to implement metrics for MMSE
                ret["log"].update({'val_loss_op2':val_loss_mean})

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

    def get_lr_scheduler(self):
        num_warmup_steps = int(self.total_steps*self.warmup_proportion)
        print(f'\n INFO: num_warmup_steps are {num_warmup_steps}, warmup_proportion is {self.warmup_proportion}, total_steps are {self.total_steps} \n')
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler


class TransformerMultiModalMultiLossSeqClassification(pl.LightningModule):
    def __init__(self, config_bert, pretrained_model_path):
        super().__init__()

        #self.output_types = config_bert.output_types
        #self.num_labels = config_bert.num_labels
        #self.num_labels = config_bert.num_labels
        #self.loss_fun = config_bert.loss_fun
        self.config = config_bert

        self.bert = BertModel(config_bert)
        self.x_vector_proj = nn.Linear(128, config_bert.hidden_size)

        self.speech_att = self.config.speech_att
        if self.speech_att:
            config_att_speech = deepcopy(config_bert)
            self.speech_att_layer = BertLayer(config_att_speech)

        config_bert2 = deepcopy(config_bert)
        #config_bert2['num_hidden_layers'] = 2
        config_bert2.is_decoder = True
        config_bert2.add_cross_attention = True
        self.bertlayer_t2s = nn.ModuleList([])
        self.bertlayer_s2t = nn.ModuleList([])
        self.no_cross_att_layers = self.config.no_cross_att_layers
        for i in range(self.no_cross_att_layers):
            self.bertlayer_t2s.append(BertLayer(config_bert2))
            self.bertlayer_s2t.append(BertLayer(config_bert2))

        self.embed_proj = nn.ModuleList([])
        if self.no_cross_att_layers == 0:
            embed_proj_dim = 400 + 768
        elif self.no_cross_att_layers > 0:
            embed_proj_dim = 768*3 + 400
            
        for ind, label_count in enumerate(self.config.loss_hyperparameters['num_labels']):
            
            self.embed_proj.append(nn.Linear(embed_proj_dim, label_count)) ## 2704 = 768*3 + 400
            
        self.bertpooler_t2s = BertPooler(config_bert2)
        self.bertpooler_s2t = BertPooler(config_bert2)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        #self.model.resize_token_embeddings(len(self.tokenizer))

        self.x_vector_model = _XvectorModel_Regression(pretrained=True, pretrained_model=pretrained_model_path, output_channels=2, input_dim=23) # output_channels does not matter here as we are not using it

    def forward(self,
        text_ip=None,
        text_attention_mask=None,
        text_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        speech_ip=None,
        speech_attention_mask=None,
        speech_token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        **kwargs
        ):
        ## forward args are copied from BertModel forward function and added labels as final argument
        bert_op = self.bert(
            text_ip,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_hidden_states, text_pooled_op = bert_op['last_hidden_state'], bert_op['pooler_output']

        speech_hidden_states = self.x_vector_model.model.forward_encoder(speech_ip)
        _, x_vectors = self.x_vector_model.forward_decoder(speech_hidden_states)

        subsampling_factor = 8
        speech_attention_mask = speech_attention_mask[:, ::subsampling_factor]
        speech_token_type_ids = speech_token_type_ids[:, ::subsampling_factor]
        speech_hidden_states = self.x_vector_proj(speech_hidden_states)

        ## get extended attention masks for selfattention operaion
        text_attention_mask = self.bert.get_extended_attention_mask(text_attention_mask, 
                                                    text_hidden_states.shape, 
                                                    device=text_hidden_states.device)
        speech_attention_mask = self.bert.get_extended_attention_mask(speech_attention_mask, 
                                                    speech_hidden_states.shape, 
                                                    device=speech_hidden_states.device)
        if self.speech_att:
            speech_hidden_states = self.speech_att_layer(speech_hidden_states, speech_attention_mask)
        
        #print(text_hidden_states.shape, speech_hidden_states.shape)
        #import pdb; pdb.set_trace()
        if self.no_cross_att_layers > 0:
            for layer_ind in range(self.no_cross_att_layers):
                text_hidden_states_copy = text_hidden_states[0] if isinstance(text_hidden_states, tuple) else text_hidden_states
                speech_hidden_states_copy = speech_hidden_states[0] if isinstance(speech_hidden_states,  tuple) else speech_hidden_states
    
                ## speech sequence --> text sequence
                text_hidden_states = self.bertlayer_s2t[layer_ind](    
                    text_hidden_states_copy,
                    attention_mask=text_attention_mask,
                    encoder_hidden_states=speech_hidden_states_copy,
                    encoder_attention_mask=speech_attention_mask,
                    )
                ## text sequence --> speech sequence
                speech_hidden_states = self.bertlayer_t2s[layer_ind](    
                    speech_hidden_states_copy,
                    attention_mask=speech_attention_mask,
                    encoder_hidden_states=text_hidden_states_copy,
                    encoder_attention_mask=text_attention_mask,
                    )
            s2t = self.bertpooler_s2t(text_hidden_states[0])
            t2s = self.bertpooler_t2s(speech_hidden_states[0])
            #t2s = t2s['pooler_output']
            ## op from all modalities
            concat_op = torch.cat([text_pooled_op, x_vectors, s2t, t2s], -1)
        else:
            concat_op = torch.cat([text_pooled_op, x_vectors], -1)

        #concat_op = self.dropout(concat_op)
        #self.config.loss_hyperparameters['num_labels'] = [2, 1, 2]
        #self.config.loss_hyperparameters['loss_types'] = [LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha), nn.MSELoss(), LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)]
        #import pdb; pdb.set_trace()
        logits = []
        total_loss = []
        for ind,label_count in enumerate(self.config.loss_hyperparameters['num_labels']):    
            logits_perlabeltype = self.embed_proj[ind](concat_op)
            loss_fct = self.config.loss_hyperparameters['loss_types'][ind]

            loss_perlabeltype = self.compute_loss(loss_fct, logits_perlabeltype, None, labels[ind], label_count)
            logits.append(logits_perlabeltype)
            total_loss.append(loss_perlabeltype)

        tb_loss = None #torch.Tensor((loss,))

        return TokenClassifierOutput(
            loss=total_loss,
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




class TransformerSpeechSeqClassification(pl.LightningModule):
    def __init__(self, target_label_encoder, model_name_or_path: str, inputs_embeds_dim: int, pre_trained_model: bool,
                        max_sequence_length: int, label_smoothing_alpha=0,
                        pretrained_model_path=None):
        ''' emospotloss_wt and emospot_concat arguments are removed as EmoSpot is not in focus currently
        '''
        super().__init__()
        self.save_hyperparameters()
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.target_label_encoder = target_label_encoder
        self.label_map = {self.target_label_encoder.transform([i])[0]:i for i in self.target_label_encoder.classes_}
        self.labels = self.target_label_encoder.classes_
        self.num_labels = len(self.labels)
        self.id2label = {str(target_label_encoder.transform([label])[0]):label for label in self.labels}
        self.label2id = {label:int(ind) for ind,label in self.id2label.items()}

        model_name_or_path = 'allenai/longformer-base-4096'
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        ### CONFIG CHANGE TO MAKE TRANSFORMERS SCRIPTS WORK FOR SPEECH INPUT  ############
        self.config.speech_feats_ip_dim = inputs_embeds_dim # x-vector input
        self.config.inputs_embeds_dim = 128 # x-vector model op and longformer input
        self.config.vocab_size = None
        self.config.max_position_embeddings = int(np.ceil(max_sequence_length/8)) + 2
        print(f'longformer max_sequence_length is set to {self.config.max_position_embeddings}')
        attention_window_context = min(512, int(np.ceil(max_sequence_length/8)))
        self.config.attention_window = [attention_window_context for _ in range(self.config.num_hidden_layers)]
        print(f'setting self.config.attention_window to {self.config.attention_window}')
        self.config.type_vocab_size = 30
        #self.config.emospotloss_wt = -100
        #self.config.emospot_concat = False
        print(f'label_smoothing_alpha is {label_smoothing_alpha}')
        self.config.loss_fun = LabelSmoothingCrossEntropy(epsilon=label_smoothing_alpha)
        if pre_trained_model:
            self.config.pretrained = True
            self.config.pretrained_model_path = pretrained_model_path #'/export/b15/rpapagari/kaldi_21Aug2019/egs/sre16/Emotion_xvector_ICASSP2020_ComParE_v2/pretrained_xvector_models/model_aug_xvector.h5'
        else:
            self.config.pretrained = False
            self.config.pretrained_model_path = None
        self.model = XFormerForSeqClassification(config=self.config)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_num):
        "Compute loss and log."
        inputs = {"inputs_embeds": batch[0], "attention_mask": batch[1], "labels": batch[2], "token_type_ids":batch[4]}
        #if self.config.model_type != "distilbert":
        #    inputs["token_type_ids"] = (
        #        batch[2] if self.config.model_type in ["bert", "xlnet"] else None
        #    )  # XLM and RoBERTa don"t use token_type_ids

        outputs = self(**inputs)
        loss = outputs[0]
        tensorboard_logs = {"loss": loss}
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
        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()
        return {"val_loss": tmp_eval_loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def _eval_end(self, outputs):
        "Evaluation called for both Val and Test"
        #outputs = pad_outputs(outputs)
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        preds = np.argmax(preds, axis=-1)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        label_map = {i: label for i, label in enumerate(self.labels)}
        out_label_list = [[label_map[i]  for i in out_label_ids]]
        preds_list = [[label_map[i] for i in preds]]

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
            self.opt, num_warmup_steps=0, num_training_steps=self.total_steps
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


