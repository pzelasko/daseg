from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from more_itertools import flatten
from torch import nn
from torch.nn import CrossEntropyLoss

from daseg.conversion import joint_coding_predictions_to_corpus
from daseg.metrics import compute_sklearn_metrics, compute_zhao_kawahara_metrics, compute_original_zhao_kawahara_metrics


class ZhaoKawaharaBiGru(pl.LightningModule):
    def __init__(
            self,
            vocab: Dict[str, int],
            labels: Dict[str, int],
            label_frequencies: Optional[Dict[str, int]] = None,
            word_embed_dim=200,
            gru_dim=100,
            loss_reduction='mean',
            weight_drop: Optional[float] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.weight_drop = weight_drop
        self.vocab = vocab
        self.labels = labels
        self.vocab_size = len(vocab) + 1  # OOV token
        self.labels_size = len(labels)

        self.word_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=word_embed_dim, padding_idx=0)
        self.utterance_gru = nn.GRU(
            input_size=word_embed_dim,
            hidden_size=gru_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(in_features=2 * gru_dim, out_features=self.labels_size)
        self.micro_f1 = pl.metrics.sklearns.F1(labels=list(self.labels), average='micro')
        self.macro_f1 = pl.metrics.sklearns.F1(labels=list(self.labels), average='macro')
        label_weights = None
        if label_frequencies is not None:
            label_weights = (
                    1. / (torch.tensor([label_frequencies[key] for key in labels], dtype=torch.float32) / sum(
                label_frequencies.values()))

            )
        self.loss = nn.CrossEntropyLoss(weight=label_weights, reduction=loss_reduction)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.word_embedding.weight, -1.0, 1.0)
        # nn.init.uniform_(self.tag_embedding.weight, -1.0, 1.0)
        for name, weight in self.utterance_gru.named_parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_uniform_(weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.uniform_(self.classifier.weight, -0.08, 0.08)
        # other weights: uniform [-0.08, 0.08]

    def forward(self, word_indices, text_lengths):
        embedded = self.word_embedding(word_indices)
        total_length = embedded.size(1)  # get the max sequence length
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            text_lengths,
            batch_first=True
        )
        packed_gru_outputs, _ = self.utterance_gru(packed_embedded)
        gru_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_gru_outputs,
            batch_first=True,
            total_length=total_length
        )
        logits = self.classifier(gru_outputs)
        return logits

    def training_step(self, batch, batch_idx):
        word_indices, text_lengths, act_indices = batch

        # WEIGHT DROP - replace original weights with "dropped out" version
        if self.weight_drop is not None and 0 < self.weight_drop < 1:
            with torch.no_grad():
                orig_params = []
                for n, p in self.utterance_gru.named_parameters():
                    orig_params.append(p.clone())
                    p.data = nn.functional.dropout(p.data, p=self.weight_drop) * (1 - self.weight_drop)
                self.utterance_gru.flatten_parameters()

        logits = self(word_indices, text_lengths).transpose(1, 2)

        # WEIGHT DROP - replace "dropped out" version with the original weights
        if self.weight_drop is not None and 0 < self.weight_drop < 1:
            with torch.no_grad():
                for orig_p, (n, p) in zip(orig_params, self.utterance_gru.named_parameters()):
                    p.data = orig_p.data

        loss = self.loss(input=logits, target=act_indices)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        if self.weight_drop is not None:
            self.utterance_gru.flatten_parameters()
        return {}

    def _common_step(self, batch, batch_idx, prefix):
        word_indices, text_lengths, act_indices = batch
        logits = self(word_indices, text_lengths).transpose(1, 2)
        loss = self.loss(input=logits, target=act_indices)
        return {
            f'{prefix}_loss': loss,
            'logits': logits.detach(),
            'true_labels': act_indices
        }

    def _common_epoch_end(self, outputs):
        aggregated_metrics = {key: 0 for key in outputs[0] if key not in {'logits', 'true_labels'}}
        for output in outputs:
            for key in aggregated_metrics:
                aggregated_metrics[key] += output[key]
        for key in aggregated_metrics:
            aggregated_metrics[key] /= len(outputs)

        metrics = self.compute_metrics(
            logits=[o['logits'] for o in outputs],
            true_labels=[o['true_labels'] for o in outputs]
        )
        aggregated_metrics.update(metrics)

        results = {
            'progress_bar': aggregated_metrics,
            'log': aggregated_metrics
        }
        return results

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, prefix='val')

    def validation_epoch_end(self, outputs):
        return self._common_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, prefix='test')

    def test_epoch_end(self, outputs):
        return self._common_epoch_end(outputs)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.0001)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optim, factor=0.5, patience=1,
        )
        return [optim], [sched]

    def compute_metrics(self, logits, true_labels):

        # true_labels = np.concatenate(true_labels, axis=0)
        # logits = np.concatenate(logits, axis=0)
        true_labels = list(flatten(true_labels))
        logits = list(flatten(logits))
        preds = [torch.argmax(l, dim=0) for l in logits]
        pad_token_label_id = CrossEntropyLoss().ignore_index
        label_map = {i: label for label, i in self.labels.items()}

        out_label_list = [[] for _ in true_labels]
        preds_list = [[] for _ in true_labels]

        for i in range(len(true_labels)):
            jmax = true_labels[i].shape[0]
            for j in range(jmax):
                if true_labels[i][j] != pad_token_label_id:
                    out_label_list[i].append(label_map[true_labels[i][j].item()])
                    preds_list[i].append(label_map[preds[i][j].item()])

        results = compute_sklearn_metrics(out_label_list, preds_list)
        metrics = compute_zhao_kawahara_metrics(
            true_dataset=joint_coding_predictions_to_corpus(out_label_list),
            pred_dataset=joint_coding_predictions_to_corpus(preds_list)
        )
        metrics.update({k: results[k] for k in ('micro_f1', 'macro_f1')})

        # We show the metrics obtained with Zhao-Kawahara code which computes them differently
        # (apparently the segment insertion errors are not counted)
        original_zhao_kawahara_metrics = compute_original_zhao_kawahara_metrics(
            true_turns=out_label_list, pred_turns=preds_list
        )
        for k in original_zhao_kawahara_metrics:
            metrics[f'ZK_{k}'] = original_zhao_kawahara_metrics[k]

        return metrics
