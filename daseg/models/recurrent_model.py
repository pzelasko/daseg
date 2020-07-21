import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class RNNForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = 1
        self.config = config

        self.num_labels = config.num_labels

        # all dimensions, number of layers, dropout prob, etc. same as for Transformers
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.rnn = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size,
                           num_layers=config.num_hidden_layers, dropout=config.hidden_dropout_prob, bidirectional=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        embeddings = self.embedding(input_ids)

        outputs = self.rnn(embeddings)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[1:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
