import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LongformerModel, LongformerConfig, BertPreTrainedModel, \
    add_start_docstrings, LONGFORMER_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.file_utils import add_start_docstrings_to_callable
from transformers.modeling_roberta import ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING


@add_start_docstrings(
    """Roberta Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    ROBERTA_START_DOCSTRING,
)
class LongformerForTokenClassification(BertPreTrainedModel):
    config_class = LongformerConfig
    pretrained_model_archive_map = LONGFORMER_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "longformer"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = LongformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_input_embeddings(self):
        return self.roberta.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        return self.roberta.set_input_embeddings(value)

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForTokenClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

        """

        try:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        except:
            # Models trained with Transformers v3.5 don't accept "head_mask" argument
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

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

# class LongformerForTokenClassification(RobertaForTokenClassification):
#     def __init__(self, config):
#         super().__init__(config)
#         if config.attention_mode == 'n2':
#             pass  # do nothing, use BertSelfAttention instead
#         else:
#             for i, layer in enumerate(self.roberta.encoder.layer):
#                 layer.attention.self = LongformerSelfAttention(config, layer_id=i)
#
#
# class LongformerCRFForTokenClassification(LongformerForTokenClassification):
#     def __init__(self, config):
#         super().__init__(config)
#         self.crf = ConditionalRandomField(
#             num_tags=self.num_labels,
#             constraints=allowed_transitions(
#                 constraint_type='IOB1',
#                 labels={int(id_): label for id_, label in config.id2label.items()}
#             ),
#         )
#
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             inputs_embeds=None,
#             labels=None,
#     ):
#         outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
#                                   position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
#                                   labels=labels)
#         ce_loss = outputs[0]
#         logits = outputs[1]
#         if labels is not None:
#             # TODO: fix ugly label-set dependent hack
#             labels[labels < 0] = 91
#             crf_loss = -self.crf(logits, labels, attention_mask)
#
#             outputs = (ce_loss, crf_loss) + outputs[1:]
#         return outputs
