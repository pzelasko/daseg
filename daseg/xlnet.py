import torch
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import XLNetPreTrainedModel, XLNetModel, add_start_docstrings
from transformers.file_utils import add_start_docstrings_to_callable
from transformers.modeling_xlnet import XLNET_START_DOCSTRING, XLNET_INPUTS_DOCSTRING


@add_start_docstrings(
    """XLNet Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    XLNET_START_DOCSTRING,
)
class XLNetCRFForTokenClassification(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = XLNetModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = ConditionalRandomField(
            num_tags=self.num_labels,
            constraints=allowed_transitions(
                constraint_type='IOB1',
                labels={int(id_): label for id_, label in config.id2label.items()}
            ),
        )
        self.init_weights()

    @add_start_docstrings_to_callable(XLNET_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.XLNetConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:(batch_size, config.num_labels)`):
            Classification scores (before SoftMax).
        mems (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
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

        from transformers import XLNetTokenizer, XLNetForTokenClassification
        import torch

        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        model = XLNetForTokenClassification.from_pretrained('xlnet-large-cased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        scores = outputs[0]

        """

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)

        best_paths = self.crf.viterbi_tags(logits, attention_mask, top_k=1)

        outputs = (logits,) + outputs[1:]  # Keep mems, hidden states, attentions if there are in it
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

            # Add negative log-likelihood as loss
            labels[labels < 0] = 91
            crf_loss = -self.crf(logits, labels, attention_mask)

            outputs = (loss + crf_loss,) + outputs + (best_paths,)

        return outputs  # return (loss), logits, (mems), (hidden states), (attentions), (best_paths)
