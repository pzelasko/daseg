from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from longformer.longformer import LongformerSelfAttention
from transformers import RobertaForTokenClassification


class LongformerForTokenClassification(RobertaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        if config.attention_mode == 'n2':
            pass  # do nothing, use BertSelfAttention instead
        else:
            for i, layer in enumerate(self.roberta.encoder.layer):
                layer.attention.self = LongformerSelfAttention(config, layer_id=i)


class LongformerCRFForTokenClassification(LongformerForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.crf = ConditionalRandomField(
            num_tags=self.num_labels,
            constraints=allowed_transitions(
                constraint_type='IOB1',
                labels={int(id_): label for id_, label in config.id2label.items()}
            ),
        )

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
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                  position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                                  labels=labels)
        ce_loss = outputs[0]
        logits = outputs[1]
        if labels is not None:
            # TODO: fix ugly label-set dependent hack
            labels[labels < 0] = 91
            crf_loss = -self.crf(logits, labels, attention_mask)

            outputs = (ce_loss, crf_loss) + outputs[1:]
        return outputs
