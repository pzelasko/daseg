from functools import lru_cache

import pytest
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

from daseg import SwdaDataset, Call, FunctionalSegment, TransformerModel
from daseg.data import NEW_TURN


@lru_cache(1)
def dummy_dataset():
    return SwdaDataset(dialogues={
        'call0': Call([
            FunctionalSegment('Hi, how are you?', 'Conventional-Opening', 'A'),
            FunctionalSegment("I'm fine, thanks. And you?", 'Conventional-Opening', 'B'),
            FunctionalSegment("Good.", 'Conventional-Opening', 'A'),
            FunctionalSegment("It's just a test.", 'Statement-non-opinion', 'A'),
            FunctionalSegment("How do you know?", 'Wh-Question', 'B'),
        ])
    })


@lru_cache(1)
def dummy_model(model_type='xlnet-base-cased'):
    dataset = dummy_dataset()
    labels = dataset.dialog_act_labels
    config = AutoConfig.from_pretrained(
        model_type,
        num_labels=len(labels),
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.add_special_tokens({'additional_special_tokens': [NEW_TURN]})
    model = AutoModelForTokenClassification.from_pretrained(
        model_type,
        config=config,
    )
    model.resize_token_embeddings(len(tokenizer))
    return TransformerModel(
        model=model,
        tokenizer=tokenizer
    )


@pytest.mark.parametrize(
    ['batch_size', 'window_len', 'crf_decoding'],
    [
        (1, None, False),
        (2, None, False),
        (1, None, True),
        (1, 8, False),
        (1, 8, True)
    ]
)
def test_dummy_model_runs(batch_size, window_len, crf_decoding):
    dataset = dummy_dataset()
    model = dummy_model()
    model.predict(
        dataset,
        batch_size=batch_size,
        window_len=window_len,
        crf_decoding=crf_decoding
    )
