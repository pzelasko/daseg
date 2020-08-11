import pytest
from transformers import RobertaForTokenClassification, \
    RobertaConfig, RobertaTokenizer

from daseg import DialogActCorpus, Call, FunctionalSegment, TransformerModel
from daseg.data import NEW_TURN


@pytest.fixture
def dummy_dataset():
    return DialogActCorpus(dialogues={
        'call0': Call([
            FunctionalSegment('Hi, how are you?', 'Conventional-Opening', 'A'),
            FunctionalSegment("I'm fine, thanks. And you?", 'Conventional-Opening', 'B'),
            FunctionalSegment("Good.", 'Conventional-Opening', 'A'),
            FunctionalSegment("It's just a test.", 'Statement-non-opinion', 'A'),
            FunctionalSegment("How do you know?", 'Wh-Question', 'B'),
        ])
    })


@pytest.fixture
def dummy_model(dummy_dataset):
    labels = dummy_dataset.joint_coding_dialog_act_labels
    config = RobertaConfig(num_labels=len(labels))
    config.id2label = {str(i): label for i, label in enumerate(labels)}
    config.label2id = {label: i for i, label in enumerate(labels)}
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.add_special_tokens({'additional_special_tokens': [NEW_TURN]})
    model = RobertaForTokenClassification(config)
    model.resize_token_embeddings(len(tokenizer))
    return TransformerModel(
        model=model,
        tokenizer=tokenizer,
        device='cpu'
    )


@pytest.mark.parametrize(
    ['batch_size', 'window_len'],
    [
        (1, None),
        (2, None),
        (1, 8),
    ]
)
def test_dummy_model_runs(dummy_dataset, dummy_model, batch_size, window_len):
    results = dummy_model.predict(
        dummy_dataset,
        batch_size=batch_size,
        window_len=window_len,
        use_joint_coding=True
    )
    assert 'dataset' in results
    assert results['dataset'].calls[0].words() == dummy_dataset.calls[0].words()
