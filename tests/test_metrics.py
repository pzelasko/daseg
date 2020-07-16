from daseg import DialogActCorpus, Call, FunctionalSegment
from daseg.metrics import compute_zhao_kawahara_metrics


def test_zhao_kwahara_metrics():
    true_dataset = DialogActCorpus(dialogues={
        'call1': Call([
            FunctionalSegment('a b c', dialog_act='sd', speaker='A'),
            FunctionalSegment('a b c', dialog_act='ad', speaker='A'),
            FunctionalSegment('a b c', dialog_act='h', speaker='A'),
            FunctionalSegment('a b', dialog_act='qy', speaker='A'),
        ])
    })

    pred_dataset = DialogActCorpus(dialogues={
        'call1': Call([
            FunctionalSegment('a b c', dialog_act='sd', speaker='A'),
            FunctionalSegment('a b c a', dialog_act='ad', speaker='A'),
            FunctionalSegment('b c', dialog_act='h', speaker='A'),
            FunctionalSegment('a b', dialog_act='qy^d', speaker='A'),
        ])
    })

    metrics = compute_zhao_kawahara_metrics(true_dataset=true_dataset, pred_dataset=pred_dataset)
    assert metrics['DSER'] == 2 / 4
    assert metrics['SegmentationWER'] == 6 / 11
    assert metrics['DER'] == 3 / 4
    assert metrics['JointWER'] == 8 / 11
