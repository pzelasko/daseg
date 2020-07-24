from functools import lru_cache
from typing import Dict

import spacy

__all__ = ['get_nlp', 'get_tokenizer', 'SWDA_BUGGY_DIALOG_ACTS']


@lru_cache(1)
def get_nlp():
    return spacy.load("en_core_web_sm")


@lru_cache(1)
def get_tokenizer():
    nlp = get_nlp()
    return nlp.Defaults.create_tokenizer(nlp)


SEGMENT_TAG = 's'
SEGMENTATION_ONLY_ACTS = {
    SEGMENT_TAG: 'Segment'
}

SWDA_DIALOG_ACT_TO_TAG = {
    "Statement-non-opinion": "sd",
    "Acknowledge-Backchannel": "b",
    "Statement-opinion": "sv",
    "Agree-Accept": "aa",
    "Abandoned-or-Turn-Exit": "%",
    "Appreciation": "ba",
    "Yes-No-Question": "qy",
    "Non-verbal": "x",
    "Yes-answers": "ny",
    "Conventional-closing": "fc",
    "Uninterpretable": "%",
    "Wh-Question": "qw",
    "No-answers": "nn",
    "Response-Acknowledgement": "bk",
    "Hedge": "h",
    "Declarative-Yes-No-Question": "qy^d",
    # Replace: "Other": "fo_o_fw_by_bc" with the following as it appears like that in SWDA
    "Other": 'fo_o_fw_"_by_bc',
    "Backchannel-in-question-form": "bh",
    "Quotation": "^q",
    "Summarize/reformulate": "bf",
    "Affirmative-non-yes-answers": "na",
    "Action-directive": "ad",
    "Collaborative-Completion": "^2",
    "Repeat-phrase": "b^m",
    "Open-Question": "qo",
    "Rhetorical-Questions": "qh",
    "Hold-before-answer-agreement": "^h",
    "Reject": "ar",
    "Negative-non-no-answers": "ng",
    "Signal-non-understanding": "br",
    "Other-answers": "no",
    "Conventional-opening": "fp",
    "Or-Clause": "qrr",
    "Dispreferred-answers": "arp_nd",
    "3rd-party-talk": "t3",
    "Offers-Options-Commits": "oo_co_cc",
    "Self-talk": "t1",
    "Downplayer": "bd",
    "Maybe-Accept-part": "aap_am",
    "Tag-Question": "^g",
    "Declarative-Wh-Question": "qw^d",
    "Apology": "fa",
    "Thanking": "ft",
    "+": "+"
}

SWDA_TAG_TO_DIALOG_ACT = {value: key for key, value in SWDA_DIALOG_ACT_TO_TAG.items()}


def to_buggy_swda_42_labels(dialog_acts: Dict[str, str]) -> Dict[str, str]:
    reduced_dialog_acts = dialog_acts.copy()
    reduced_dialog_acts.update({
        'sd^e': 'Other',
        'fe': 'Other',
        'qe': "Other",
        'qr': 'Other'
    })
    return reduced_dialog_acts


SWDA_BUGGY_DIALOG_ACTS = {
    'sd': 'Statement-non-opinion',
    'b': 'Acknowledge-Backchannel',
    'sv': 'Statement-opinion',
    'aa': 'Agree/Accept',
    '% -': 'Abandoned-or-Turn-Exit',
    'ba': 'Appreciation',
    'qy': 'Yes-No-Question',
    'x': 'Non-verbal',
    'ny': 'Yes-answers',
    'fc': 'Conventional-closing',
    '%': 'Uninterpretable',
    'qw': 'Wh-Question',
    'nn': 'No-answers',
    'bk': 'Response-Acknowledgement',
    'h': 'Hedge',
    'qy^d': 'Declarative-Yes-No-Question',
    'o': 'Other',
    'fo': 'Other',
    'bc': 'Other',
    'by': 'Other',
    'fw': 'Other',
    'bh': 'Backchannel-in-question-form',
    '^q': 'Quotation',
    'bf': 'Summarize/reformulate',
    'na': 'Affirmative-non-yes-answers',
    'ny^e': 'Affirmative-non-yes-answers',
    'ad': 'Action-directive',
    '^2': 'Collaborative-Completion',
    'b^m': 'Repeat-phrase',
    'qo': 'Open-Question',
    'qh': 'Rhetorical-Questions',
    '^h': 'Hold-before-answer/agreement',
    'ar': 'Reject',
    'ng': 'Negative-non-no-answers',
    'nn^e': 'Negative-non-no-answers',
    'br': 'Signal-non-understanding',
    'no': 'Other-answers',
    'fp': 'Conventional-opening',
    'qrr': 'Or-Clause',
    'arp': 'Dispreferred-answers',
    'nd': 'Dispreferred-answers',
    't3': '3rd-party-talk',
    'oo': 'Offers/Options/Commits',
    'cc': 'Offers/Options/Commits',
    'co': 'Offers/Options/Commits',
    't1': 'Self-talk',
    'bd': 'Downplayer',
    'aap': 'Maybe/Accept-part',
    'am': 'Maybe/Accept-part',
    '^g': 'Tag-Question',
    'qw^d': 'Declarative-Wh-Question',
    'fa': 'Apology',
    'ft': 'Thanking',
    '+': '+',
    'sd^e': 'Statement-expanding-y/n-answer',
    'fe': 'Exclamation',
    'qe': "Alternative-or-question",
    'qr': 'Or-question'
}

MRDA_BASIC_DIALOG_ACTS = {
    'S': 'Statement',
    'B': 'Backchannel',
    'D': 'Disruption',
    'F': 'Floor-grabber',
    'Q': 'Question'
}

MRDA_GENERAL_DIALOG_ACTS = {
    's': 'Statement',
    'b': 'Continuer',
    'fh': 'Floor-Holder',
    'qy': 'Yes-No-question',
    '%': 'Interrupted-Abandoned-Uninterpretable',
    'fg': 'Floor-Grabber',
    'qw': 'Wh-Question',
    'h': 'Hold-Before-Answer-Agreement',
    'qrr': 'Or-clause',
    'qh': 'Rhetorical-question',
    'qr': 'Or-question',
    'qo': 'Open-ended-question'
}

MRDA_FULL_DIALOG_ACTS = {
    's': 'Statement',
    'b': 'Continuer',
    'fh': 'Floor-Holder',
    'bk': 'Acknowledge-answer',
    'aa': 'Accept',
    'df': 'Defending-Explanation',
    'e': 'Expansions-of-y-n-Answers',
    '%': 'Interrupted-Abandoned-Uninterpretable',
    'rt': 'Rising-Tone',
    'fg': 'Floor-Grabber',
    'cs': 'Offer',
    'ba': 'Assessment-Appreciation',
    'bu': 'Understanding-Check',
    'd': 'Declarative-Question',
    'na': 'Affirmative-Non-yes-Answers',
    'qw': 'Wh-Question',
    'ar': 'Reject',
    '2': 'Collaborative-Completion',
    'no': 'Other-Answers',
    'h': 'Hold-Before-Answer-Agreement',
    'co': 'Action-directive',
    'qy': 'Yes-No-question',
    'nd': 'Dispreferred-Answers',
    'j': 'Humorous-Material',
    'bd': 'Downplayer',
    'cc': 'Commit',
    'ng': 'Negative-Non-no-Answers',
    'am': 'Maybe',
    'qrr': 'Or-Clause',
    'fe': 'Exclamation',
    'm': 'Mimic-Other',
    'fa': 'Apology',
    't': 'About-task',
    'br': 'Signal-non-understanding',
    'aap': 'Accept-part',
    'qh': 'Rhetorical-Question',
    'tc': 'Topic-Change',
    'r': 'Repeat',
    't1': 'Self-talk',
    't3': '3rd-party-talk',
    'bh': 'Rhetorical-question-Continue',
    'bsc': 'Reject-part',
    'arp': 'Misspeak-Self-Correction',
    'bs': 'Reformulate-Summarize',
    'f': '"Follow-Me"',
    'qr': 'Or-Question',
    'ft': 'Thanking',
    'g': 'Tag-Question',
    'qo': 'Open-Question',
    'bc': 'Correct-misspeaking',
    'by': 'Sympathy',
    'fw': 'Welcome'
}

COLORMAP = [
    '#66c2a5',
    '#fc8d62',
    '#8da0cb',
    '#e78ac3',
    '#a6d854',
    '#ffd92f',
    '#e5c494',
    '#b3b3b3',
    '#66c2a5',
    '#fc8d62',
    '#8da0cb',
    '#e78ac3',
    '#a6d854',
    '#ffd92f',
    '#e5c494',
    '#b3b3b3',
    '#66c2a5',
    '#fc8d62',
    '#8da0cb',
    '#e78ac3',
    '#a6d854',
    '#ffd92f',
    '#e5c494',
    '#b3b3b3',
    '#66c2a5',
    '#fc8d62',
    '#8da0cb',
    '#e78ac3',
    '#a6d854',
    '#ffd92f',
    '#e5c494',
    '#b3b3b3',
    '#66c2a5',
    '#fc8d62',
    '#8da0cb',
    '#e78ac3',
    '#a6d854',
    '#ffd92f',
    '#e5c494',
    '#b3b3b3',
    '#66c2a5',
    '#fc8d62',
    '#8da0cb',
    '#e78ac3',
    '#a6d854'
]
