# conda install pandas numpy matplotlib seaborn
# pip install jupyterlab tqdm kaldialign ipywidgets jupyterlab_widgets transformers cytoolz datasets seqeval
import pickle
import re
import string
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from tqdm.auto import tqdm

from daseg import DialogActCorpus


class Example(TypedDict):
    words: List[str]
    upper_words: List[str]
    norm_words: List[str]
    punct: List[str]
    is_upper: List[bool]
    speaker: str


class PunctuationData(TypedDict):
    train: List[Example]
    dev: List[Example]
    test: List[Example]
    idx2punct: Dict[int, str]
    punct2idx: Dict[str, int]
    vocab: List[str]


@lru_cache(20000)
def is_special_token(word: str) -> bool:
    return any(
        word.startswith(ltok) and word.endswith(rtok)
        for ltok, rtok in ['[]', '<>']
    )


@lru_cache(20000)
def split_punctuation_from_word(word: str, _pattern=re.compile(r'(\w|[\[\]<>])')) -> Tuple[str, str]:
    # this silly _pattern will correctly handle "[NOISE]." -> "[NOISE]", "."
    if not word:
        return '', ''
    word = word[::-1]  # hi. -> .ih
    match = _pattern.search(word)
    if match is not None:
        first_non_punc_idx = match.span()[0]
        text = word[first_non_punc_idx:][::-1]
        punc = word[:first_non_punc_idx][::-1]
        return text, punc
    else:
        # pure punctuation
        return '', word


def preprocess_punctuation(
        text: str,
        _precedences=['?', '!', '...', '.', ',', '--', ';'],
) -> str:
    text = text.replace('"', '')
    text.replace('+', '')
    text = re.sub(r'--+', '--', text)
    words = text.split()
    norm_words = []
    for w in words:
        w, punc = split_punctuation_from_word(w)
        for sym in _precedences:
            if sym in punc:
                norm_words.append(f'{w}{sym}')
                break
        else:
            norm_words.append(w)
    return ' '.join(norm_words)


def create_example(
        text: str,
        _punctuation=string.punctuation.replace("'", ""),
        _special=re.compile(r'\[\[.*?\]\]|\[.*?\]|<.*?>', )
) -> Optional[Example]:
    """
    Converts a text segment / utterance into a dict that
    can be used for punctuation/truecasing model training/eval.

    .. code-block:: python

        {
            'words': List[str],
            'upper_words': List[str],
            'norm_words': List[str],
            'punct': List[str],
            'is_upper': List[bool],
            'speaker': str
        }

    :param text:
    :param _punctuation: list of punctuation symbols (default is globally cached).
    :param _special: regex pattern for detecting special symbols like [UNK] or <unk> (default is globally cached).
    :return: see above.
    """
    text = text.replace(' --', '--').replace(' ...', '...').strip()  # stick punctuation to the text
    text = _special.sub('', text)
    text = ' '.join(text.split())

    text_base, text_punct = split_punctuation_from_word(text)
    if not text or not text_base:
        return None

    # get rid of pesky punctuations like "hey...?!;" -> "hey?"
    text = preprocess_punctuation(text)
    # rich words and lower/no-punc words
    words = text.split()
    norm_words = [w.lower().translate(str.maketrans('', '', _punctuation)) for w in words]
    # filter out the words that consisted only of punctuation
    idx_to_remove = []
    for idx, (w, nw) in enumerate(zip(words, norm_words)):
        if not nw:
            idx_to_remove.append(idx)
    norm_words = [
        w if not is_special_token(split_punctuation_from_word(words[idx])[0]) else
        split_punctuation_from_word(words[idx])[0]
        for idx, w in enumerate(norm_words)
        if idx not in idx_to_remove
    ]
    words = [
        w
        for idx, w in enumerate(words)
        if idx not in idx_to_remove
    ]
    upper_words, labels = zip(*(split_punctuation_from_word(w) for w in words))
    is_upper = [not is_special_token(w) and any(c.isupper() for c in w) for w in upper_words]
    return {
        'words': words,
        'upper_words': upper_words,
        'norm_words': norm_words,
        'punct': labels,
        'is_upper': is_upper
    }


def train_dev_test_split(
        texts: List[Example],
        train_portion: float = 0.9,
        dev_portion: float = 0.05
) -> Dict[str, List[Example]]:
    train_part = round(len(texts) * train_portion)
    dev_part = round(len(texts) * dev_portion)
    data = {
        'train': texts[:train_part],
        'dev': texts[train_part:train_part + dev_part],
        'test': texts[train_part + dev_part:]
    }
    return data


def add_vocab_and_labels(
        data: Dict[str, Any],
        texts: Dict[str, List[Example]]
) -> PunctuationData:
    """Pickle structure:
    {
        'train': [
            {
                # Each dict represents a single turn
                'words': List[str],
                'upper_words': List[str],
                'norm_words': List[str],
                'punct': List[str],
                'is_upper': List[bool],
                'speaker': str
            }
        ],
        'dev': [...],
        'test': [...],
        'idx2punct': Dict[int, str],
        'punct2idx': Dict[str, int],
        'vocab': List[str]
    }
    """
    puncts = {p for conv in texts for t in conv for p in t['punct']};
    puncts
    idx2punct = list(puncts)
    punct2idx = {p: idx for idx, p in enumerate(puncts)}
    vocab = sorted({p for conv in texts for t in conv for p in t['norm_words']})
    data.update({
        'idx2punct': idx2punct,
        'punct2idx': punct2idx,
        'vocab': vocab
    })
    return data


"""Corpus specific parts"""

CLSP_FISHER_PATHS = (
    Path('/export/corpora3/LDC/LDC2004T19'),
    Path('/export/corpora3/LDC/LDC2005T19')
)


def prepare_no_timing(txo: Path) -> Optional[List[Tuple[str, str]]]:
    try:
        lines = txo.read_text().splitlines()
        turns = (l.split() for l in lines if l.strip())
        turns = ((t[0][0], ' '.join(t[1:])) for t in turns)  # (speaker, text)
        turns = [
            {**data, 'speaker': speaker}
            for speaker, data in
            ((speaker, create_example(text)) for speaker, text in turns)
            if data is not None
        ]
        return turns
    except Exception as e:
        print(f'Error processing path: {txo} -- {e}')
        return None


def prepare_fisher(
        paths: Sequence[Path] = CLSP_FISHER_PATHS,
        output_path: Path = Path('fisher.pkl')
) -> Dict[str, Any]:
    txos = list(
        tqdm(
            chain.from_iterable(
                path.rglob('*.txo') for path in paths,
            ),
            desc='Scanning for Fisher transcripts'
        )
    )
    texts = [
        t
        for t in (
            prepare_no_timing(txo)
            for txo in tqdm(txos, desc='Processing txos')
        )
        if t is not None
    ]
    data = train_dev_test_split(texts)
    data = add_vocab_and_labels(data, texts)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    return data


def prepare_swda(
        corpus: DialogActCorpus,
        output_path: Path = Path('swda.pkl')
) -> List[Tuple[str, str]]:
    data = {}
    splits = corpus.train_dev_test_split()
    for key, split in splits.items():
        calls = []
        for call in split.calls:
            turns = []
            for speaker, turn in call.turns:
                text = ' '.join(fs.text for fs in turn)
                turns.append({
                    **create_example(text),
                    'speaker': speaker
                })
            calls.append(turns)
        data[key] = calls
    texts = sum(data.values(), [])
    data = add_vocab_and_labels(data, texts)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    return data
