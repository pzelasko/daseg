"""
Utilities for retrieving, manipulating and storing the dialog act data.
"""
import random
import re
from functools import partial
from itertools import groupby, chain
from typing import NamedTuple, Tuple, List, FrozenSet, Iterable, Dict, Optional, Callable

import torch
from spacy import displacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from swda import Transcript, CorpusReader
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from daseg.resources import DIALOG_ACTS, COLORMAP, get_nlp
from daseg.splits import SWDA_SPLITS

__all__ = ['FunctionalSegment', 'Call', 'SwdaDataset']

# Symbol used in SWDA to indicate that the dialog act is the same as in previous turn
from daseg.utils_ner import InputExample, convert_examples_to_features

CONTINUATION = '+'

# Marks beginning of a new tag;
# Without it, we lose the segmentation information when an act is repeated
# and when an act is continued in the next turn
BEGIN_TAG = 'B-'

# Marks this is a continuation of the same act
CONTINUE_TAG = 'I-'

# Blank symbol just for the NEW_TURN symbol
BLANK = 'O'

# Marks new turn for sequence tagging models (e.g. "I am here <TURN> Okay")
NEW_TURN = '<TURN>'


class SwdaDataset:
    def __init__(self, dialogues: Dict[str, 'Call']):
        self.dialogues = dialogues

    @staticmethod
    def from_path(
            swda_path: str,
            splits=('train', 'dev', 'test'),
            strip_punctuation_and_lowercase: bool = False
    ) -> 'SwdaDataset':
        cr = CorpusReader(swda_path)
        selected_calls = frozenset(chain.from_iterable(SWDA_SPLITS[split] for split in splits))
        dialogues = dict(
            map(
                partial(parse_transcript, strip_punctuation_and_lowercase=strip_punctuation_and_lowercase),
                filter(
                    lambda tr: decode_swda_id(tr) in selected_calls,
                    cr.iter_transcripts(display_progress=False)
                )
            )
        )
        return SwdaDataset(dialogues)

    @staticmethod
    def from_transformers_predictions(preds_file_path: str) -> 'SwdaDataset':
        return read_transformers_preds(preds_file_path)

    @property
    def call_ids(self) -> List[str]:
        return list(self.dialogues.keys())

    @property
    def calls(self) -> List['Call']:
        return list(self.dialogues.values())

    @property
    def dialog_acts(self) -> List[str]:
        return sorted(set(segment.dialog_act for call in self.calls for segment in call))

    @property
    def dialog_act_labels(self) -> List[str]:
        return list(chain.from_iterable(
            (f'{BEGIN_TAG}{da}', f'{CONTINUE_TAG}{da}') for da in self.dialog_acts
        )) + [BLANK]

    @property
    def vocabulary(self) -> List[str]:
        return sorted(set(w for call in self.calls for segment in call for w in segment.text.split()))

    def special_symbols(self) -> FrozenSet[str]:
        """Return the set of symbols in brackets found in SWDA transcripts."""
        uniq_words = {w for segments in self.dialogues.values() for text, _, _, _ in segments for w in text.split()}
        special_symbols = {re.sub(r'[\?\.,!;:]', '', w) for w in uniq_words if w.startswith('<')}
        return frozenset(special_symbols)

    def acts_with_examples(self):
        from cytoolz import groupby
        return {
            act: list(zip(*group))[1]
            for act, group in groupby(
                lambda tpl: tpl[0],
                sorted(
                    (segment.dialog_act, segment.text)
                    for call in self.calls
                    for segment in call
                )
            ).items()
        }

    def train_dev_test_split(self) -> Dict[str, 'SwdaDataset']:
        return {name: self.subset(indices) for name, indices in SWDA_SPLITS.items()}

    def subset(self, selected_ids: Iterable[str]) -> 'SwdaDataset':
        selected_ids = set(selected_ids)
        dialogues = {call_id: segments for call_id, segments in self.dialogues.items() if call_id in selected_ids}
        return SwdaDataset(dialogues)

    def dump_for_transformers_ner(
            self,
            path: str,
            acts_count_per_sample: Optional[int] = None,
            acts_count_overlap: Optional[int] = None,
            continuations_allowed: bool = False,
    ):
        """
        Write this dataset to a text file used by Transformers NER recipe.
        Optionally, split the original conversations into smaller chunks with
        `acts_count_per_sample` dialog acts in each sample, and
        `acts_count_overlap` dialog acts overlap between samples in each call.
        """
        with open(path, 'w') as f:
            for call in tqdm(self.calls, desc='Calls'):
                call_windows = prepare_call_windows(
                    call=call,
                    acts_count_per_sample=acts_count_per_sample,
                    acts_count_overlap=acts_count_overlap,
                )
                for window in tqdm(call_windows, desc='Windows (if requested)', leave=False):
                    lines = to_transformers_ner_dataset(window, continuations_allowed=continuations_allowed)
                    for line in lines:
                        print(line, file=f)
                    print(file=f)

    def to_transformers_ner_format(
            self,
            tokenizer: PreTrainedTokenizer,
            max_seq_length: int,
            model_type: str,
            batch_size: int,
            labels: Iterable[str]
    ) -> DataLoader:
        """
        Convert the DA dataset into a PyTorch DataLoader for inference.
        :param tokenizer: Transformers pre-trained tokenizer object
        :param max_seq_length: self-explanatory
        :param model_type: string describing Transformers model type (e.g. xlnet, xlmroberta, bert, ...)
        :param batch_size: self-explanatory
        :param labels: you might run into problems if this is a subset of larger dataset which doesn't cover every label
            - use this arg to supply the full list of labels
        :return: PyTorch DataLoader
        """
        # tokenizer.add_special_tokens({'additional_special_tokens': [NEW_TURN]})

        ner_examples = []
        for idx, call in enumerate(self.calls):
            # This does some unnecessary back-and-forth but it's convenient
            lines = to_transformers_ner_dataset(call)
            words, tags = zip(*[l.split() for l in lines])
            ner_examples.append(InputExample(guid=idx, words=words, labels=tags))

        # The following lines are basically a copy-paste of Transformer's NER code
        # TODO: It could be modified to create "ragged" batches for faster CPU inference
        ner_features = convert_examples_to_features(
            examples=ner_examples,
            label_list=labels,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            cls_token_at_end=bool(model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
            pad_token_label_id=CrossEntropyLoss().ignore_index,
        )

        all_input_ids = torch.tensor([f.input_ids for f in ner_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in ner_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in ner_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in ner_features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        dataloader = DataLoader(
            dataset=dataset,
            sampler=SequentialSampler(dataset),
            batch_size=batch_size
        )

        return dataloader


class Call(List['FunctionalSegment']):
    def words(self, add_turn_token: bool = True) -> List[str]:
        ws = [w for seg in self for w in seg.text.split() + ([NEW_TURN] if add_turn_token else [])]
        if add_turn_token:
            ws = ws[:-1]
        return ws

    def words_with_tags(self) -> List[Tuple[str, str]]:
        return [(w, seg.dialog_act) for seg in self for w in seg.text.split()]

    def render(self, max_turns=None, jupyter=True):
        """Render the call as annotated with dialog acts in a Jupyter notebook"""

        # Render DAs options
        nlp = get_nlp()
        rand = random.Random(0)
        colors = COLORMAP[50:]
        rand.shuffle(colors)
        cmap = {k.upper(): col for k, col in zip(DIALOG_ACTS.values(), reversed(colors))}
        displacy_opts = {"colors": cmap}

        # Convert text to Doc
        rendered_htmls = []
        speakers = {'A:': 'B:', 'B:': 'A:'}
        spk = 'A:'
        for turn_no, (key, group) in enumerate(groupby(self, key=lambda tpl: tpl[2])):
            if max_turns is not None and turn_no > max_turns:
                break
            group = list(group)
            words = ' '.join(t for t, _, _, _ in group).split()
            doc = Doc(nlp.vocab, words=[spk] + words)

            ents = []
            begin = 1
            for (t, act, _, _) in group:
                words = t.split()
                end = begin + len(words)
                ents.append(Span(doc, begin, end, label=act or 'None'))
                begin = end
            doc.ents = ents

            rendered_htmls.append(displacy.render(doc, style="ent", jupyter=jupyter, options=displacy_opts))

            spk = speakers[spk]
        return rendered_htmls

    @property
    def turns(self) -> Iterable[Tuple[str, List['FunctionalSegment']]]:
        for name, group in groupby(self, key=lambda fs: fs.speaker):
            yield name, list(group)

    def dialog_act_spans(self, include_label: bool = True) -> Iterable[Tuple[int, int, str]]:
        idx = 0
        for segment in self:
            n_toks = len(segment.text.split())
            if include_label:
                yield idx, idx + n_toks, segment.dialog_act
            else:
                yield idx, idx + n_toks
            idx += n_toks


def prepare_call_windows(
        call: Call,
        acts_count_per_sample: Optional[int],
        acts_count_overlap: Optional[int],
) -> List[Call]:
    call_windowing = acts_count_per_sample is not None and acts_count_overlap is not None
    if call_windowing:
        step_size = acts_count_per_sample - acts_count_overlap
        indices = []
        for begin in range(0, len(call), step_size):
            end = begin + acts_count_per_sample
            if end > len(call):
                step_back = end - len(call)
                rng = (begin - step_back, end - step_back)
                if len(indices) and rng != indices[-1]:
                    indices.append(rng)
                break
            indices.append((begin, end))
        # Handle the final window - if it's shorter, extend its beginning (more overlap but simpler to work with)
        call_windows = [call[b: e] for b, e in indices]
    else:
        call_windows = [call]
    return call_windows


class FunctionalSegment(NamedTuple):
    text: str
    dialog_act: Optional[str]
    speaker: str
    is_continuation: bool = False


def parse_transcript(swda_tr: Transcript, strip_punctuation_and_lowercase: bool = False) -> Tuple[str, Call]:
    normalize_text = create_text_normalizer(strip_punctuation_and_lowercase)
    call_id = decode_swda_id(swda_tr)
    segments = (
        FunctionalSegment(
            text=normalize_text(' '.join(utt.text_words(filter_disfluency=True))),
            dialog_act=lookup_or_fix(utt.act_tag),
            speaker=utt.caller
        ) for utt in swda_tr.utterances
    )
    # Remove segments which became empty as a result of text normalization (i.e. have no text, just punctuation)
    characters = re.compile(r'[a-zA-Z]+')
    segments = (seg for seg in segments if characters.search(seg.text))
    # Resolve '+' into dialog act
    resolved_segments = []
    prev_tag = {'A': 'Other', 'B': 'Other'}  # there seems to be exactly one case where the first DA is '+'
    for text, tag, spk, _ in segments:
        is_continuation = tag == CONTINUATION
        resolved_tag = prev_tag[spk] if is_continuation else tag
        resolved_segments.append(
            FunctionalSegment(
                text=text,
                dialog_act=resolved_tag,
                speaker=spk,
                is_continuation=is_continuation
            )
        )
        prev_tag[spk] = resolved_tag
    return call_id, Call(resolved_segments)


def lookup_or_fix(tag: str) -> str:
    if tag in DIALOG_ACTS:
        return DIALOG_ACTS[tag]
    # https://web.stanford.edu/~jurafsky/ws97/manual.august1.html
    # "We did the clustering by removing the secondary carat-dimensions (^2,^g,^m,^r,^e,^q,^d), with 5 exceptions"
    # (PZ): I added ^t to this list, docs say "about-task",
    #       they used it to indicate which sentences are "on-topic" in SWBD calls
    fixed_tag = re.sub(r'(.+)(\^2|\^g|\^m|\^r|\^e|\^q|\^d|\^t)', '\\1', tag)
    # "we folded the few examples of nn^e into ng, and ny^e into na"
    fixed_tag = 'ng' if tag == 'nn^e' else 'na' if tag == 'ny^e' else fixed_tag
    # (PZ): 'sd(^q)' is very frequent, sv(^q) somewhat frequent;
    #       'sd' is statement-non-opinion, 'sv' statement-opinion,
    #       but upon closer inspection they all seem to be quotes
    fixed_tag = '^q' if any(tag == name for name in ('sd(^q)', 'sv(^q)')) else fixed_tag
    if fixed_tag in DIALOG_ACTS:
        return DIALOG_ACTS[fixed_tag]
    return 'Other'


def decode_swda_id(transcript: Transcript) -> str:
    return f"sw{transcript.swda_filename.split('_')[2].split('.')[0]}"


"""
Transformers IO specific methods.
"""


def to_transformers_ner_dataset(call: List, continuations_allowed: bool = True) -> List[str]:
    """
    Convert a list of functional segments into text representations,
    used by the Transformers library to train NER models.
    """
    lines = []
    prev_spk = None
    prev_tag = {'A': None, 'B': None}
    for utt, tag, who, is_continuation in call:
        tag = '-'.join(tag.split()) if tag is not None else tag
        tokens = utt.split()
        if tag is None:
            labels = [BLANK] * len(tokens)
        elif continuations_allowed and is_continuation:
            labels = [f'{CONTINUE_TAG}{prev_tag[who]}'] * len(tokens)
        else:
            labels = [f'{BEGIN_TAG}{tag}'] + [f'{CONTINUE_TAG}{tag}'] * (len(tokens) - 1)
        if prev_spk is not None and prev_spk != who:
            tokens = [NEW_TURN] + tokens
            labels = [BLANK] + labels
        for token, label in zip(tokens, labels):
            lines.append(f'{token} {label}')
        prev_spk = who
        prev_tag[who] = tag
    return lines


def is_begin_act(tag):
    return tag.startswith(BEGIN_TAG)


def is_continued_act(tag):
    return tag.startswith(CONTINUE_TAG)


def decode_act(tag):
    if is_begin_act(tag) or is_continued_act(tag):
        return tag[2:]
    return tag


def read_transformers_preds(preds_path: str) -> 'SwdaDataset':
    lines = (l.strip() for l in open(preds_path))

    def calls(lines):
        predictions = []
        for line in lines:
            if not line:
                yield predictions
                predictions = []
                continue
            predictions.append(line)
        if predictions:
            yield predictions

    def turns(call):
        turn = []
        for line in call:
            if NEW_TURN in line or not line:
                yield turn
                turn = []
                continue
            turn.append(line)
        if turn:
            yield turn

    def segments(turn):
        segment = []
        prev_tag = None
        for line in turn:
            text, tag = line.split()
            if prev_tag is None or tag == prev_tag or (
                    is_begin_act(prev_tag) and is_continued_act(tag) and decode_act(prev_tag) == decode_act(tag)):
                segment.append((text, tag))
            else:
                yield segment
                segment = [(text, tag)]
            prev_tag = tag
        if segment:
            yield segment

    resolved_calls = []
    for call in calls(lines):
        resolved_segments = []
        speaker = 'A'
        for turn in turns(call):
            for segment in segments(turn):
                first_tag = segment[0][1]
                resolved_segments.append(
                    FunctionalSegment(
                        text=' '.join(w for w, _ in segment),
                        dialog_act=decode_act(first_tag),
                        speaker=speaker,
                        is_continuation=is_begin_act(first_tag)
                    )
                )
            speaker = {'A': 'B', 'B': 'A'}[speaker]
        resolved_calls.append(Call(resolved_segments))
    # TODO: resolve correct call ids
    return SwdaDataset({str(i): c for i, c in zip(range(1000000), resolved_calls)})


def create_text_normalizer(strip_punctuation_and_lowercase: bool = False) -> Callable[[str], str]:
    remove_patterns = list(map(
        re.compile,
        [
            r'<<[^>]*>>',  # <<talks to another person>>
            r'<[^>]*>',  # <noise>
            r'\(\([^)]*\)\)',  # ((Hailey)), what is that?
            r'\([^)]*\)',  # ... ?
            r'\(',  # unbalanced parentheses
            r'\)',  #
            r'#',  # comments?
        ]
    ))

    if strip_punctuation_and_lowercase:
        remove_patterns.append(
            re.compile(r'[!"#$%&()*+,./:;<=>?@\[\\\]^_`{|}~]')
        )

    remove_leading_nontext = re.compile(r'^[^a-zA-Z]+([a-zA-Z])')
    correct_punctuation_whitespace = re.compile(r' ([.,?!])')
    wild_dashes = re.compile(r'(\s-+\s|-+$)')

    def normalize(text: str) -> str:
        for p in remove_patterns:
            text = p.sub('', text)
        text = text.split('*')[0].strip()  # Comments after asterisk
        text = remove_leading_nontext.sub(r'\1', text)  # ". . Hi again." => "Hi again."
        text = correct_punctuation_whitespace.sub(r'\1', text)  # "Hi Jack ." -> "Hi Jack."
        if strip_punctuation_and_lowercase:
            text = wild_dashes.sub(' ', text).strip()
            text = text.lower()
        return text

    return normalize
