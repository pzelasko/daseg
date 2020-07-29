"""
Utilities for retrieving, manipulating and storing the dialog act data.
"""
import pickle
import re
from collections import Counter
from functools import partial
from itertools import groupby, chain
from pathlib import Path
from typing import NamedTuple, Tuple, List, FrozenSet, Iterable, Dict, Optional, Callable, Mapping, Set

from cytoolz.itertoolz import sliding_window
from more_itertools import flatten
from spacy import displacy
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from tqdm.autonotebook import tqdm

from daseg.resources import SWDA_BUGGY_DIALOG_ACTS, COLORMAP, get_nlp, to_buggy_swda_42_labels, MRDA_BASIC_DIALOG_ACTS, \
    MRDA_GENERAL_DIALOG_ACTS, MRDA_FULL_DIALOG_ACTS, SEGMENTATION_ONLY_ACTS, SEGMENT_TAG, SWDA_TAG_TO_DIALOG_ACT
from daseg.splits import SWDA_SPLITS

__all__ = ['FunctionalSegment', 'Call', 'DialogActCorpus']

# Symbol used in SWDA to indicate that the dialog act is the same as in previous turn

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

OOV = '<UNK>'


class DialogActCorpus:
    def __init__(self, dialogues: Dict[str, 'Call'], splits: Optional[Dict[str, List[str]]] = None):
        self.dialogues = dialogues
        self.splits = splits

    @staticmethod
    def from_path(
            dataset_path: str,
            splits=('train', 'dev', 'test'),
            strip_punctuation_and_lowercase: bool = False,
            tagset: str = 'basic',
            merge_continuations: bool = False
    ):
        """Infers whether the dataset is a pickle, or raw SWDA/MRDA and loads it."""
        if dataset_path.endswith('.pkl'):
            with open(dataset_path, 'rb') as f:
                corpus = pickle.load(f)
                if isinstance(corpus, DialogActCorpus):
                    subsets = [data for key, data in corpus.train_dev_test_split().items() if key in splits]
                    new_splits = {key: val for key, val in corpus.splits.items() if key in splits}
                    return DialogActCorpus(
                        dialogues={id_: call for data in subsets for id_, call in data.dialogues.items()},
                        splits=new_splits
                    )
                else:
                    raise ValueError(
                        f'The object found in the pickle is not an instance of DialogActCorpus (type: {type(corpus)})')
        if 'swda' in dataset_path:
            return DialogActCorpus.from_swda_path(
                swda_path=dataset_path,
                splits=splits,
                strip_punctuation_and_lowercase=strip_punctuation_and_lowercase,
                tagset=tagset,
                merge_continuations=merge_continuations
            )
        if 'mrda' in dataset_path:
            return DialogActCorpus.from_mrda_path(
                mrda_path=dataset_path,
                splits=splits,
                strip_punctuation_and_lowercase=strip_punctuation_and_lowercase,
                tagset=tagset
            )
        raise ValueError("Cannot infer the corpus from the path!")

    @staticmethod
    def from_mrda_path(
            mrda_path: str,
            splits=('train', 'dev', 'test'),
            strip_punctuation_and_lowercase: bool = False,
            tagset: str = 'basic',
    ) -> 'DialogActCorpus':
        # (PZ) The code below is not super clean, I mostly copied and adapted the usage from
        # https://github.com/NathanDuran/MRDA-Corpus
        from mrda.mrda_utilities import load_text_data, get_da_maps
        from mrda.process_transcript import process_transcript

        normalize_text = create_text_normalizer(strip_punctuation_and_lowercase)

        da_lookup = {
            'basic': MRDA_BASIC_DIALOG_ACTS,
            'general': MRDA_GENERAL_DIALOG_ACTS,
            'full': MRDA_FULL_DIALOG_ACTS,
            'segmentation': SEGMENTATION_ONLY_ACTS
        }[tagset]

        mrda_path = Path(mrda_path)
        archive_dir = mrda_path / 'mrda_archive'
        data_dir = mrda_path / 'mrda_data'
        metadata_dir = data_dir / 'metadata'

        # Load training, test, validation and development splits
        split_to_meetings = {
            'train': load_text_data(metadata_dir / 'train_split.txt'),
            'dev': load_text_data(metadata_dir / 'val_split.txt'),
            'test': load_text_data(metadata_dir / 'test_split.txt')
        }
        meetings_to_read = []
        for split in splits:
            meetings_to_read.extend(split_to_meetings[split])

        # Excluded dialogue act tags i.e. x = Non-verbal and z = Non-labeled
        excluded_tags = ['x', 'z']
        # Excluded characters for ignoring i.e. '=='
        excluded_chars = {'<', '>', '(', ')', '-', '#', '|', '=', '@'}

        da_map = get_da_maps(metadata_dir / 'basic_da_map.txt')
        transcript_list = (archive_dir / 'transcripts').glob('*')

        dialogues = {}
        for meeting in transcript_list:
            # Get the id for this meeting
            meeting_name = str(meeting.name.split('.')[0])
            if meeting_name not in meetings_to_read:
                continue
            # Get the transcript and database file
            transcript = load_text_data(archive_dir / 'transcripts' / f'{meeting_name}.trans', verbose=False)
            database = load_text_data(archive_dir / 'database' / f'{meeting_name}.dadb', verbose=False)
            # Process the utterances and create a dialogue object
            raw_dialogue = process_transcript(transcript, database, da_map, excluded_chars, excluded_tags)
            call = Call([
                FunctionalSegment(
                    text=normalize_text(utterance.text),
                    dialog_act=da_lookup[
                        {
                            'basic': utterance.basic_da_label,
                            'general': utterance.general_da_label,
                            'full': (  # in the paper they say "Rising Tone" is not a dialog act
                                utterance.full_da_label
                                if utterance.full_da_label != 'rt'
                                else utterance.general_da_label
                            ),
                            'segmentation': SEGMENT_TAG
                        }[tagset]
                    ],
                    speaker=utterance.speaker
                )
                for utterance in raw_dialogue.utterances
            ])
            dialogues[meeting_name] = call

        return DialogActCorpus(dialogues, splits=split_to_meetings)

    @staticmethod
    def from_swda_path(
            swda_path: str,
            splits=('train', 'dev', 'test'),
            strip_punctuation_and_lowercase: bool = False,
            merge_continuations: bool = False,
            tagset: str = 'basic',
    ) -> 'DialogActCorpus':
        from swda import CorpusReader
        cr = CorpusReader(swda_path)
        selected_calls = frozenset(chain.from_iterable(SWDA_SPLITS[split] for split in splits))
        dialogues = dict(
            map(
                partial(
                    parse_swda_transcript,
                    strip_punctuation_and_lowercase=strip_punctuation_and_lowercase,
                    merge_continuations=merge_continuations,
                    tagset=tagset
                ),
                filter(
                    lambda tr: decode_swda_id(tr) in selected_calls,
                    cr.iter_transcripts(display_progress=False)
                )
            )
        )
        return DialogActCorpus(dialogues)

    @property
    def call_ids(self) -> List[str]:
        return list(self.dialogues.keys())

    @property
    def calls(self) -> List['Call']:
        return list(self.dialogues.values())

    @property
    def turns(self) -> Iterable[List['FunctionalSegment']]:
        for call in self.dialogues.values():
            yield from (segments for spk, segments in call.turns)

    @property
    def dialog_acts(self) -> List[str]:
        return sorted(set(segment.dialog_act for call in self.calls for segment in call))

    @property
    def dialog_act_labels(self) -> List[str]:
        return list(chain.from_iterable(
            (f'{BEGIN_TAG}{da}', f'{CONTINUE_TAG}{da}') for da in self.dialog_acts
        )) + [BLANK]

    @property
    def dialog_act_frequencies(self):
        return Counter(segment.dialog_act for call in self.calls for segment in call)

    @property
    def joint_coding_dialog_act_label_frequencies(self):
        return Counter(
            segment.dialog_act if idx == 0 else CONTINUE_TAG
            for call in self.calls
            for segment in call
            for idx, word in enumerate(segment.text.split())
        )

    @property
    def joint_coding_dialog_act_labels(self) -> List[str]:
        return list(chain([BLANK, CONTINUE_TAG], self.dialog_acts))

    @property
    def vocabulary(self) -> Counter:
        return Counter(w for call in self.calls for segment in call for w in segment.text.split())

    def with_limited_vocabulary(self, n_most_common: int = 10000) -> 'DialogActCorpus':
        vocab = {w for w, n in self.vocabulary.most_common(n_most_common)}
        return DialogActCorpus(
            dialogues={call_id: call.with_vocabulary(vocab) for call_id, call in self.dialogues.items()},
            splits=self.splits
        )

    def __len__(self) -> int:
        return len(self.dialogues)

    def search(
            self,
            dialog_act: str,
            left_segments_context: int = 5,
            right_segments_context: int = 5
    ) -> Iterable['Call']:
        for call in self.calls:
            padded_segments = chain([None] * left_segments_context, call, [None] * right_segments_context)
            for segments_window in sliding_window(1 + left_segments_context + right_segments_context, padded_segments):
                segment = segments_window[left_segments_context]
                if segment.dialog_act == dialog_act:
                    yield Call([s for s in segments_window if s is not None])

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

    def train_dev_test_split(self) -> Dict[str, 'DialogActCorpus']:
        if self.splits is not None:
            return {
                split: self.subset(call_ids)
                for split, call_ids in self.splits.items()
            }
        # Otherwise, assume this is SWDA
        return {name: self.subset(indices) for name, indices in SWDA_SPLITS.items()}

    def subset(self, selected_ids: Iterable[str]) -> 'DialogActCorpus':
        selected_ids = set(selected_ids)
        dialogues = {call_id: segments for call_id, segments in self.dialogues.items() if call_id in selected_ids}
        return DialogActCorpus(dialogues)

    def dump_for_transformers_ner(
            self,
            path: str,
            acts_count_per_sample: Optional[int] = None,
            acts_count_overlap: Optional[int] = None,
            continuations_allowed: bool = False,
            use_joint_coding: bool = True,
            use_turns: bool = False
    ):
        """
        Write this dataset to a text file used by Transformers NER recipe.
        Optionally, split the original conversations into smaller chunks with
        `acts_count_per_sample` dialog acts in each sample, and
        `acts_count_overlap` dialog acts overlap between samples in each call.
        """
        with open(path, 'w') as f:
            for call in tqdm(self.calls, desc='Calls'):
                if use_turns:
                    samples = (turn for speaker, turn in call.turns)
                else:
                    samples = prepare_call_windows(
                        call=call,
                        acts_count_per_sample=acts_count_per_sample,
                        acts_count_overlap=acts_count_overlap,
                    )
                for sample in tqdm(samples, desc='Windows/turns (if requested)', leave=False):
                    lines = to_transformers_ner_dataset(
                        sample,
                        continuations_allowed=continuations_allowed,
                        use_joint_coding=use_joint_coding
                    )
                    for line in lines:
                        print(line, file=f)
                    print(file=f)


class Call(List['FunctionalSegment']):
    def words(self, add_turn_token: bool = True) -> List[str]:
        words, tags = self.words_with_tags(add_turn_token=add_turn_token)
        return words

    def speakers(self, add_turn_token: bool = True) -> List[str]:
        words, tags, speakers = self.words_with_metadata(add_turn_token=add_turn_token)
        return speakers

    def words_with_metadata(
            self,
            add_turn_token: bool = True,
            continuations_allowed: bool = True,
            use_joint_coding: bool = False
    ) -> Tuple[List[str], List[str], List[str]]:
        encoded_segments = self.encode(
            use_joint_coding=use_joint_coding,
            continuations_allowed=continuations_allowed,
            add_turn_token=add_turn_token,
        )
        words = list(flatten(segment.words for segment in encoded_segments))
        tags = list(flatten(segment.encoded_acts for segment in encoded_segments))
        speakers = list(flatten(segment.speakers for segment in encoded_segments))
        return words, tags, speakers

    def words_with_tags(
            self,
            add_turn_token: bool = True,
            use_joint_coding: bool = False,
            continuations_allowed: bool = True
    ) -> Tuple[List[str], List[str]]:
        words, tags, speakers = self.words_with_metadata(
            use_joint_coding=use_joint_coding,
            continuations_allowed=continuations_allowed,
            add_turn_token=add_turn_token,
        )
        return words, tags

    def render(self, max_turns=None, jupyter=True, tagset=SWDA_BUGGY_DIALOG_ACTS.values()):
        """Render the call as annotated with dialog acts in a Jupyter notebook"""

        # Render DAs options
        nlp = get_nlp()
        cmap = {k.upper(): col for k, col in zip(list(tagset), COLORMAP)}
        displacy_opts = {"colors": cmap}
        labels = 'ABCDEFGHIJKLMNOPRSTUVWXYZ'

        # Convert text to Doc
        rendered_htmls = []
        speakers = []
        for turn_no, (speaker, group) in enumerate(groupby(self, key=lambda segment: segment.speaker)):
            if speaker not in speakers:
                speakers.append(speaker)

            if max_turns is not None and turn_no > max_turns:
                break

            group = list(group)
            words = ' '.join(segment.text for segment in group).split()
            doc = Doc(nlp.vocab, words=[labels[speakers.index(speaker)]] + words)

            ents = []
            begin = 1
            for segment in group:
                words = segment.text.split()
                end = begin + len(words)
                ents.append(Span(doc, begin, end, label=segment.dialog_act or 'None'))
                begin = end
            doc.ents = ents

            rendered_htmls.append(displacy.render(doc, style="ent", jupyter=jupyter, options=displacy_opts))
        return rendered_htmls

    @property
    def turns(self) -> Iterable[Tuple[str, 'Call']]:
        for name, group in groupby(self, key=lambda fs: fs.speaker):
            yield name, Call(group)

    def dialog_act_spans(self, include_label: bool = True) -> Iterable[Tuple[int, int, str]]:
        idx = 0
        for segment in self:
            n_toks = len(segment.text.split())
            if include_label:
                yield idx, idx + n_toks, segment.dialog_act
            else:
                yield idx, idx + n_toks
            idx += n_toks

    def with_vocabulary(self, vocabulary: Set[str]) -> 'Call':
        return Call(segment.with_vocabulary(vocabulary) for segment in self)

    def encode(
            self,
            use_joint_coding: bool = False,
            continuations_allowed: bool = True,
            add_turn_token: bool = True,
    ) -> List['EncodedSegment']:
        speakers = [segment.speaker for segment in self]
        unique_speakers = set(speakers)

        encoded_with_idx = []
        for speaker in unique_speakers:
            speaker_segments = ((idx, segment) for idx, segment in enumerate(self) if segment.speaker == speaker)
            segment_windows = sliding_window(3, chain([(None, None)], speaker_segments, [(None, None)]))
            for (_, prv), (idx, cur), (_, nxt) in segment_windows:
                if use_joint_coding:
                    enc = [(word, CONTINUE_TAG) for word in cur.text.split()]
                    if not (continuations_allowed and nxt is not None and nxt.is_continuation):
                        enc[-1] = (enc[-1][0], cur.dialog_act)
                else:
                    enc = [(word, f'{CONTINUE_TAG}{cur.dialog_act}') for word in cur.text.split()]
                    if not (continuations_allowed and cur.is_continuation):
                        enc[0] = (enc[0][0], f'{BEGIN_TAG}{cur.dialog_act}')
                words, acts = zip(*enc)
                encoded_segment = EncodedSegment(
                    words=list(words),
                    encoded_acts=list(acts),
                    speakers=[cur.speaker] * len(words)
                )
                encoded_with_idx.append((idx, encoded_segment))

        encoded_with_idx = sorted(encoded_with_idx, key=lambda tpl: tpl[0])
        encoded_call = [enc for idx, enc in encoded_with_idx]

        if add_turn_token:
            encoded_call_with_turns = []
            segment_windows = sliding_window(2, chain([None], encoded_call))
            for prev_segment, encoded_segment in segment_windows:
                if prev_segment is not None and prev_segment.speakers[0] != encoded_segment.speakers[0]:
                    encoded_call_with_turns.append(EncodedSegment(
                        words=[NEW_TURN],
                        encoded_acts=[BLANK],
                        speakers=[encoded_segment.speakers[0]]
                    ))
                encoded_call_with_turns.append(encoded_segment)
            return encoded_call_with_turns

        return encoded_call


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
        call_windows = [Call(call[b: e]) for b, e in indices]
    else:
        call_windows = [call]
    return call_windows


class FunctionalSegment(NamedTuple):
    text: str
    dialog_act: Optional[str]
    speaker: str
    is_continuation: bool = False
    start: Optional[float] = None
    end: Optional[float] = None

    def words_with_metadata(self) -> Iterable[Tuple[str, Optional[str], str]]:
        for word in self.text.split():
            yield word, self.dialog_act, self.speaker

    def with_vocabulary(self, vocabulary: Set[str]) -> 'FunctionalSegment':
        new_text = ' '.join(w if w in vocabulary else OOV for w in self.text.split())
        return FunctionalSegment(new_text, *self[1:])


class EncodedSegment(NamedTuple):
    words: List[str]
    encoded_acts: List[str]
    speakers: List[str]

    def iter_tuples(self):
        return zip(self.words, self.encoded_acts, self.speakers)


def parse_swda_transcript(
        swda_tr,  # swda.Transcript
        strip_punctuation_and_lowercase: bool = False,
        merge_continuations: bool = False,
        tagset: str = 'basic'
) -> Tuple[str, Call]:
    normalize_text = create_text_normalizer(strip_punctuation_and_lowercase)
    dialog_acts = {
        'basic': SWDA_TAG_TO_DIALOG_ACT,
        'broken_swda': to_buggy_swda_42_labels(SWDA_BUGGY_DIALOG_ACTS),
        'segmentation': SEGMENTATION_ONLY_ACTS
    }[tagset]
    call_id = decode_swda_id(swda_tr)
    segments = (
        FunctionalSegment(
            text=normalize_text(' '.join(utt.text_words(filter_disfluency=True))),
            dialog_act=(
                dialog_acts[utt.damsl_act_tag()] if tagset == 'basic'
                else lookup_or_fix(utt.act_tag if tagset != 'segmentation' else SEGMENT_TAG, dialog_acts=dialog_acts)
            ),
            speaker=utt.caller
        ) for utt in swda_tr.utterances
    )
    # Remove segments which became empty as a result of text normalization (i.e. have no text, just punctuation)
    characters = re.compile(r'[a-zA-Z]+')
    segments = (seg for seg in segments if characters.search(seg.text))
    resolved_segments = (
        merge_continuations_with_their_previous_segments(segments)
        if merge_continuations
        else mark_continuations_as_separate_acts(segments)
    )
    return call_id, Call(resolved_segments)


def mark_continuations_as_separate_acts(segments: Iterable[FunctionalSegment]) -> List[FunctionalSegment]:
    resolved_segments = []
    prev_tag = {'A': 'Other', 'B': 'Other'}  # there seems to be exactly one case where the first DA is '+'
    for segment in segments:
        is_continuation = segment.dialog_act == CONTINUATION
        resolved_tag = prev_tag[segment.speaker] if is_continuation else segment.dialog_act
        resolved_segments.append(
            FunctionalSegment(
                text=segment.text,
                dialog_act=resolved_tag,
                speaker=segment.speaker,
                is_continuation=is_continuation
            )
        )
        prev_tag[segment.speaker] = resolved_tag
    return resolved_segments


def merge_continuations_with_their_previous_segments(segments: Iterable[FunctionalSegment]) -> List[FunctionalSegment]:
    resolved_segments = []
    for segment in segments:
        is_continuation = segment.dialog_act == CONTINUATION
        if is_continuation:
            try:
                index, prev_segment = [
                    (idx, seg)
                    for idx, seg in reversed(list(enumerate(resolved_segments)))
                    if seg.speaker == segment.speaker
                ][0]
                resolved_segments[index] = FunctionalSegment(
                    text=f'{prev_segment.text} {segment.text}',
                    dialog_act=prev_segment.dialog_act,
                    speaker=prev_segment.speaker,
                )
            except:
                # Exactly one case of continuation being the beginning of the conversation
                resolved_segments.append(FunctionalSegment(
                    text=segment.text,
                    dialog_act='Other',
                    speaker=segment.speaker
                ))
        else:
            resolved_segments.append(segment)
    return resolved_segments


def lookup_or_fix(tag: str, dialog_acts: Mapping[str, str]) -> str:
    if tag in dialog_acts:
        return dialog_acts[tag]
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
    if fixed_tag in dialog_acts:
        return dialog_acts[fixed_tag]
    return 'Other'


def decode_swda_id(transcript) -> str:
    return f"sw{transcript.swda_filename.split('_')[2].split('.')[0]}"


def to_transformers_ner_dataset(
        call: Call,
        continuations_allowed: bool = True,
        add_turn_token: bool = True,
        use_joint_coding: bool = True
) -> List[str]:
    """
    Convert a list of functional segments into text representations,
    used by the Transformers library to train NER models.
    """
    lines = []
    for segment in call.encode(
            use_joint_coding=use_joint_coding,
            continuations_allowed=continuations_allowed,
            add_turn_token=add_turn_token
    ):
        for word, act, speaker in segment.iter_tuples():
            lines.append(f'{word} {act}')
    return lines


def is_begin_act(tag):
    return tag.startswith(BEGIN_TAG)


def is_continued_act(tag):
    return tag.startswith(CONTINUE_TAG)


def decode_act(tag):
    if is_begin_act(tag) or is_continued_act(tag):
        return tag[2:]
    return tag


def create_text_normalizer(strip_punctuation_and_lowercase: bool = False) -> Callable[[str], str]:
    remove_patterns = list(map(
        re.compile,
        [
            r'<<[^>]*>>',  # <<talks to another person>>
            r'\(\([^)]*\)\)',  # ((Hailey)), what is that?
            r'\([^)]*\)',  # ... ?
            r'\(',  # unbalanced parentheses
            r'\)',  #
            r'#',  # comments?
            r'\*.+'  # comments about typos: e.g. "i think their *their -> they're"
        ]
    ))

    if strip_punctuation_and_lowercase:
        remove_patterns.append(re.compile(r'[!"#$%&()*+,./:;=?@\[\\\]^_`{|}~]'))

    remove_leading_nontext = re.compile(r'^[^a-zA-Z<]+([a-zA-Z])')
    correct_punctuation_whitespace = re.compile(r' ([.,?!])')
    wild_dashes = re.compile(r'(\s-+\s|-+$)')

    def normalize(text: str) -> str:
        for p in remove_patterns:
            text = p.sub('', text)
        text = text.replace('--', '')
        text = text.split('*')[0].strip()  # Comments after asterisk
        text = remove_leading_nontext.sub(r'\1', text)  # ". . Hi again." => "Hi again."
        text = correct_punctuation_whitespace.sub(r'\1', text)  # "Hi Jack ." -> "Hi Jack."
        if strip_punctuation_and_lowercase:
            text = wild_dashes.sub(' ', text).strip()
            text = text.lower()
        return ' '.join(text.split())

    return normalize
