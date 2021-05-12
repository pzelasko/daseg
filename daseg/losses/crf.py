from typing import Dict, List

import k2
import torch
from torch import Tensor, nn


class CRFLoss(nn.Module):
    """
    Conditional Random Field loss implemented with K2 library. It supports GPU computation.

    Currently, this loss assumes specific topologies for dialog acts/punctuation labeling.
    """

    def __init__(
            self,
            label_set: List[str],
            label2id: Dict[str, int],
            trainable_transition_scores: bool = True,
            ignore_index: int = -100
    ):
        super().__init__()
        self.label_set = label_set
        self.label2id = label2id
        self.ignore_index = ignore_index
        self.den = make_denominator(label_set, label2id, shared=True).to('cuda')
        self.A = create_bigram_lm([self.label2id[l] for l in label_set]).to('cuda')
        self.A_scores = nn.Parameter(self.A.scores.clone(), requires_grad=trainable_transition_scores).to('cuda')

    def forward(self, log_probs: Tensor, input_lens: Tensor, labels: Tensor):
        global it

        # Determine all relevant shapes - max_seqlen_scored is the longest sequence length of log_probs
        # after we remove ignored indices.
        bs, seqlen, nclass = log_probs.shape
        max_seqlen_scored = (labels[0] != self.ignore_index).sum()
        supervision_segments = make_segments(labels)

        log_probs_scored = log_probs.new_zeros(bs, max_seqlen_scored, nclass)
        assert max_seqlen_scored == supervision_segments[0, 2]
        for i in range(bs):
            log_probs_scored[i, :supervision_segments[i, 2], :] = log_probs[i, labels[i] != self.ignore_index, :]

        # (batch, seqlen, classes)
        posteriors = k2.DenseFsaVec(log_probs_scored, supervision_segments)

        # (fsavec)
        nums = make_numerator(labels)
        for i in range(nums.shape[0]):
            # The supervision has to have exactly the same number of arcs as the number of tokens
            # which contain labels to score, plus one extra arc for k2's special end-of-fst arc.
            assert nums[i].num_arcs == supervision_segments[i, 2] + 1
        self.A.set_scores_stochastic_(self.A_scores)
        nums = k2.intersect(self.A.to('cpu'), nums).to(log_probs.device)
        for i in range(nums.shape[0]):
            # The supervision has to have exactly the same number of arcs as the number of tokens
            # which contain labels to score, plus one extra arc for k2's special end-of-fst arc.
            assert nums[i].num_arcs == supervision_segments[i, 2] + 1

        if it % 100 == 0:
            for i in range(min(3, labels.size(0))):
                print('*' * 120)
                print('log_probs_scored.shape', log_probs_scored.shape)
                print(f'labels[{i}][:20] = ', labels[i][:20])
                print(f'labels[{i}][labels[{i}] != self.ignore_index][:20] = ', labels[i][labels[i] != self.ignore_index][:20])
                print(f'nums[{i}].labels[:20] = ', nums[i].labels[:20])
                print(f'log_probs_scored[{i}][:20] = ', log_probs_scored.argmax(dim=2)[i][:20])
                print('*' * 120)
        it += 1

        # (fsavec)
        num_lattices = k2.intersect_dense(nums, posteriors, output_beam=10.0)
        #den_lattice = k2.intersect_dense(self.den, posteriors, output_beam=10.0)

        # (batch,)
        num_scores = num_lattices.get_tot_scores(use_double_scores=True, log_semiring=True)
        #den_scores = den_lattice.get_tot_scores(use_double_scores=True, log_semiring=True)

        # (scalar)
        num_tokens = (labels != self.ignore_index).to(torch.int32).sum()
        #loss = (num_scores - den_scores).sum() / num_tokens
        loss = num_scores.sum() / num_tokens
        return loss

it = 0


def create_bigram_lm(labels: List[int]) -> k2.Fsa:
    """
    Create a bigram LM.
    The resulting FSA (A) has a start-state and a state for
    each label 0, 1, 2, ....; and each of the above-mentioned states
    has a transition to the state for each phone and also to the final-state.
    """
    final_state = len(labels) + 1
    rules = ''
    for i in range(1, final_state):
        rules += f'0 {i} {labels[i-1]} 0.0\n'

    for i in range(1, final_state):
        for j in range(1, final_state):
            rules += f'{i} {j} {labels[j-1]} 0.0\n'
        rules += f'{i} {final_state} -1 0.0\n'
    rules += f'{final_state}'
    return k2.Fsa.from_str(rules)


def make_symbol_table(label_set: List[str], label2id: Dict[str, int], shared: bool = True) -> k2.SymbolTable:
    """
    Creates a symbol table given a list of classes (e.g. dialog acts, punctuation, etc.).
    It adds extra symbols:
    - 'O' which is used to indicate special tokens such as <TURN>
    - (when shared=True) 'I-' which is the "in-the-middle" symbol shared between all classes
    - (when shared=False) 'I-<class>' which is the "in-the-middle" symbol,
        specific for each class (N x classes -> N x I- symbols)
    """
    symtab = k2.SymbolTable()
    del symtab._sym2id['<eps>']
    del symtab._id2sym[0]
    if shared:
        symtab.add('I-', label2id['I-'])
    for l in label_set:
        symtab.add(l, label2id[l])
        if not shared:
            symtab.add(f'I-{l}', label2id[f'I-{l}'])
    return symtab


def make_numerator(labels: Tensor) -> k2.Fsa:
    """
    Creates a numerator supervision FSA.
    It simply encodes the ground truth label sequence and allows no leeway.
    Returns a :class:`k2.FsaVec` with FSAs of differing length.
    """
    assert len(labels.shape) == 2
    nums = k2.create_fsa_vec([k2.linear_fsa(lab[lab != -100].tolist()) for lab in labels])
    return nums


def make_denominator(label_set: List[str], label2id: Dict[str, int], shared: bool = True) -> k2.Fsa:
    """
    Creates a "simple" denominator that encodes all possible transitions
    given the input label set.

    The labeling scheme is assumed to be IE with joint coding, e.g.:

        Here I am.~~~~~~ How are you today?~~
        I~~~ I Statement I~~ I~~ I~~ Question

    Or without joint coding:

        Here~~~~~~~ I~~~~~~~~~~ am.~~~~~~ How~~~~~~~ are~~~~~~~ you~~~~~~~ today?~~
        I-Statement I-Statement Statement I-Question I-Question I-Question Question

    When shared=True, it uses a shared "in-the-middle" label for all classes;
    otherwise each class has a separate one.
    """
    symtab = make_symbol_table(label_set, label2id, shared=shared)

    """
    shared=True
    0 0 Statement
    0 0 Question
    0 1 I-
    1 1 I-
    1 0 Statement
    1 0 Question
    0 2 -1
    2
    """

    """
    shared=False
    0 0 Statement
    0 0 Question
    0 1 I-Statement
    0 1 I-Question
    1 1 I-Statement
    1 1 I-Question
    1 0 Statement
    1 0 Question
    0 2 -1
    2
    """

    s = []
    if shared:
        s += [
            f'0 1 {symtab["I-"]} 0.0',
            f'1 1 {symtab["I-"]} 0.0'
        ]
    for idx, label in enumerate(label_set):
        s += [f'0 0 {symtab[label]} 0.0']
        if not shared:
            s += [
                f'0 1 {symtab["I-" + label]} 0.0'
                f'1 1 {symtab["I-" + label]} 0.0'
            ]
        s += [f'1 0 {symtab[label]} 0.0']
    s += ['0 2 -1 0.0', '2']
    s.sort()
    fsa = k2.Fsa.from_str('\n'.join(s))
    fsa.symbols = symtab
    return fsa


def make_segments(labels: Tensor) -> Tensor:
    """
    Creates a supervision segments tensor that indicates for each batch example,
    at which index the example has started, and how many tokens it has.
    """
    bs = labels.size(0)
    return torch.stack([
        torch.arange(bs, dtype=torch.int32),
        torch.zeros(bs, dtype=torch.int32),
        (labels != -100).to(torch.int32).sum(dim=1).cpu()
    ], dim=1).to(torch.int32)
