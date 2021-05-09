from typing import List

import k2
import torch
from torch import Tensor, nn


class CRFLoss(nn.Module):
    """
    Conditional Random Field loss implemented with K2 library. It supports GPU computation.

    Currently, this loss assumes specific topologies for dialog acts/punctuation labeling.
    """

    def __init__(self, label_set: List[str], trainable_transition_scores: bool = True):
        super().__init__()
        self.label_set = label_set
        self.den = make_denominator(label_set, shared=True)
        self.den_scores = nn.Parameter(self.den.scores.clone(), requires_grad=trainable_transition_scores)

    def forward(self, log_probs: Tensor, input_lens: Tensor, labels: Tensor):
        # (batch, seqlen, classes)
        posteriors = k2.DenseFsaVec(
            log_probs,
            supervision_segments=make_segments(input_lens)
        )
        # (fsavec)
        nums = make_numerator(labels, input_lens)
        self.den.set_scores_stochastic_(self.den_scores)

        # (fsavec)
        num_lattices = k2.intersect_dense(nums, posteriors, output_beam=10.0)
        den_lattice = k2.intersect_dense(self.den, posteriors, output_beam=10.0)

        # (batch,)
        num_scores = num_lattices.get_tot_scores(use_double_scores=True, log_semiring=True)
        den_scores = den_lattice.get_tot_scores(use_double_scores=True, log_semiring=True)

        # (scalar)
        loss = (num_scores - den_scores).sum()
        return loss


def make_symbol_table(label_set: List[str], shared: bool = True) -> k2.SymbolTable:
    """
    Creates a symbol table given a list of classes (e.g. dialog acts, punctuation, etc.).
    It adds extra symbols:
    - 'O' which is used to indicate special tokens such as <TURN>
    - (when shared=True) 'I-' which is the "in-the-middle" symbol shared between all classes
    - (when shared=False) 'I-<class>' which is the "in-the-middle" symbol,
        specific for each class (N x classes -> N x I- symbols)
    """
    symtab = k2.SymbolTable()
    symtab.add('O')
    if shared:
        symtab.add('I-')
    for l in label_set:
        symtab.add(l)
        if not shared:
            symtab.add(f'I-{l}')
    return symtab


def make_numerator(labels: Tensor, input_lens: Tensor) -> k2.Fsa:
    """
    Creates a numerator supervision FSA.
    It simply encodes the ground truth label sequence and allows no leeway.
    Returns a :class:`k2.FsaVec` with FSAs of differing length.
    """
    assert labels.size(0) == input_lens.size(0)
    assert len(labels.shape) == 2
    assert len(input_lens.shape) == 1
    nums = k2.create_fsa_vec([
        k2.linear_fsa(l[:llen]) for l, llen in zip(labels, input_lens)
    ])
    return nums


def make_denominator(label_set: List[str], shared: bool = True) -> k2.Fsa:
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
    symtab = make_symbol_table(label_set, shared=shared)

    """
    shared=True
    0 0 O
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
    0 0 O
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

    s = [f'0 0 {symtab["O"]} 0.0']
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
    fsa = k2.Fsa.from_str(s)
    fsa.symbols = symtab
    return fsa


def make_segments(input_lens: Tensor) -> Tensor:
    """
    Creates a supervision segments tensor that indicates for each batch example,
    at which index the example has started, and how many tokens it has.
    """
    bs = input_lens.size(0)
    return torch.stack([
        torch.arange(bs, dtype=torch.int32),
        torch.zeros(bs, dtype=torch.int32),
        input_lens.cpu().to(torch.int32)
    ])
