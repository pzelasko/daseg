from typing import List

import k2
import torch
from torch import Tensor, nn


class CRFLoss(nn.Module):
    def __init__(self, label_set: List[str]):
        super().__init__()
        self.label_set = label_set
        self.den = make_topology(label_set, shared=True)
        self.den_scores = nn.Parameter(self.den.scores.clone(), requires_grad=True)

    def forward(self, log_probs: Tensor, input_lens: Tensor, labels: Tensor):
        posteriors = k2.DenseFsaVec(
            log_probs,
            supervision_segments=make_segments(input_lens)
        )
        nums = make_numerator(labels, input_lens)
        self.den.set_scores_stochastic_(self.den_scores)

        num_lattices = k2.intersect_dense(nums, posteriors, output_beam=10.0)
        den_lattice = k2.intersect_dense(self.den, posteriors, output_beam=10.0)

        num_score = num_lattices.get_tot_scores(use_double_scores=True, log_semiring=True).sum()
        den_score = den_lattice.get_tot_scores(use_double_scores=True, log_semiring=True).sum()

        loss = num_score - den_score
        return loss


def make_symbol_table(label_set: List[str], shared: bool = True) -> k2.SymbolTable:
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
    assert labels.size(0) == input_lens.size(0)
    assert len(labels.shape) == 2
    assert len(input_lens.shape) == 1
    nums = k2.create_fsa_vec([
        k2.linear_fsa(l[:llen]) for l, llen in zip(labels, input_lens)
    ])
    return nums


def make_topology(label_set: List[str], shared: bool = True) -> k2.Fsa:
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

    s = [f'0 0 {symtab["O"]}']
    if shared:
        s += [
            f'0 1 {symtab["I-"]}',
            f'1 1 {symtab["I-"]}'
        ]
    for idx, label in enumerate(label_set):
        s += [f'0 0 {symtab[label]}']
        if not shared:
            s += [
                f'0 1 {symtab["I-" + label]}'
                f'1 1 {symtab["I-" + label]}'
            ]
        s += [f'1 0 {symtab[label]}']
    s += ['0 2 -1', '2']
    s.sort()
    fsa = k2.Fsa.from_str(s)
    fsa.symbols = symtab
    return fsa


def make_segments(input_lens: Tensor) -> Tensor:
    bs = input_lens.size(0)
    return torch.stack([
        torch.arange(bs, dtype=torch.int32),
        torch.zeros(bs, dtype=torch.int32),
        input_lens.cpu().to(torch.int32)
    ])
