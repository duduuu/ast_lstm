# coding=utf-8
from __future__ import print_function

import sys
import numpy as np
import pickle

import os
from pdfrw import PdfReader, PdfWriter, PdfName, PdfArray, PdfDict, PdfObject, IndirectPdfDict, PdfString
from pdfrw.objects.pdfname import BasePdfName

from collections import Counter
from itertools import chain

from asdl.hypothesis import Hypothesis
from components.action_info import get_action_infos
from asdl.transition_system import *
from components.dataset import Example
from components.vocab import VocabEntry
from asdl.lang.pdf.pdf_transition_system import *


def load_dataset(transition_system, path, num, reorder_predicates=True):
    grammar = transition_system.grammar

    examples = []
    pre_len = 0
    if os.path.exists('data/pdf/train.bin'):
        examples = pickle.load(open('data/pdf/train.bin', 'rb'))
        pre_len = len(examples)

    idx = 0
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        print(item)

        try:
            pdf = PdfReader(item_path)
        except:
            continue

        for page in pdf.pages:
            idx += 1
            if idx <= pre_len:
                continue
            print(idx)

            try:
                tgt_ast = pdf_to_ast(grammar, page, [])
            except:
                continue

            tgt_actions = transition_system.get_actions(tgt_ast)

            """
            hyp = Hypothesis()
            for action in tgt_actions:
                assert action.__class__ in transition_system.get_valid_continuation_types(hyp)
                if isinstance(action, ApplyRuleAction):
                    assert action.production in transition_system.get_valid_continuating_productions(hyp)
                hyp = hyp.clone_and_apply_action(action)
            assert hyp.frontier_node is None and hyp.frontier_field is None 
            """

            tgt_action_infos = get_action_infos(tgt_actions)

            example = Example(idx=idx, tgt_actions=tgt_action_infos, meta=None)

            examples.append(example)

        if idx >= num:
            break

    return examples

def prepare_pdf_dataset():
    grammar = ASDLGrammar.from_text(open('asdl/lang/pdf/pdf_asdl.txt').read())
    transition_system = PdfTransitionSystem(grammar)

    train_set = load_dataset(transition_system, 'data/pdf/pdfs', 3000)

    # generate vocabulary
    primitive_tokens = [map(lambda a: a.action.token, filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                        for e in train_set]

    primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=5000, freq_cutoff=0)

    # print('generated vocabulary %s' % repr(primitive_vocab), file=sys.stderr)

    action_len = [len(e.tgt_actions) for e in train_set]
    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_len))), file=sys.stderr)

    pickle.dump(train_set, open('data/pdf/train.bin', 'wb'))
    pickle.dump(primitive_vocab, open('data/pdf/vocab.bin', 'wb'))

if __name__ == '__main__':
    prepare_pdf_dataset()
