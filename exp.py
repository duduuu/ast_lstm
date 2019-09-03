import argparse

import pickle

import numpy as np
import time
import os
import sys

import torch
from torch.autograd import Variable

from asdl import *
from asdl.asdl import ASDLGrammar
from common.registerable import Registrable
from components.dataset import Dataset, Example
from common.utils import update_args, init_arg_parser
from model import Parser
import nn_utils


def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        
    np.random.seed(int(args.seed * 13 / 7))

    return args


def train(args):
    """Maximum Likelihood Estimation"""

    # load in train/dev set
    train_set = Dataset.from_bin_file(args.train_file)

    vocab = pickle.load(open(args.vocab, 'rb'))

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = Registrable.by_name(args.transition_system)(grammar)

    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg
    model = parser_cls(args, vocab, transition_system)
    model.train()

    if args.cuda: model.cuda()

    optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)

    nn_utils.glorot_init(model.parameters())

    print('begin training, %d training examples' % len(train_set), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = 0.
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_actions)]
            train_iter += 1
            optimizer.zero_grad()

            ret_val = model.score(batch_examples)
            loss = -ret_val

            # print(loss.data)
            loss_val = torch.sum(loss).data
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            loss.backward()

            # clip gradient
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            if train_iter % args.log_every == 0:
                log_str = '[Iter %d] loss=%.5f' % (train_iter, report_loss / report_examples)

                print(log_str, file=sys.stderr)
                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        model_file = args.save_to + '.iter%d.bin' % train_iter
        print('save model to [%s]' % model_file, file=sys.stderr)
        model.save(model_file)

        if epoch == args.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)

def test(args):
    assert args.load_model

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    """
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_args.cuda = args.cuda
    args.lang = saved_args.lang
    """

    parser_cls = Registrable.by_name(args.parser)
    model = parser_cls.load(model_path=args.load_model, cuda=args.cuda)

    decode_results = []
    count = 0
    
    hyps = model.parse(beam_size)
    decoded_hyps = []
    for hyp_id, hyp in enumerate(hyps):
        try:
            hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
            print(hyp.code)
            decoded_hyps.append(hyp)
        except:
            pass

    """
    if args.save_decode_to:
        pickle.dump(decode_results, open(args.save_decode_to, 'wb'))
    """

if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    print(args, file=sys.stderr)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise RuntimeError('unknown mode')