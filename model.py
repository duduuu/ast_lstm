# coding=utf-8
from __future__ import print_function

import os
import math
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F

from asdl.hypothesis import Hypothesis, GenTokenAction
from asdl.transition_system import ApplyRuleAction, ReduceAction, Action
from common.registerable import Registrable
from components.decode_hypothesis import DecodeHypothesis
from components.action_info import ActionInfo
from components.dataset import Batch
from common.utils import update_args, init_arg_parser
from nn_utils import LabelSmoothing

@Registrable.register('default_parser')
class Parser(nn.Module):

    def __init__(self, args, vocab, transition_system):
        super(Parser, self).__init__()

        self.args = args
        self.vocab = vocab

        self.transition_system = transition_system
        self.grammar = self.transition_system.grammar

        # Embedding layers

        # embedding table of ASDL production rules (constructors), one for each ApplyConstructor action,
        # the last entry is the embedding for Reduce action
        self.production_embed = nn.Embedding(len(transition_system.grammar) + 1, args.action_embed_size)

        # embedding table for target primitive tokens
        self.primitive_embed = nn.Embedding(len(vocab), args.action_embed_size)

        # embedding table for ASDL fields in constructors
        self.field_embed = nn.Embedding(len(transition_system.grammar.fields), args.field_embed_size)

		# embedding table for ASDL types
        self.type_embed = nn.Embedding(len(transition_system.grammar.types), args.type_embed_size)

        init.xavier_normal_(self.production_embed.weight.data)
        init.xavier_normal_(self.primitive_embed.weight.data)
        init.xavier_normal_(self.field_embed.weight.data)
        init.xavier_normal_(self.type_embed.weight.data)

        # LSTMs
        input_dim = args.action_embed_size # previous action
		# frontier info
        input_dim += args.action_embed_size
        input_dim += args.field_embed_size
        input_dim += args.type_embed_size
        input_dim += args.hidden_size

        input_dim += args.hidden_size # input feeding

        self.lstm_cell = nn.LSTMCell(input_dim, args.hidden_size)

        if args.primitive_token_label_smoothing:
            self.label_smoothing = LabelSmoothing(args.primitive_token_label_smoothing, len(self.vocab), ignore_indices=[0, 1, 2])

        self.production_readout_b = nn.Parameter(torch.FloatTensor(len(transition_system.grammar) + 1).zero_())
        self.tgt_token_readout_b = nn.Parameter(torch.FloatTensor(len(vocab)).zero_())

        self.production_readout = lambda q: F.linear(q, self.production_embed.weight, self.production_readout_b)
        self.tgt_token_readout = lambda q: F.linear(q, self.primitive_embed.weight, self.tgt_token_readout_b)

        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        if args.cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

    def init_hidden(self, batch_size):
        return Variable(self.new_tensor(batch_size, self.args.hidden_size).zero_()), Variable(self.new_tensor(batch_size, self.args.hidden_size).zero_())

    def score(self, examples):
        args = self.args

        batch = Batch(examples, self.grammar, self.vocab, cuda=self.args.cuda)
        batch_size = len(batch)

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        out_vecs = []
        history_states = []

        h_tm1 = self.init_hidden(batch_size)

        for t in range(batch.max_action_num):
            if t == 0:
                x = Variable(self.new_tensor(batch_size, self.lstm_cell.input_size).zero_(), requires_grad=False)

                offset = args.action_embed_size
                offset += args.hidden_size
                offset += args.action_embed_size
                offset += args.field_embed_size

                x[:, offset: offset + args.type_embed_size] = self.type_embed(Variable(self.new_long_tensor(
                    [self.grammar.type2id[self.grammar.root_type] for e in batch.examples])))

            else:
                a_tm1_embeds = []
                for example in batch.examples:
                    # action t -1
                    if t < len(example.tgt_actions):
                        a_tm1 = example.tgt_actions[t - 1]
                        if isinstance(a_tm1.action, ApplyRuleAction):
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.action.production]]
                        elif isinstance(a_tm1.action, ReduceAction):
                            a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                        else: #GetTokenAction
                            a_tm1_embed = self.primitive_embed.weight[self.vocab[a_tm1.action.token]]
                    else:
                        a_tm1_embed = zero_action_embed

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds] # previous action
                inputs.append(out_tm1) # input feeding
                # parent info
                parent_production_embed = self.production_embed(batch.get_frontier_prod_idx(t))
                inputs.append(parent_production_embed)
                parent_field_embed = self.field_embed(batch.get_frontier_field_idx(t))
                inputs.append(parent_field_embed)
                parent_field_type_embed = self.type_embed(batch.get_frontier_field_type_idx(t))
                inputs.append(parent_field_type_embed)

                actions_t = [e.tgt_actions[t] if t < len(e.tgt_actions) else None for e in batch.examples]
                parent_states = torch.stack([history_states[p_t][0][batch_id]
                                                 for batch_id, p_t in
                                                 enumerate(a_t.parent_t if a_t else 0 for a_t in actions_t)])
                inputs.append(parent_states)

                x = torch.cat(inputs, dim=-1)

            h_t, cell_t = self.lstm_cell(x, h_tm1)

            out_vecs.append(h_t)
            history_states.append((h_t, cell_t))

            h_tm1 = (h_t, cell_t)
            out_tm1 = h_t

        query_vectors = torch.stack(out_vecs, dim=0)

        # ApplyRule (i.e., ApplyConstructor) action probabilities
        # (tgt_action_len, batch_size, grammar_size)
        apply_rule_prob = F.softmax(self.production_readout(query_vectors), dim=-1)

        # probabilities of target (gold-standard) ApplyRule actions
        # (tgt_action_len, batch_size)
        tgt_apply_rule_prob = torch.gather(apply_rule_prob, dim=2,
											index=batch.apply_rule_idx_matrix.unsqueeze(2)).squeeze(2)

        #### compute generation and copying probabilities

        # (tgt_action_len, batch_size, primitive_vocab_size)
        gen_from_vocab_prob = F.softmax(self.tgt_token_readout(query_vectors), dim=-1)

        # (tgt_action_len, batch_size)
        tgt_primitive_gen_from_vocab_prob = torch.gather(gen_from_vocab_prob, dim=2,
														index=batch.primitive_idx_matrix.unsqueeze(2)).squeeze(2)

        if self.training:
            # (tgt_action_len, batch_size)
            tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                gen_from_vocab_prob.log(),
                batch.primitive_idx_matrix)

        # (tgt_action_len, batch_size)
        action_prob = tgt_apply_rule_prob.log() * batch.apply_rule_mask + \
                    tgt_primitive_gen_from_vocab_log_prob * batch.gen_token_mask

        scores = torch.sum(action_prob, dim=0)

        return scores


    def parse(self, beam_size=100):
        args = self.args
        vocab = self.vocab

        T = torch.cuda if args.cuda else torch

        zero_action_embed = Variable(self.new_tensor(args.action_embed_size).zero_())

        hyp_scores = Variable(self.new_tensor([0.]), volatile=True)

        t = 0
        h_tm1 = self.init_hidden(1)
        hypotheses = [DecodeHypothesis()]
        hyp_states = [[]]
        completed_hypotheses = []

        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            hyp_num = len(hypotheses)

            if t == 0:
                x = Variable(self.new_tensor(1, self.lstm_cell.input_size).zero_(), volatile=True)

                offset = args.action_embed_size  # prev_action
                offset += args.hidden_size
                offset += args.action_embed_size
                offset += args.field_embed_size

                x[0, offset: offset + args.type_embed_size] = \
                    self.type_embed.weight[self.grammar.type2id[self.grammar.root_type]]
            else:
                actions_tm1 = [hyp.actions[-1] for hyp in hypotheses]

                a_tm1_embeds = []
                for a_tm1 in actions_tm1:
                    if a_tm1:
                        if isinstance(a_tm1, ApplyRuleAction):
                            a_tm1_embed = self.production_embed.weight[self.grammar.prod2id[a_tm1.production]]
                        elif isinstance(a_tm1, ReduceAction):
                            a_tm1_embed = self.production_embed.weight[len(self.grammar)]
                        else:
                            a_tm1_embed = self.primitive_embed.weight[self.vocab[a_tm1.token]]

                        a_tm1_embeds.append(a_tm1_embed)
                    else:
                        a_tm1_embeds.append(zero_action_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]

                inputs.append(att_tm1)

                # frontier production
                frontier_prods = [hyp.frontier_node.production for hyp in hypotheses]
                frontier_prod_embeds = self.production_embed(Variable(self.new_long_tensor(
                	[self.grammar.prod2id[prod] for prod in frontier_prods])))
                inputs.append(frontier_prod_embeds)

                # frontier field
                frontier_fields = [hyp.frontier_field.field for hyp in hypotheses]
                frontier_field_embeds = self.field_embed(Variable(self.new_long_tensor([
                	self.grammar.field2id[field] for field in frontier_fields])))
                inputs.append(frontier_field_embeds)

                # frontier field type
                frontier_field_types = [hyp.frontier_field.type for hyp in hypotheses]
                frontier_field_type_embeds = self.type_embed(Variable(self.new_long_tensor([
                	self.grammar.type2id[type] for type in frontier_field_types])))
                inputs.append(frontier_field_type_embeds)

                # parent states
                p_ts = [hyp.frontier_node.created_time for hyp in hypotheses]
                parent_states = torch.stack([hyp_states[hyp_id][p_t][0] for hyp_id, p_t in enumerate(p_ts)])
                inputs.append(parent_states)

                x = torch.cat(inputs, dim=-1)

            h_t, cell_t = self.lstm_cell(x, h_tm1)

            # Variable(batch_size, grammar_size)
            apply_rule_log_prob = F.log_softmax(self.production_readout(h_t), dim=-1)

            # Variable(batch_size, primitive_vocab_size)
            gen_from_vocab_prob = F.softmax(self.tgt_token_readout(h_t), dim=-1)

            primitive_prob = gen_from_vocab_prob

            gentoken_prev_hyp_ids = []
            gentoken_new_hyp_unks = []
            applyrule_new_hyp_scores = []
            applyrule_new_hyp_prod_ids = []
            applyrule_prev_hyp_ids = []

            for hyp_id, hyp in enumerate(hypotheses):
                # generate new continuations
                action_types = self.transition_system.get_valid_continuation_types(hyp)

                for action_type in action_types:
                    if action_type == ApplyRuleAction:
                        productions = self.transition_system.get_valid_continuating_productions(hyp)
                        for production in productions:
                            prod_id = self.grammar.prod2id[production]
                            prod_score = apply_rule_log_prob[hyp_id, prod_id].data
                            new_hyp_score = hyp.score + prod_score

                            applyrule_new_hyp_scores.append(new_hyp_score)
                            applyrule_new_hyp_prod_ids.append(prod_id)
                            applyrule_prev_hyp_ids.append(hyp_id)
                    elif action_type == ReduceAction:
                        action_score = apply_rule_log_prob[hyp_id, len(self.grammar)].data
                        new_hyp_score = hyp.score + action_score

                        applyrule_new_hyp_scores.append(new_hyp_score)
                        applyrule_new_hyp_prod_ids.append(len(self.grammar))
                        applyrule_prev_hyp_ids.append(hyp_id)
                    else:
                        # GenToken action
                        gentoken_prev_hyp_ids.append(hyp_id)

            new_hyp_scores = None
            if applyrule_new_hyp_scores:
                new_hyp_scores = Variable(self.new_tensor(applyrule_new_hyp_scores))
            if gentoken_prev_hyp_ids:
                primitive_log_prob = torch.log(primitive_prob)
                gen_token_new_hyp_scores = (hyp_scores[gentoken_prev_hyp_ids].unsqueeze(1) + primitive_log_prob[gentoken_prev_hyp_ids, :]).view(-1)

                if new_hyp_scores is None: new_hyp_scores = gen_token_new_hyp_scores
                else: new_hyp_scores = torch.cat([new_hyp_scores, gen_token_new_hyp_scores])

            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores,
                                                             k=min(new_hyp_scores.size(0), beam_size - len(completed_hypotheses)))
            
            live_hyp_ids = []
            new_hypotheses = []
            for new_hyp_score, new_hyp_pos in zip(top_new_hyp_scores.data.cpu(), top_new_hyp_pos.data.cpu()):
                action_info = ActionInfo()
                new_hyp_pos = int(new_hyp_pos)
                if new_hyp_pos < len(applyrule_new_hyp_scores):
                    # it's an ApplyRule or Reduce action
                    prev_hyp_id = applyrule_prev_hyp_ids[new_hyp_pos]
                    prev_hyp = hypotheses[prev_hyp_id]

                    prod_id = applyrule_new_hyp_prod_ids[new_hyp_pos]
                    # ApplyRule action
                    if prod_id < len(self.grammar):
                        production = self.grammar.id2prod[prod_id]
                        action = ApplyRuleAction(production)
                    # Reduce action
                    else:
                        action = ReduceAction()
                else:
                    # it's a GenToken action
                    token_id = (new_hyp_pos - len(applyrule_new_hyp_scores)) % primitive_prob.size(1)

                    k = (new_hyp_pos - len(applyrule_new_hyp_scores)) // primitive_prob.size(1)

                    prev_hyp_id = gentoken_prev_hyp_ids[k]
                    prev_hyp = hypotheses[prev_hyp_id]

                    if token_id == vocab.unk_id:
                        if gentoken_new_hyp_unks:
                            token = gentoken_new_hyp_unks[k]
                        else:
                            token = vocab.id2word[vocab.unk_id]
                    else:
                        token = vocab.id2word[token_id]

                    action = GenTokenAction(token)

                action_info.action = action
                action_info.t = t
                if t > 0:
                    action_info.parent_t = prev_hyp.frontier_node.created_time
                    action_info.frontier_prod = prev_hyp.frontier_node.production
                    action_info.frontier_field = prev_hyp.frontier_field.field

                new_hyp = prev_hyp.clone_and_apply_action_info(action_info)
                new_hyp.score = new_hyp_score

                if new_hyp.completed:
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                hyp_states = [hyp_states[i] + [(h_t[i], cell_t[i])] for i in live_hyp_ids]
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = h_t[live_hyp_ids]
                hypotheses = new_hypotheses
                hyp_scores = Variable(self.new_tensor([hyp.score for hyp in hypotheses]))
                t += 1
            else:
                break

        completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'transition_system': self.transition_system,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)

    @classmethod
    def load(cls, model_path, cuda=False):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        transition_system = params['transition_system']
        saved_args = params['args']
        # update saved args
        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']
        saved_args.cuda = cuda

        parser = cls(saved_args, vocab, transition_system)

        parser.load_state_dict(saved_state)

        if cuda: parser = parser.cuda()
        parser.eval()

        return parser