# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

def glorot_init(params):
    for p in params:
        if len(p.data.size()) > 1:
            init.xavier_normal_(p.data)
            
class LabelSmoothing(nn.Module):
    """Implement label smoothing.

    Reference: the annotated transformer
    """

    def __init__(self, smoothing, tgt_vocab_size, ignore_indices=None):
        if ignore_indices is None: ignore_indices = []

        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        smoothing_value = smoothing / float(tgt_vocab_size - 1 - len(ignore_indices))
        one_hot = torch.zeros((tgt_vocab_size,)).fill_(smoothing_value)
        for idx in ignore_indices:
            one_hot[idx] = 0.

        self.confidence = 1.0 - smoothing
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

    def forward(self, model_prob, target):
        # (batch_size, *, tgt_vocab_size)
        dim = list(model_prob.size())[:-1] + [1]
        true_dist = Variable(self.one_hot, requires_grad=False).repeat(*dim)
        true_dist.scatter_(-1, target.unsqueeze(-1), self.confidence)
        # true_dist = model_prob.data.clone()
        # true_dist.fill_(self.smoothing / (model_prob.size(1) - 1))  # FIXME: no label smoothing for <pad> <s> and </s>
        # true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return self.criterion(model_prob, true_dist).sum(dim=-1)