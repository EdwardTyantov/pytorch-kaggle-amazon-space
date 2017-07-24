#-*- coding: utf8 -*-
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function import InplaceFunction
from torch._thnn import type2backend


# class SELU(InplaceFunction):
#     alpha = 1.6732632423543772848170429916717
#     scale = 1.0507009873554804934193349852946
#
#     def __init__(self, inplace=False):
#         super(SELU, self).__init__()
#         self.inplace = inplace
#         self.save_for_backward = None
#
#     def forward(self, input):
#         backend = type2backend[type(input)]
#         if self.inplace:
#             self.mark_dirty(input)
#             output = input
#         else:
#             output = input.new(input.size())
#         backend.ELU_updateOutput(
#             backend.library_state,
#             input,
#             output,
#             self.alpha,
#             self.inplace,
#         )
#         output.mul_(self.scale)
#         self.save_for_backward = (input, output)
#         return output
#
#     def backward(self, grad_output):
#         input, output = self.save_for_backward
#         if grad_output.volatile:
#             grad_input = Variable(input.data.new(input.size()), volatile=True)
#             backend = type2backend[type(input.data)]
#             backend.ELU_updateGradInput(
#                 backend.library_state,
#                 input.data,
#                 grad_output.data.mul(self.scale),
#                 grad_input.data,
#                 output.data.div(self.scale),
#                 self.alpha,
#                 False
#             )
#         else:
#             positive_mask = (output > 0).type_as(grad_output)
#             negative_mask = (output <= 0).type_as(grad_output)
#             grad_input = grad_output * self.scale * (positive_mask +
#                                                      negative_mask * (output / self.scale + self.alpha))
#         return grad_input, None
#
#
# def f_selu(input, inplace=False):
#     return SELU(inplace)(input)
#
#
# class SeluActivation(nn.Module):
#     def __init__(self, inplace=False):
#         super(SeluActivation, self).__init__()
#         self.inplace = inplace
#
#     def forward(self, input):
#         return f_selu(input, self.inplace)
#
#     def __repr__(self):
#         inplace_str = ' (inplace)' if self.inplace else ''
#         return self.__class__.__name__ + inplace_str

class SeluActivation(nn.Module):
    def __init__(self):
        super(SeluActivation, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
    def forward(self, x):
        temp1 = self.scale * F.relu(x)
        temp2 = self.scale * self.alpha * (F.elu(-1*F.relu(-1*x)))
        return temp1 + temp2


def alpha_dropout(input, p=0.5, training=False):
    if p < 0 or p > 1:
        raise ValueError("dropout probability has to be between 0 and 1, "
                         "but got {}".format(p))

    if p == 0 or not training:
        return input

    alpha = -1.7580993408473766
    keep_prob = 1 - p
    # TODO avoid casting to byte after resize
    noise = input.data.new().resize_(input.size())
    noise.bernoulli_(p)
    noise = Variable(noise.byte())

    output = input.masked_fill(noise, alpha)

    a = (keep_prob + alpha ** 2 * keep_prob * (1 - keep_prob)) ** (-0.5)
    b = -a * alpha * (1 - keep_prob)

    return output.mul_(a).add_(b)


class AlphaDropout(nn.Module):

    def __init__(self, p=0.5):
        super(AlphaDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, input):
        return alpha_dropout(input, self.p, self.training)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + 'p = ' + str(self.p) + ')'