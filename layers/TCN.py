import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .DynamicConv import Dynamic_conv1d

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalResidualUnit(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, seq_len, dropout=0.2, dynamic=True, res=True,
                 bidirectional=True):
        """
        This class implements the main structure of DBT unit, except the batchnorm
        :param n_inputs: input channel
        :param n_outputs: out channel
        :param kernel_size: kernel size
        :param stride: stride, set to 1
        :param dilation: dilation size
        :param padding: padding number, which is relevant to kernel size the dilation size
        :param seq_len: input seq_length
        :param dropout: dropout ratio
        :param dynamic: whether to use dynamic kernel
        :param res: whether to use residual structure in TCN network
        :param bidirectional: whether to apply the time flipping trick
        """
        super(TemporalResidualUnit, self).__init__()
        self.res = res
        self.bidirectional = bidirectional
        self.dynamic = dynamic
        if not dynamic:
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
            if bidirectional:
                self.conv1_b = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        else:
            self.conv1 = weight_norm(
                Dynamic_conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation,
                               seq_len=seq_len))
            if bidirectional:
                self.conv1_b = weight_norm(
                    Dynamic_conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                   seq_len=seq_len))
        if bidirectional:
            self.conv1_linear = nn.Conv1d(2*n_outputs, n_outputs, 1)

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        if not dynamic:
            self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
            if bidirectional:
                self.conv2_b = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        else:
            self.conv2 = weight_norm(
                Dynamic_conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation,
                               seq_len=seq_len))
            if bidirectional:
                self.conv2_b = weight_norm(
                    Dynamic_conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                   seq_len=seq_len))
        if bidirectional:
            self.conv2_linear = nn.Conv1d(2*n_outputs, n_outputs, 1)

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, dynamic_uniform=False):
        """
        The forward function for DBT.
        :param x: input x, of shape [batch, n_inputs, seq_length]
        :param dynamic_uniform: when set to True, use the dynamic mechanism to compute the K different weights,
        otherwise the weights are uniformly set
        :return: [batch, n_outputs, seq_length]
        """
        if self.dynamic:
            out = self.conv1(x, uniform=dynamic_uniform)
        else:
            out = self.conv1(x)
        if self.bidirectional:
            if self.dynamic:
                out_b = self.conv1_b(torch.flip(x, dims=[-1]), uniform=dynamic_uniform)
            else:
                out_b = self.conv1_b(torch.flip(x, dims=[-1]))
            out_b = torch.flip(out_b, dims=[-1])
            out = self.conv1_linear(torch.cat((out, out_b), dim=-2))
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        if self.dynamic:
            out_f = self.conv2(out, uniform=dynamic_uniform)
        else:
            out_f = self.conv2(out)
        if self.bidirectional:
            if self.dynamic:
                out_b = self.conv2_b(torch.flip(out, dims=[-1]), uniform=dynamic_uniform)
            else:
                out_b = self.conv2_b(torch.flip(out, dims=[-1]))
            out_b = torch.flip(out_b, dims=[-1])
            out_f = self.conv2_linear(torch.cat((out_f, out_b), dim=-2))
        out = self.chomp2(out_f)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        if self.res:
            return self.relu(out + res)
        else:
            return self.relu(out)


class DBT(nn.Module):
    def __init__(self, in_feature, out_feature, seq_len, level, kernel_size=2, dropout=0.2, dynamic=True, bidirectional=True, res=True):
        """
        Implementation of the Dynamic Bidirectional Temporal Convolution Network
        :param in_feature: in feature num
        :param out_features: out feature num
        :param seq_len: input sequence length
        :param level: the level of this DBT unit in the whole model
        :param kernel_size: kernel size of the convolution
        :param dropout: dropout ratio
        :param dynamic: whether to use Dynamic Kernel
        :param bidirectional: whether to use the time flip trick
        :param res: whether to use the residual structure
        """
        super(DBT, self).__init__()
        self.bidirectional = bidirectional
        self.level = level
        dilation_size = 2 ** (level - 1)
        in_channel = in_feature
        out_channel = out_feature
        self.dbt = TemporalResidualUnit(in_channel, out_channel, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size-1) * dilation_size, dropout=dropout, dynamic=dynamic,
                                        res=res, bidirectional=bidirectional, seq_len=seq_len)
        self.bn = nn.BatchNorm1d(out_channel)


    def forward(self, x, dynamic_uniform=False):
        x = self.dbt(x, dynamic_uniform)
        x = self.bn(x)
        return x

