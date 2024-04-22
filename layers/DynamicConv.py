import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicWeights(nn.Module):
    def __init__(self, in_feature, ratios, K, lam, seq_len, init_weight=True):
        """
        Obtain the K weights from the input feature assigning to K candidate kernel
        :param in_feature: in feature number
        :param ratios: hidden_feature / in_feature
        :param K: candidates number
        :param lam:
        :param seq_len:
        :param init_weight:
        """
        super(DynamicWeights, self).__init__()
        self.avgpool_v = nn.AdaptiveAvgPool1d(1)
        self.avgpool_t = nn.AdaptiveAvgPool1d(1)
        if in_feature != 3:
            hidden_states = int(in_feature * ratios) + 1
        else:
            hidden_states = K
        self.fc1 = nn.Conv1d(in_feature + seq_len, hidden_states, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_states)
        self.fc2 = nn.Conv1d(hidden_states, K, 1, bias=True)
        self.lam = lam
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        """

        :param x: x ~ [batch, in_feature, seq_length]
        :return:  softmax weights ~ [batch, K]
        """
        # x ~ batch, in_feature, seq_length
        x_t = self.avgpool_t(x)  # batch, infeature, 1
        x_v = self.avgpool_v(torch.permute(x, (0, 2, 1)))  # batch, infeature, 1
        # print(x_v.shape, x_t.shape)
        x = self.fc1(torch.cat((x_v, x_t), dim=-2))  # batch, hiddenenfeature, 1
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)  # batch, K, 1 -> batch, K
        return F.softmax(x / self.lam, 1)


class Dynamic_conv1d(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1,
                 bias=True, K=4, lam=49, init_weight=True, seq_len=64):
        """
        This class implements the dynamic kernel used in DBT
        :param in_feature: input channel
        :param out_feature: out channel
        :param kernel_size: convolution kernel size
        :param ratio: hidden states / in feature
        :param stride: convolution stride
        :param padding: convolution padding
        :param dilation: convolution dilation
        :param bias: whether ths convolution use biase
        :param K: number of candidate kernels
        :param lam: penalty factor to neutralize one-shot
        :param init_weight:
        :param seq_len:
        """
        super(Dynamic_conv1d, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.K = K
        self.attention = DynamicWeights(in_feature, ratio, K, lam, seq_len)

        self.weight = nn.Parameter(torch.randn(K, out_feature, in_feature, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_feature))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def forward(self, x, uniform=False):
        """

        :param x: x ~ [batch, in_feature, seq_length]
        :param uniform: if true, set all weight to uniformed weights: 1/K
        :return: [batch, out_feature, seq_length]
        """
        if uniform:
            softmax_attention = torch.ones((x.shape[0], self.K)).to(x.device)/self.K
        else:
            softmax_attention = self.attention(x)  # batch, K

        batch_size, in_feature, tw = x.size()
        x = x.reshape(1, -1, tw, )
        weight = self.weight.view(self.K, -1)

        # construct the dynamic kernel parameters
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_feature, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=batch_size)
        else:
            output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=batch_size)

        output = output.reshape(batch_size, self.out_feature, -1)
        return output