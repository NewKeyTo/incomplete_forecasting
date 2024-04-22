import time
from layers.TCN import DBT
from einops import rearrange, repeat
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

class AttentionScaleFusion(nn.Module):

    def __init__(self, ks_num, seq_len, in_feature, ):
        """
        This class implement the Attention Scale Fusion Layer used in DBT Block.
        This layer maintain the input shape
        :param ks_num: candidate kernel numbers in Dynamic Kernel, default is 3
        :param seq_len: input seq_length
        :param in_feature: in feature number
        """
        super(AttentionScaleFusion, self).__init__()
        self.pool = nn.Linear(seq_len * ks_num, 1)
        self.Wg = nn.Linear(in_feature, in_feature, bias=False)
        self.Wh = nn.Linear(in_feature, in_feature, bias=False)
        self.v = nn.Linear(in_feature, 1, bias=False)
        self._init_weights()

    def forward(self, x, dynamic_uniform=False):
        """
        forward function of the Attention Scale Fusion Layer
        :param x: input of shape [batch, in_feature, seq_length, ks_num]
        :return:
        """
        batch, in_feature, seq_len, ks_num = x.shape
        if not dynamic_uniform:
            hg = self.pool(x.view(batch, in_feature, -1)).squeeze(-1)  # [batch, in_feature]
            weighted_hs = self.Wh(x.permute((0, 2, 3, 1)))  # [batch, seq_len, ks_num, in_feature]
            weighted_hg = self.Wg(hg)[:, None, None, :].tile(1, seq_len, ks_num, 1)  # [batch, seq_len, ks_num, in_feature]
            e = self.v(weighted_hs + weighted_hg)  # [batch, seq_len, ks_num, 1]
            alpha = torch.softmax(e/49, dim=-1)  # [batch, seq_len, ks_num, 1]
            rs = (x.permute((0, 2, 1, 3)) @ alpha).squeeze(-1).transpose(-1, -2)  # [batch, in_feature, seq_len]
            return rs
        else:
            return x.mean(dim=-1)

    def _init_weights(self):
        nn.init.xavier_normal_(self.pool.weight)
        nn.init.xavier_normal_(self.Wg.weight)
        nn.init.xavier_normal_(self.Wh.weight)
        nn.init.xavier_normal_(self.v.weight)

class DBT_Block(nn.Module):

    def __init__(self, in_feature, out_feature, seq_len, kernel_sizes, level, bidirectional=True, dynamic=True):
        """
        This class implements the DBT block with three parallel DBT units and one Attention Scale Fusion layer.
        :param in_feature: in feature number
        :param out_feature: out feature number
        :param seq_len: input sequence length
        :param kernel_sizes: the parallel DBT unit kernel sizes, list-like
        :param level: indicates which layer this block appears, used to set the dilation size.
        :param bidirectional: whether bi-directional
        :param dynamic: whether to use dynamic kernel
        """
        super(DBT_Block, self).__init__()
        dbt_ls = []
        for ks in kernel_sizes:
            dbt_ls.append(DBT(
                in_feature,
                out_feature,
                seq_len=seq_len,
                level=level,
                kernel_size=ks,
                dynamic=dynamic,
                bidirectional=bidirectional,
                res=True)
            )
        self.multiscale_dbt = nn.ModuleList(dbt_ls)
        self.asf = AttentionScaleFusion(ks_num=len(kernel_sizes),
                                        seq_len=seq_len,
                                        in_feature=out_feature)
        self.kernel_sizes = kernel_sizes

    def forward(self, x, dynamic_uniform=False):
        multiscale_ls = []
        for i in range(len(self.kernel_sizes)):
            tmp_rs = self.multiscale_dbt[i](x, dynamic_uniform)
            multiscale_ls.append(tmp_rs)
        multiscl_rs = torch.stack(multiscale_ls, dim=-1)
        asf_rs = self.asf(multiscl_rs, dynamic_uniform)
        return asf_rs



class Model(nn.Module):

    def __init__(self,  configs, enc_hiddens = [32, 12, 64], bidirectional=True, dynamic=True,
                 kernel_sizes=(3, 5, 7), attention_embedding=True, mode = 'predict', *args, **kwargs):
        """
        The entire DBT-DMAE model implementation.
        :param in_feature: MTS in feature
        :param enc_hiddens: encoder hidden states, list-like, e.g. [32, 16, 64] means a three layered DMAE with hidden states
            equal to 32, 16, 64, respectively.
        :param seq_len: MTS input seq_length
        :param bidirectional: whether bidirectional
        :param dynamic: whether to use the Dynamic Kernel
        :param kernel_sizes: the parallel DBT unit kernel sizes, list-like, e.g. [3, 5, 7] means using three parallel
            DBT units within a DBT block whose kernel size is set to 3, 5, and 7, respectively.
        :param attention_embedding: whether to use Dynamic Positional Embedding in the missing entries or
            just hard-coded embedding.
        :param transferout: the ultimate output dimension, setting to None means the model is under the pretrain mode,
            and the transferout is automatically set to the same dimension of entire input MTS.
        :param args:
        :param kwargs:
        """
        super(Model, self).__init__()
        in_feature = configs.enc_in
        # enc_hiddens = configs.enc_hiddens
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        c_out = configs.c_out
        
        self.mode = mode
        self.in_feature = in_feature
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.c_out = c_out
        self.enc_hiddens = enc_hiddens
        self.dynamic = dynamic
        self.bidirectional = bidirectional
        self.kernel_sizes = kernel_sizes

        self.attention_embedding = attention_embedding

        # transferout set to None means pretraining, so the out dim is seq_len * in_feature
        self.args = args
        self.kwargs = kwargs
        encoder_ls = []
        level = 1
        for hs in enc_hiddens:
            encoder_ls.append(DBT_Block(
                in_feature=in_feature,
                out_feature=hs,
                seq_len=seq_len,
                kernel_sizes=kernel_sizes,
                level=level,
                bidirectional=bidirectional,
                dynamic=dynamic
            ))
            in_feature = hs
            level += 1
        self.encoder = nn.ModuleList(encoder_ls)
        
        if self.mode == 'pretrain':
            self.decoder = nn.Conv1d(self.enc_hiddens[-1], self.in_feature, 1)
        else:
            self.decoder = nn.Linear(self.enc_hiddens[-1] * self.seq_len, configs.pred_len * configs.c_out)

        if attention_embedding:
            print(self.in_feature)
            self.position_embedding = DBT_Block(
                                                in_feature=self.in_feature,
                                                out_feature=self.in_feature,
                                                seq_len=seq_len,
                                                kernel_sizes=kernel_sizes,
                                                level=1,
                                                bidirectional=bidirectional,
                                                dynamic=dynamic
                                                      )
        else:
            self.position_embedding = torch.nn.Parameter(torch.zeros(self.seq_len, self.in_feature), requires_grad=True)

    def forward(self, *x, dynamic_uniform=False, **kwargs):
        """
        forward function of the whole DBT-DMAE
        :param x: input MTS data with missing and artificial masking, of shape [batch, seq_length, in_feature]
        :param dynamic_uniform: the warm-up trick switch, setting to True means warm-up and setting all the softmax
        weights in the model as with uniformed weights
        :param kwargs:
        :return:
        """
        hs = x[0]
        mask = x[1]  # batch, tw, infeature
        batch, tw, in_feature = hs.shape

        if not self.attention_embedding:
            embedding_res = self.position_embedding.repeat(batch, 1, 1)
        else:
            embedding_res = self.position_embedding(hs.transpose(-1, -2), dynamic_uniform).transpose(-1, -2)

        hs = hs * mask + (1 - mask) * embedding_res

        for dbtblock in self.encoder:
            hs = dbtblock(hs.transpose(-1, -2), dynamic_uniform).transpose(-1, -2)

        if self.mode == 'pretrain':
            decoded_hs = self.decoder(hs.transpose(-1, -2))
        else:
            hs = rearrange(hs, 'b l d -> b (l d)')
            decoded_hs = self.decoder(hs)
            decoded_hs = decoded_hs.view(hs.shape[0], self.pred_len, self.c_out)
    
        return decoded_hs, embedding_res

# if __name__ == '__main__':
#     #import torchsummary
#     in_feature = 64
#     out_feature = 8
#     seq_len = 128
#     kernel_sizes = [3,5,7]
#     enc_hiddens = [in_feature, in_feature, in_feature]
#     dbt_dmae = DBT_DMAE(in_feature=in_feature, enc_hiddens=enc_hiddens, seq_len=seq_len,
#                           kernel_sizes=kernel_sizes).cuda()
#     x = torch.randn((1, seq_len, in_feature)).cuda()
#     mask = 1 * (x < 0)
#     rs = dbt_dmae(x, mask, dynamic_uniform=False)
    # torchsummary.summary(dbt_dmae, x, mask)