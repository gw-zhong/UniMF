import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys


# Code adapted from the fairseq repo.

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, lens, modalities, missing, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.l_len, self.a_len, self.v_len = lens
        self.modalities = modalities
        self.missing = missing
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        #################################################################################
        # Masked Multi-Head Self-Attention
        mask = torch.zeros_like(attn_weights)
        if self.modalities == 'L':
            mask[:, :self.l_len, self.l_len + 1:] = float('-inf')
            mask[:, self.l_len + 1:, :self.l_len] = float('-inf')
            if self.missing == 'A':
                a_len = src_len - self.l_len
                a_mask = self.generate_square_subsequent_mask(a_len)
                mask[:, self.l_len:, self.l_len:] = a_mask
            elif self.missing == 'V':
                v_len = src_len - self.l_len
                v_mask = self.generate_square_subsequent_mask(v_len)
                mask[:, self.l_len:, self.l_len:] = v_mask
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'A':
            mask[:, :self.a_len, self.a_len + 1:] = float('-inf')
            mask[:, self.a_len + 1:, :self.a_len] = float('-inf')
            if self.missing == 'L':
                l_len = src_len - self.a_len
                l_mask = self.generate_square_subsequent_mask(l_len)
                mask[:, self.a_len:, self.a_len:] = l_mask
            elif self.missing == 'V':
                v_len = src_len - self.a_len
                v_mask = self.generate_square_subsequent_mask(v_len)
                mask[:, self.a_len:, self.a_len:] = v_mask
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'V':
            mask[:, :self.v_len, self.v_len + 1:] = float('-inf')
            mask[:, self.v_len + 1:, :self.v_len] = float('-inf')
            if self.missing == 'L':
                l_len = src_len - self.v_len
                l_mask = self.generate_square_subsequent_mask(l_len)
                mask[:, self.v_len:, self.v_len:] = l_mask
            elif self.missing == 'A':
                a_len = src_len - self.v_len
                a_mask = self.generate_square_subsequent_mask(a_len)
                mask[:, self.v_len:, self.v_len:] = a_mask
            else:
                raise ValueError('Unknown missing modality type')
        elif self.modalities == 'LA':
            v_len = src_len - self.l_len - self.a_len
            mask[:, :self.l_len, self.l_len:self.l_len + self.a_len] = float('-inf')
            mask[:, self.l_len:self.l_len + self.a_len, :self.l_len] = float('-inf')
            mask[:, :self.l_len + self.a_len, self.l_len + self.a_len + 1:] = float('-inf')
            mask[:, self.l_len + self.a_len + 1:, :self.l_len + self.a_len] = float('-inf')
            v_mask = self.generate_square_subsequent_mask(v_len)
            mask[:, self.l_len + self.a_len:, self.l_len + self.a_len:] = v_mask
        elif self.modalities == 'LV':
            a_len = src_len - self.l_len - self.v_len
            mask[:, :self.l_len, self.l_len:self.l_len + self.v_len] = float('-inf')
            mask[:, self.l_len:self.l_len + self.v_len, :self.l_len] = float('-inf')
            mask[:, :self.l_len + self.v_len, self.l_len + self.v_len + 1:] = float('-inf')
            mask[:, self.l_len + self.v_len + 1:, :self.l_len + self.v_len] = float('-inf')
            a_mask = self.generate_square_subsequent_mask(a_len)
            mask[:, self.l_len + self.v_len:, self.l_len + self.v_len:] = a_mask
        elif self.modalities == 'AV':
            l_len = src_len - self.a_len - self.v_len
            mask[:, :self.a_len, self.a_len:self.a_len + self.v_len] = float('-inf')
            mask[:, self.a_len:self.a_len + self.v_len, :self.a_len] = float('-inf')
            mask[:, :self.a_len + self.v_len, self.a_len + self.v_len + 1:] = float('-inf')
            mask[:, self.a_len + self.v_len + 1:, :self.a_len + self.v_len] = float('-inf')
            l_mask = self.generate_square_subsequent_mask(l_len)
            mask[:, self.a_len + self.v_len:, self.a_len + self.v_len:] = l_mask
        else:
            raise ValueError('Unknown modalities type')
        attn_weights = attn_weights + mask
        #################################################################################
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
