from functools import partial
from torch import nn
from torch.nn.modules.transformer import _get_activation_fn, Module, Tensor, MultiheadAttention, Linear, \
    Dropout, LayerNorm

from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from einops import repeat
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch import _VF, sym_int as _sym_int, Tensor
from torch.nn import _reduction as _Reduction, grad  # noqa: F401


# from: Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting
class RouterMultiHeadAttention(Module):
    # for context-points only (self-attention), can't be used for cross-attention between context-points and test points
    def __init__(self, d_model, nhead, dropout, batch_first, device, dtype, d_ff=None, num_R=50,
                 dropout_rate=0.2,
                 **kwargs):
        super(RouterMultiHeadAttention, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.batch_first = batch_first
        # att_compressor: router (bs, R, d) as Q, input (bs, L, d) as K,V -> compressed (bs, R, d)
        self.att_compressor = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # att_recover: input (bs, L, d) as Q, compressed (bs, R, d) as K,V -> recovered (bs, L, d)
        self.att_recover = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                              **factory_kwargs)
        if batch_first:
            self.router = nn.Parameter(torch.randn(1, num_R, d_model))
        else:
            self.router = nn.Parameter(torch.randn(num_R, 1, d_model))

        self.dropout = nn.Dropout(dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)

    def forward(self, query, key, value, average_attn_weights, skip_att=False):  # mimic the input of MultiheadAttention
        # query, key, value are the same; if not batch_first: (seq_len, bs, d), else: (bs, seq_len, d)
        if skip_att:
            return query, None
        else:
            if self.batch_first:
                batch = query.shape[0]
                batch_router = repeat(self.router, 'batch_placeholder factor d -> (repeat batch_placeholder) factor d',
                                      repeat=batch)
            else:
                batch = query.shape[1]
                batch_router = repeat(self.router, 'factor batch_placeholder d -> factor (repeat batch_placeholder) d',
                                      repeat=batch)
            router, router_att_weight = self.att_compressor(batch_router, key, value,
                                                            average_attn_weights=average_attn_weights)

            recovered_rep, recover_att_weight = self.att_recover(query, router, router,
                                                                 average_attn_weights=average_attn_weights)
            rep = query + self.dropout(recovered_rep)
            rep = self.ln1(rep)

            return rep, {'router_att': router_att_weight, 'recover_att': recover_att_weight}


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    If num_R is not None, we utilize a special RouterMultiHeadAttention to replace standard attention

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network prior (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, pre_norm=False,
                 device=None, dtype=None, recompute_attn=False,
                 model_para_dict=None, is_final_layer=None) -> None:
        self.src_right_att = None
        self.src_left_att = None
        self.num_R = model_para_dict['num_R']
        self.last_layer_no_R = model_para_dict['last_layer_no_R']
        self.is_final_layer = is_final_layer
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        if self.num_R is not None:
            print(f'using router attention (num_R={self.num_R}, last_layer_no_R={self.last_layer_no_R}, is_final_layer={self.is_final_layer})')
            self.router_att = RouterMultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                       num_R=self.num_R, **factory_kwargs)
        else:
            print('using vanilla MHA')
        # Implementation of Feedforward prior
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.pre_norm = pre_norm
        self.recompute_attn = recompute_attn
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        assert type(src_mask) == int
        assert src_key_padding_mask is None

        single_eval_position = src_mask
        src_ = self.norm1(src) if self.pre_norm else src
        src_to_attend_to = src_[:single_eval_position]

        # self-attention on context (left) positions
        if self.num_R is not None:
            src_left, self.src_left_att = self.router_att(
                src_[:single_eval_position], src_[:single_eval_position], src_[:single_eval_position],
                average_attn_weights=False,
                skip_att=self.last_layer_no_R and self.is_final_layer,
            )
        else:
            src_left, self.src_left_att = self.self_attn(
                src_[:single_eval_position], src_[:single_eval_position], src_[:single_eval_position],
                average_attn_weights=False,
            )

        # cross-attention: test positions attend to context
        src_right, self.src_right_att = self.self_attn(
            src_[single_eval_position:], src_to_attend_to, src_to_attend_to, average_attn_weights=False,
        )

        src = src + self.dropout1(torch.cat([src_left, src_right], dim=0))
        if not self.pre_norm:
            src = self.norm1(src)

        src_ = self.norm2(src) if self.pre_norm else src
        src = src + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(src_)))))
        if not self.pre_norm:
            src = self.norm2(src)

        return src