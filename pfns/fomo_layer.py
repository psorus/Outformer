from functools import partial
from torch import nn
from torch.nn.modules.transformer import _get_activation_fn, Module, Tensor, Optional, MultiheadAttention, Linear, \
    Dropout, LayerNorm

from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from einops import repeat
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch import _VF, sym_int as _sym_int, Tensor
from torch.nn import _reduction as _Reduction, grad  # noqa: F401
from .customized_att import CustomizedMHA


# from: Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting
class RouterMultiHeadAttention(Module):
    # for context-points only (self-attention), can't be used for cross-attention between context-points and test points
    def __init__(self, d_model, nhead, dropout, batch_first, device, dtype, d_ff=None, num_R=50,
                 dropout_rate=0.2,
                 **kwargs):
        super(RouterMultiHeadAttention, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        d_ff = d_ff or 2 * d_model
        self.batch_first = batch_first
        self.att_compressor = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # self.att_compressor = CustomizedMHA(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                     **factory_kwargs)
        # use router (bs, router, d) as Q, and input (bs, L, d) as K, V ---> compressed result: (bs, router, d)
        self.att_recover = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                              **factory_kwargs)
        # self.att_recover = CustomizedMHA(d_model, nhead, dropout=dropout, batch_first=batch_first,
        #                                  **factory_kwargs)
        # use input (bs, L, d) as Q, and compressed result (bs, router, d) as K, V ---> recovered result: (bs, L, d)
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
                # batch_router = self.router.expand(batch, -1, -1)
            else:
                batch = query.shape[1]
                batch_router = repeat(self.router, 'factor batch_placeholder d -> factor (repeat batch_placeholder) d',
                                      repeat=batch)
                # batch_router = self.router.expand(-1, batch, -1)

            router, router_att_weight = self.att_compressor(batch_router, key, value,
                                                            average_attn_weights=average_attn_weights)

            recovered_rep, recover_att_weight = self.att_recover(query, router, router,
                                                                 average_attn_weights=average_attn_weights)

            rep = query + self.dropout(recovered_rep)
            rep = self.ln1(rep)

            return rep, {'router_att': router_att_weight, 'recover_att': recover_att_weight}


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

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
                 device=None, dtype=None, recompute_attn=False, save_trainingset_representations=False,
                 model_para_dict=None, is_final_layer=None) -> None:
        self.src_right_att = None
        self.src_left_att = None
        self.num_R = model_para_dict['num_R']
        self.last_layer_no_R = model_para_dict['last_layer_no_R']
        self.is_final_layer = is_final_layer
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if self.num_R is not None:  # and not (self.last_layer_no_R and self.is_final_layer):  # use RouterMHA
            print('using vanilla + router')
            print(f'last_layer_no_R={self.last_layer_no_R}, is_final_layer={self.is_final_layer}')
            self.router_att = RouterMultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                       num_R=self.num_R, **factory_kwargs)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
            # self.self_attn = CustomizedMHA(d_model, nhead, dropout=dropout, batch_first=batch_first,
            #                                **factory_kwargs)
        else:  # use vanillaMHA
            print('using vanilla HMA')
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)
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
        self.save_trainingset_representations = save_trainingset_representations
        self.saved_src_to_attend_to = None

        self.activation = _get_activation_fn(activation)

        self.before_ffn = None
        self.final_rep = None

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)
        self.__dict__.setdefault('save_trainingset_representations', False)

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
        if self.save_trainingset_representations: assert isinstance(src_mask, int) and not self.training, \
            "save_trainingset_representations is only supported in eval mode and requires src_mask to be an int"

        if self.pre_norm:
            src_ = self.norm1(src)
        else:
            src_ = src
        
        if isinstance(src_mask, tuple):
            # global attention setup
            assert not self.self_attn.batch_first
            assert src_key_padding_mask is None

            global_src_mask, trainset_src_mask, valset_src_mask = src_mask

            num_global_tokens = global_src_mask.shape[0]
            num_train_tokens = trainset_src_mask.shape[0]

            global_tokens_src = src_[:num_global_tokens]
            train_tokens_src = src_[num_global_tokens:num_global_tokens + num_train_tokens]
            global_and_train_tokens_src = src_[:num_global_tokens + num_train_tokens]
            eval_tokens_src = src_[num_global_tokens + num_train_tokens:]

            attn = partial(checkpoint, self.self_attn) if self.recompute_attn else self.self_attn

            global_tokens_src2 = \
                attn(global_tokens_src, global_and_train_tokens_src, global_and_train_tokens_src, None, True,
                     global_src_mask)[0]
            train_tokens_src2 = \
                attn(train_tokens_src, global_tokens_src, global_tokens_src, None, True, trainset_src_mask)[0]
            eval_tokens_src2 = attn(eval_tokens_src, src_, src_,
                                    None, True, valset_src_mask)[0]

            src2 = torch.cat([global_tokens_src2, train_tokens_src2, eval_tokens_src2], dim=0)

        elif isinstance(src_mask, int):
            # since efficient_eval_masking will be true, then src_mask will be int (in transformer.py)
            assert src_key_padding_mask is None
            single_eval_position = src_mask
            src_to_attend_to = src_[:single_eval_position]
            if self.save_trainingset_representations:
                if single_eval_position == src_.shape[0] or single_eval_position is None:
                    self.saved_src_to_attend_to = src_to_attend_to
                elif single_eval_position == 0:
                    if self.saved_src_to_attend_to is None:
                        raise ValueError(
                            "First save the trainingset representations by passing in a src_mask of None or the length of the src")
                    src_to_attend_to = self.saved_src_to_attend_to
                else:
                    raise ValueError(
                        "save_trainingset_representations only supports single_eval_position == 0 or single_eval_position == src.shape[0]")
            # src_left = self.self_attn(src_[:single_eval_position], src_[:single_eval_position], src_[:single_eval_position])[0]
            # src_right = self.self_attn(src_[single_eval_position:], src_to_attend_to, src_to_attend_to)[0]

            # self-att
            if self.num_R is not None:  # and not (self.last_layer_no_R and self.is_final_layer):  # use RouterMHA
                src_left, self.src_left_att = self.router_att(src_[:single_eval_position],
                                                              src_[:single_eval_position],
                                                              src_[:single_eval_position],
                                                              average_attn_weights=False,
                                                              skip_att=self.last_layer_no_R and self.is_final_layer
                                                              )

            else:  # use vanillaMHA
                src_left, self.src_left_att = self.self_attn(src_[:single_eval_position],
                                                             src_[:single_eval_position],
                                                             src_[:single_eval_position], average_attn_weights=False)
            # cross-att
            src_right, self.src_right_att = self.self_attn(src_[single_eval_position:], src_to_attend_to,
                                                           src_to_attend_to, average_attn_weights=False)

            src2 = torch.cat([src_left, src_right], dim=0)
        else:
            if self.recompute_attn:
                src2 = checkpoint(self.self_attn, src_, src_, src_, src_key_padding_mask, True, src_mask)[0]
            else:
                src2 = self.self_attn(src_, src_, src_, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)

        if not self.pre_norm:
            src = self.norm1(src)

        if self.pre_norm:
            src_ = self.norm2(src)
        else:
            src_ = src

        self.before_ffn = src_
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        src = src + self.dropout2(src2)

        if not self.pre_norm:
            src = self.norm2(src)
        self.final_rep = src
        return src