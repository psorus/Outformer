from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module, TransformerEncoder

from pfns.layer import TransformerEncoderLayer, _get_activation_fn
from pfns.utils import SeqBN


class TransformerModel(nn.Module):
    def __init__(self,
                 encoder, 
                 ninp, 
                 nhead, 
                 nhid, 
                 nlayers, 
                 dropout=0.0, 
                 style_encoder=None, 
                 y_encoder=None,
                 pos_encoder=None, 
                 decoder_dict=None, 
                 input_normalization=False, 
                 init_method=None, 
                 pre_norm=False,
                 activation='gelu', 
                 recompute_attn=False, 
                 num_global_att_tokens=0,
                 all_layers_same_init=False, 
                 efficient_eval_masking=True, 
                 model_para_dict=None):
        super().__init__()

        self.model_para_dict = model_para_dict  # its info is used during initialization
        encoder_layer_creator = lambda is_final_layer: TransformerEncoderLayer(ninp, nhead, nhid, dropout,
                                                                               activation=activation,
                                                                               pre_norm=pre_norm,
                                                                               recompute_attn=recompute_attn,
                                                                               model_para_dict=model_para_dict,
                                                                               is_final_layer=is_final_layer)
        self.transformer_encoder = TransformerEncoder(encoder_layer_creator(is_final_layer=None), nlayers) \
            if all_layers_same_init else TransformerEncoderDiffInit(encoder_layer_creator, nlayers)
        self.ninp = ninp
        self.encoder = encoder
        self.y_encoder = y_encoder
        self.pos_encoder = pos_encoder

        def make_decoder_dict(decoder_description_dict):
            if decoder_description_dict is None or len(decoder_description_dict) == 0:
                return None
            initialized_decoder_dict = {}
            for decoder_key in decoder_description_dict:
                decoder_model, decoder_n_out = decoder_description_dict[decoder_key]
                if decoder_model is None:
                    initialized_decoder_dict[decoder_key] = nn.Sequential(
                        nn.Linear(ninp, nhid),
                        nn.GELU(),
                        nn.Linear(nhid, decoder_n_out),
                    )
                else:
                    initialized_decoder_dict[decoder_key] = decoder_model(ninp, nhid, decoder_n_out)
                print('Initialized decoder for', decoder_key, 'with', decoder_description_dict[decoder_key],
                      ' and nout', decoder_n_out)
            return torch.nn.ModuleDict(initialized_decoder_dict)

        self.decoder_dict = make_decoder_dict(decoder_dict)
        self.input_ln = SeqBN(ninp) if input_normalization else None
        self.style_encoder = style_encoder
        self.init_method = init_method
        assert num_global_att_tokens == 0, 'global attention tokens are not used in the retraining path'
        self.efficient_eval_masking = efficient_eval_masking

        self.nhid = nhid

        self.init_weights()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault('efficient_eval_masking', False)
        if hasattr(self, 'decoder') and not hasattr(self, 'decoder_dict'):
            self.add_module('decoder_dict', nn.ModuleDict({'standard': self.decoder}))

        def add_approximate_false(module):
            if isinstance(module, nn.GELU):
                module.__dict__.setdefault('approximate', 'none')

        self.apply(add_approximate_false)

    def init_weights(self):
        initrange = 1.
        # if isinstance(self.encoder,EmbeddingEncoder):
        #    self.encoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.zero_()
        # self.decoder.weight.data.uniform_(-initrange, initrange)

        num_R = self.model_para_dict['num_R']

        def init_router(layer):  # following PFN below
            # att_compressor of LandmarkMHA
            attns = layer.router_att.att_compressor if isinstance(layer.router_att.att_compressor, nn.ModuleList) else [
                layer.router_att.att_compressor]
            for attn in attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)
            # att_recover of LandmarkMHA
            attns = layer.router_att.att_recover if isinstance(layer.router_att.att_recover, nn.ModuleList) else [
                layer.router_att.att_recover]
            for attn in attns:
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)

        if self.init_method is not None:
            self.apply(self.init_method)

        for layer in self.transformer_encoder.layers:
            print('+'*20,
                  '[CAUTIOUS] You are now using zero-initialization of the attention, which might result in untrainable'
                  ' parameters. Please make sure this is the intended behavior!',
                  '+'*20)
            nn.init.zeros_(layer.linear2.weight)
            nn.init.zeros_(layer.linear2.bias)
            if num_R is None:  # just vanillaMHA
                attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    nn.init.zeros_(attn.out_proj.bias)
            else:  # LandmarkMHA x 1 + vanillaMHA x 1
                attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
                for attn in attns:
                    nn.init.zeros_(attn.out_proj.weight)
                    nn.init.zeros_(attn.out_proj.bias)

                init_router(layer=layer)

    def forward(self, *args, **kwargs):
        """
        This will perform a forward-pass (possibly recording gradients) of the prior.
        We have multiple interfaces we support with this prior:

        prior(train_x, train_y, test_x, src_mask=None, style=None, only_return_standard_out=True)
        prior((x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
        prior((style,x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
        """
        if len(args) == 3:
            # print('args=3 is called')
            # case prior(train_x, train_y, test_x, src_mask=None, style=None, only_return_standard_out=True)
            assert all(kwarg in {'src_mask', 'style', 'only_return_standard_out'} for kwarg in kwargs.keys()), \
                f"Unrecognized keyword argument in kwargs: {set(kwargs.keys()) - {'src_mask', 'style', 'only_return_standard_out'} }"
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=0)
            style = kwargs.pop('style', None)
            # print('single eval pos', len(args[0]))
            return self._forward((style, x, args[1]), single_eval_pos=len(args[0]), **kwargs)
        elif len(args) == 1 and isinstance(args, tuple):
            # case prior((x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
            # case prior((style,x,y), src_mask=None, single_eval_pos=None, only_return_standard_out=True)
            assert all(kwarg in {'src_mask', 'single_eval_pos', 'only_return_standard_out'} for kwarg in kwargs.keys()), \
                f"Unrecognized keyword argument in kwargs: {set(kwargs.keys()) - {'src_mask', 'single_eval_pos', 'only_return_standard_out'} }"
            return self._forward(*args, **kwargs)

    def _forward(self, src, src_mask=None, single_eval_pos=None, only_return_standard_out=True):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        if len(src) == 2:  # (x,y) and no style
            src = (None,) + src

        style_src, x_src, y_src = src  # x_src: (Num_samples, 1, N_features (100))
        if single_eval_pos is None:
            single_eval_pos = x_src.shape[0]
        x_src = self.encoder(x_src)

        y_src = self.y_encoder(
            y_src.unsqueeze(-1) if len(y_src.shape) < len(x_src.shape) else y_src) if y_src is not None else None
        
        if self.style_encoder:
            assert style_src is not None, 'style_src must be given if style_encoder is used'
            style_src = self.style_encoder(style_src).unsqueeze(0)
        else:
            style_src = torch.tensor([], device=x_src.device)

        if src_mask is None:
            src_mask = single_eval_pos + len(style_src)

        train_x = x_src[:single_eval_pos]
        if y_src is not None:
            train_x = train_x + y_src[:single_eval_pos]
            
        src = torch.cat([style_src, train_x, x_src[single_eval_pos:]], 0)

        if self.input_ln is not None:
            src = self.input_ln(src)

        if self.pos_encoder is not None:
            src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src, src_mask)

        out_range_start = single_eval_pos + len(style_src)
        output = {k: v(output[out_range_start:]) for k, v in self.decoder_dict.items()} if self.decoder_dict is not None else {}
        if only_return_standard_out:
            return output['standard']
        return output


class TransformerEncoderDiffInit(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer_creator: a function generating objects of TransformerEncoderLayer class without args (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer_creator, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [encoder_layer_creator(is_final_layer=(layer_id == num_layers - 1)) for layer_id in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output