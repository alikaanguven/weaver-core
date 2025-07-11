import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
import torch
from weaver.nn.model.ParticleTransformer import *
from weaver.utils.logger import _logger

import time
torch.autograd.set_detect_anomaly(True)


class ParticleTransformerDVTagger(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 # network configurations
                 pair_input_type='pp',
                 pair_input_dim=None,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=(128, 512, 128),
                 pair_embed_dims=(64, 64, 64),
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params=None,
                 fc_params=(),
                 activation='gelu',
                 # misc
                 version=1,
                 weight_init='moco',
                 fix_init=True,
                 trim=True,
                 for_inference=False,
                 for_segmentation=False,
                 use_amp=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        _logger.info('ParticleTransformer init-ed: %s', locals())

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.for_segmentation = for_segmentation
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           activation=activation,
                           layer_scale_init_values=None,
                           drop_path_rate=0.,
                           scale_attn_mask=False,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)
        if version > 1:
            default_cfg.update(
                activation='swiglu',
                scale_fc=False, scale_attn=False, scale_heads=False, scale_resids=False,
            )

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info('cfg_block: %s' % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        cfg_cls_block.update({'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0})
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        _logger.info('cfg_cls_block: %s' % str(cfg_cls_block))

        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()

        if pair_input_dim is None:
            pair_input_dim = 4 if pair_input_type == 'pp' else 6
        self.pair_extra_dim = pair_extra_dim
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, (*pair_embed_dims, cfg_block['num_heads']),
            pairwise_lv_type=pair_input_type,
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList([Block(**cfg_cls_block)
                                        for _ in range(num_cls_layers)]) if num_cls_layers > 0 else None
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for param in fc_params:
                try:
                    out_dim, drop_rate, act = param
                except ValueError:
                    (out_dim, drop_rate), act = param, 'relu'
                if act == 'swiglu':
                    layer = nn.Sequential(SwiGLUFFN(in_dim, out_dim * 4, out_dim, drop=drop_rate),
                                          nn.LayerNorm(out_dim))
                else:
                    layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                          nn.GELU() if act == 'gelu' else nn.ReLU(),
                                          nn.Dropout(drop_rate))
                fcs.append(layer)
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # cls tokens
        if not self.for_segmentation and num_cls_layers > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
            nn.init.trunc_normal_(self.cls_token, std=.02)
        else:
            self.cls_token = None

        # weight initialization
        if weight_init is not None:
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.out_proj.weight.data, layer_id + 1)
            rescale(layer.fc2.weight.data, layer_id + 1)

    def init_weights(self, mode: str = '') -> None:
        assert mode in ('timm', 'moco')
        if mode == 'timm':
            named_apply(init_weights_vit_timm, self)
        elif mode == 'moco':
            named_apply(init_weights_vit_moco, self)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def _forward_encoder(self, x, v=None, mask=None, uu=None, uu_idx=None):
        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (batch_size, seq_len)

        with torch.autocast('cuda', enabled=self.use_amp):
            # input embedding
            embedded_x = self.embed(x)
            filled_x = embedded_x.masked_fill(padding_mask.unsqueeze(-1), 0)  # (batch_size, seq_len, num_fts)
            x = filled_x
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                attn_mask = self.pair_embed(v, uu=uu, mask=mask)  # (batch_size, num_heads, seq_len, seq_len)

            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

        # x: (batch, seq_len, embed_dim)
        # padding_mask: (batch, seq_len)
        return x, padding_mask

    def _forward_aggregator(self, x, padding_mask):
        with torch.autocast('cuda', enabled=self.use_amp):
            if self.cls_blocks is not None:
                # for classification: extract using class token
                cls_tokens = self.cls_token.expand(x.size(0), 1, -1)  # (batch, 1, embed_dim)
                for block in self.cls_blocks:
                    cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)  # (batch, 1, embed_dim)
                cls_tokens = cls_tokens.squeeze(1)  # (batch, embed_dim)
            else:
                # for classification: simple average pooling
                mask = ~padding_mask.unsqueeze(1)  # (batch, 1, seq_len)
                x = x.transpose(1, 2).contiguous()  # (batch, embed_dim, seq_len)
                counts = mask.float().sum(-1)  # (batch, 1)
                counts = torch.max(counts, torch.ones_like(counts))  # >=1
                cls_tokens = (x * mask).sum(-1) / counts  # (batch, embed_dim)

            x_cls = self.norm(cls_tokens)  # (batch, embed_dim)
        return x_cls

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (batch_size, num_fts, seq_len)
        # v: (batch_size, 4, seq_len) [px,py,pz,energy]
        # mask: (batch_size, 1, seq_len) -- real particle = 1, padded = 0
        # for pytorch: uu (batch_size, C', num_pairs), uu_idx (batch_size, 2, num_pairs)
        # for onnx: uu (batch_size, C', seq_len, seq_len), uu_idx=None
        x, padding_mask = self._forward_encoder(x, v=v, mask=mask, uu=uu, uu_idx=uu_idx)

        if self.cls_blocks is None and self.fc is None:
            # x: (batch, seq_len, embed_dim)
            # padding_mask: (batch, seq_len)
            return x, padding_mask

        with torch.autocast('cuda', enabled=self.use_amp):
            # === for segmentation ===
            if self.for_segmentation:
                x = self.norm(x)
                if self.fc is not None:
                    x = self.fc(x)
                # x: (P, N, C) -> output: (N, C, P)
                output = x.transpose(1, 2).contiguous()
                if self.for_inference:
                    output = torch.softmax(output, dim=1)
                # print('output:\n', output)
                return output

            x_cls = self._forward_aggregator(x, padding_mask)
            if self.fc is None:
                return x_cls

            # fc
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output



def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim               = len(data_config.input_dicts['pf_features']),
        num_classes             = len(data_config.label_value),
        
        # network configurations
        # pair_input_dim          = 4,
        pair_extra_dim          = 0,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    
    model = ParticleTransformerDVTagger(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()