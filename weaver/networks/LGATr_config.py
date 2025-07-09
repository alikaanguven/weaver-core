import torch
import weaver.nn.model.LGATr as LGATr_model
from weaver.utils.logger import _logger


def get_model(data_config, **kwargs):

    cfg = dict(
        in_s_channels            =  len(data_config.input_dicts['pf_features']), # vtx_match as a placeholder (essentially a dummy)
        hidden_mv_channels       = 16,
        hidden_s_channels        =  len(data_config.input_dicts['pf_features']), # vtx_match as a placeholder (essentially a dummy)
        num_classes              =  len(data_config.label_value),
        num_blocks               =  6,
        num_heads                =  8,
        
        # symmetry-breaking configurations
        spurion_token            = True,
        beam_spurion             = "xyplane",
        add_time_spurion         = True,
        beam_mirror              = True,
        
        # network configurations
        global_token                         = True,
        activation                           = "gelu",
        multi_query                          = False,
        increase_hidden_channels_attention   = 2,
        increase_hidden_channels_mlp         = 2,
        num_hidden_layers_mlp                = 1,
        head_scale                           = False,
        dropout_prob                         = None,
        
        # time/memory configurations
        checkpoint_blocks                    = False,
        
        # gatr configurations
        use_fully_connected_subgroup         = True,
        mix_pseudoscalar_into_scalar         = True,
        use_bivector                         = True,
        use_geometric_product                = True,

        # tagger args
        use_amp           = False,
        trim              = True,
        for_inference     = False,
        for_segmentation  = False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = LGATr_model.LGATrTagger(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }
    for i in range(5): print("-"*80)
    print("DEBUG:")
    print(model_info)
    print("DEBUG:")
    print(cfg)
    for i in range(5): print("-"*80)
    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()