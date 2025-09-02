import torch
import torch.nn as nn
from weaver.utils.logger import _logger

# Import the fixed μP implementation
from example_ParticleTransformer_muP import make_mup_model

'''
Fixed μP implementation wrapper for submission script compatibility.
This provides the get_model and get_loss functions that the training scripts expect.
'''


class ParticleTransformerMuPFixed(torch.nn.Module):
    """
    Wrapper for the fixed μP ParticleTransformer to make it compatible with training scripts.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        # Store architectural parameters
        self.embed_dims = kwargs.get('embed_dims', [128, 512, 128])
        self.pair_embed_dims = kwargs.get('pair_embed_dims', [64, 64, 64])
        self.num_heads = kwargs.get('num_heads', 8)
        self.num_layers = kwargs.get('num_layers', 8)
        self.num_cls_layers = kwargs.get('num_cls_layers', 2)
        
        # Create the fixed μP model
        self.mod = make_mup_model(kwargs)
        
        # Log μP configuration
        _logger.info(f'Applied FIXED μP parameterization for {self.embed_dims[1]}d x {self.num_layers}L')
        _logger.info(f'Using custom μP implementation with CORRECT 1/d_k attention scaling')
        _logger.info(f'This ensures the same learning rate is optimal across all model sizes!')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token'} if hasattr(self.mod, 'cls_token') else set()

    def forward(self, points, features, lorentz_vectors, mask):
        """Forward pass with μP scaling applied by μP layers."""
        return self.mod(features, v=lorentz_vectors, mask=mask)

    def get_mup_info(self):
        """Get μP scaling information for debugging."""
        return {
            'd_model': self.embed_dims[1],
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'embed_dims': self.embed_dims,
            'pair_embed_dims': self.pair_embed_dims,
            'scaling_method': 'custom μP implementation (FIXED)',
            'attention_scaling': '1/d_k (NOT 1/sqrt(d_k)) - correct μP scaling',
            'total_params': sum(p.numel() for p in self.parameters())
        }

    def get_optimal_lr_for_architecture(self, base_lr, task_complexity=1.0):
        """
        Get optimal learning rate for this architecture using μP theory.
        
        According to μP, the optimal learning rate should be the same across
        all architectures when proper parameterization is applied.
        """
        # μP theory: same learning rate should work across all architectures
        optimal_lr = base_lr * task_complexity
        
        _logger.info(f'μP optimal LR: {optimal_lr} (should work for all architectures)')
        
        return optimal_lr


def get_model(data_config, **kwargs):
    """
    Create a ParticleTransformer with FIXED μ-Parameterization for hyperparameter transfer.
    
    This function creates a model that demonstrates the μP principle:
    the same learning rate is optimal across all model architectures.
    """
    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('FIXED μP Model config: %s' % str(cfg))

    model = ParticleTransformerMuPFixed(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
