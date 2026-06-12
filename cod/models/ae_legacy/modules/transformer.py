import torch
from torch import nn

from .timm_layers import DropPath
from .blocks import GEGLU


class CrossAttn(nn.Module):
    _debug = False

    @classmethod
    def enable_debug_mode(cls):
        cls._debug = True

    def __init__(
            self,
            d_model,
            n_head,
            norm_layer=nn.LayerNorm,
            dropout: float = 0,
            out_dim: int = -1,
    ):
        super().__init__()

        self.ln_cross = norm_layer(d_model)
        self.ln_source = norm_layer(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.to_out = nn.Linear(d_model, out_dim) if out_dim > 0 else nn.Identity()
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self._attn = None

    def forward(self, x: torch.Tensor, source: torch.Tensor):
        x = self.ln_cross(x)
        source = self.ln_source(source)
        outputs = self.cross_attn(x, source, source, need_weights=CrossAttn._debug, average_attn_weights=False)
        attn_out = outputs[0]

        if CrossAttn._debug:
            attn_weights = outputs[1]
            self._attn = attn_weights
        return self.drop_path(self.to_out(attn_out))


class FFN(nn.Module):

    def __init__(self, d_model,
                 mlp_ratio=4.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 dropout: float = 0,
                 ):
        super().__init__()
        activation = act_layer()
        mlp_width = int(d_model * mlp_ratio)
        input_width = mlp_width
        if isinstance(activation, GEGLU):
            input_width *= 2

        self.ln = norm_layer(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, input_width),
            activation,
            nn.Linear(mlp_width, d_model),
        )
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.mlp(self.ln(x)))
