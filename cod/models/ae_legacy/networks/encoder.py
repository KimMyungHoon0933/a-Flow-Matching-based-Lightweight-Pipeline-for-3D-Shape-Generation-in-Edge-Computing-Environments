from typing import Tuple, List, Callable

import torch
from torch import nn

from pointops.functions import pointops
from ..modules.blocks import GEGLU
from ..modules.external import ResidualAttentionBlock, ResidualCrossAttnBlock
from ..modules.transformer import CrossAttn, FFN


class PointSetEncoder(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 encoder_blocks: List[Tuple[int, int]],

                 use_latent_attn: bool = True,
                 point_update_mode: str = 'patch',
                 patch_updates=None,

                 ## transformer params
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        assert point_update_mode in ('patch', 'z', None)

        self.patch_updates = patch_updates if patch_updates is not None else [True for _ in range(len(encoder_blocks))]
        self.point_update_mode = point_update_mode
        self.num_patches_list = []
        self.encoders = nn.ModuleList()
        self.latent_encoders = nn.ModuleList()
        self.point_update_layers = nn.ModuleList()
        self.patch_update_layers = nn.ModuleList()
        self.norm_point = nn.LayerNorm(embed_dim)
        for i, (num_patches, num_layers) in enumerate(encoder_blocks):
            update_patch = self.patch_updates[i]
            self.num_patches_list.append(num_patches)
            self.encoders.append(_EncoderBlock(embed_dim=embed_dim, num_layers=num_layers,
                                               num_heads=num_heads,
                                               update_patch=update_patch,
                                               mlp_ratio=mlp_ratio, dropout=dropout))
            if self.point_update_mode is not None:
                self.point_update_layers.append(CrossAttn(embed_dim, num_heads, dropout=0))
        for _ in range(len(encoder_blocks) + 1):
            self.latent_encoders.append(ResidualCrossAttnBlock(embed_dim, num_heads,
                                                               act_layer=GEGLU,
                                                               mlp_ratio=mlp_ratio,
                                                               use_self_attn=use_latent_attn,
                                                               dropout=dropout))

        self._attns = []
        self._patch_pos = None

    def forward(self, pc: torch.Tensor, z: torch.Tensor, embed_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        point_features = embed_fn(pc)
        point_features = self.norm_point(point_features)
        patches = None
        for i, encoder in enumerate(self.encoders):
            num_patches = self.num_patches_list[i]
            if (patches is None) or (patches.size(1) != num_patches):
                patch_pos = pointops.fps(pc, num_patches)
                patches = embed_fn(patch_pos)
                patches = self.norm_point(patches)

            patches = encoder(point_features, patches)
            latent_encoder = self.latent_encoders[i]
            z = latent_encoder(z, patches)
            update_layer = self.point_update_layers[i]
            if self.point_update_mode == 'patch':
                source = patches
            elif self.point_update_mode == 'z':
                source = z
            elif self.point_update_mode is None:
                continue
            else:
                raise NotImplementedError
            point_features = point_features + update_layer(point_features, source)

        z = self.latent_encoders[-1](z, point_features)
        return z


class _EncoderBlock(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_layers: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 rescale_cross_attn: bool = False,
                 attn_proj: bool = False,
                 update_patch: bool = True,
                 ):
        super().__init__()

        self.rescale_cross_attn = rescale_cross_attn
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        attn_out_dim = embed_dim if attn_proj else -1
        self.processing_layers = nn.ModuleList()
        self.points2patch = CrossAttn(embed_dim, num_heads, dropout=0, out_dim=attn_out_dim) if update_patch else None
        self.ln_ffn = nn.LayerNorm(embed_dim)
        self.patch_ffn = FFN(embed_dim, act_layer=GEGLU, mlp_ratio=mlp_ratio, dropout=dropout)

        for i in range(num_layers):
            self.processing_layers.append(ResidualAttentionBlock(embed_dim, num_heads,
                                                                 act_layer=GEGLU,
                                                                 mlp_ratio=mlp_ratio,
                                                                 dropout=dropout))

    def forward(self, point_features, patches):
        if self.points2patch is not None:
            patches = patches + self.points2patch(patches, point_features)
        patches = patches + self.patch_ffn(self.ln_ffn(patches))

        for layer in self.processing_layers:
            patches = layer(patches)

        return patches
