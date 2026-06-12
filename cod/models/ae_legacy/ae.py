import functools
from typing import List, Tuple

import torch
from torch import nn

from pointops.functions import pointops
from .modules.external import PointEmbed
from .modules.blocks import init_embedding
from .networks import PointSetEncoder, TriplaneDecoder
from ..vae.base import BaseAutoencoder

class CompactLatentAutoEncoder(nn.Module):

    def __init__(self,
                 output_dim: int,
                 num_latents: int,
                 embed_dim: int,
                 query_dim: int,

                 encoder_params: dict,
                 decoder_params: dict,

                 use_latent_fps: bool = True,
                 use_init_out: bool = True,
                 ):
        super().__init__()
        encoder_cls = functools.partial(PointSetEncoder, **encoder_params)
        decoder_cls = functools.partial(TriplaneDecoder, **decoder_params)

        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.query_dim = query_dim
        self.use_init_out = use_init_out

        self.use_latent_fps = use_latent_fps

        ## layers
        self.point_embed = PointEmbed(dim=embed_dim)
        self.norm_latent = nn.LayerNorm(embed_dim)
        self.encoder = encoder_cls(embed_dim=embed_dim)
        self.decoder = decoder_cls(embed_dim=embed_dim, query_dim=query_dim)
        self.head = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, output_dim),
        )

        self.latent_pos = nn.Parameter(torch.empty(num_latents, embed_dim)) if not use_latent_fps else None
        self.reset_parameters()

    def reset_parameters(self):
        scale = self.embed_dim ** -0.5
        if self.latent_pos is not None:
            init_embedding(self.latent_pos, scale)

    def forward(self, pc, queries):
        z, posterior = self.encode(pc)
        patches, init_patches, uncertainty, z_recon = self.decode(z)
        out = self.decode_queries(patches, queries)

        outputs = {
            'logits': out,
            'patches': patches,
            'latents': z,
        }
        if (init_patches is not None):
            init_out = self.decode_queries(init_patches, queries)
            outputs['init_out'] = init_out
        if posterior is not None:
            outputs['posterior'] = posterior
        if uncertainty is not None:
            uncertainty_queries = self.decoder.decode_uncertainty(uncertainty, queries)
            outputs['uncertainty'] = uncertainty_queries
            outputs['uncertainty_planes'] = uncertainty

        if hasattr(self.decoder, '_attn') and self.decoder._attn is not None:
            outputs['dec_attn'] = self.decoder._attn

        return outputs

    def encode(self, pc):
        z = self.encode_embed(pc)
        return z, None

    def encode_embed(self, pc):
        z = self._get_init_z(pc)
        z = self.norm_latent(z)

        return self.encoder(pc, z, self.point_embed)

    def _get_init_z(self, pc):
        if self.use_latent_fps:
            z = pointops.fps(pc, self.num_latents)
            z = self.point_embed(z)
        else:
            z = self.latent_pos.unsqueeze(0).expand(pc.size(0), -1, -1)
        return z

    def decode(self, z):
        context, init_context, uncertainty = self.decoder.decode(z)
        return context, init_context, uncertainty, z

    def decode_queries(self, context, queries):
        query_features = self.decoder.decode_queries(context, queries)
        out = self.head(query_features).squeeze(-1)
        return out
