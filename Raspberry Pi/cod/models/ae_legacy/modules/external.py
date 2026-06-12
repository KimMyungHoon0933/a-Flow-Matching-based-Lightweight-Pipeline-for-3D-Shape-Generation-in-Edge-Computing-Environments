"""
External modules from ShapeVec and TiTok
"""

from functools import wraps
from collections import OrderedDict

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F
import einops
from einops import rearrange, repeat

from .timm_layers import DropPath


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, drop_path_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, drop_path_rate=0.0,
                 use_empty_kv=False,
                 use_temperature=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        # self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.t = nn.Parameter(torch.ones(1)) if use_temperature else None

        if use_empty_kv:
            self.empty_k = nn.Parameter(torch.ones(1) * 1e-4)
            self.empty_v = nn.Parameter(torch.zeros(dim_head))
        else:
            self.empty_k = self.empty_v = None

        self._attn = None
        self._attn_logits = None

    def forward(self, x, context=None, pos=None):
        h = self.heads
        context = default(context, x)
        if pos is None:
            pos = context

        q = self.to_q(x)
        k = self.to_k(pos)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if self.t is not None:
            k = F.normalize(k, dim=-1) / (self.t + 1e-10)
            v = F.normalize(v, dim=-1) / (self.t + 1e-10)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.empty_k is not None:
            sim = torch.cat([sim, self.empty_k.expand(sim.size(0), sim.size(1), 1)], dim=-1)
            v = torch.cat([v, self.empty_v.expand(v.size(0), 1, v.size(-1))], dim=1)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        self._attn = attn
        self._attn_logits = sim

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.drop_path(self.to_out(out))

    def compute_sim(self, q, k):
        q = self.to_q(q)
        k = self.to_k(k)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        return sim


class AttentionBuiltin(nn.Module):
    def __init__(self, query_dim, heads=8, drop_path_rate=0.0):
        super().__init__()

        self.heads = heads
        self.attn = nn.MultiheadAttention(query_dim, heads, batch_first=True, dropout=0)
        self.to_out = nn.Linear(query_dim, query_dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self._attn = None

    def forward(self, x, context=None):
        out = self.attn(x, context, context, need_weights=False)[0]
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))  # B x N x C
        return embed

class MultiRangePointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128, freq_start_idx=0, num_ranges=2, num_freqs=10):
        super().__init__()

        assert hidden_dim % 6 == 0
        self.embedding_dim = hidden_dim

        freq_range_size = num_freqs // num_ranges
        freq_start = freq_start_idx * freq_range_size

        e = torch.pow(2, torch.linspace(freq_start, freq_start + freq_range_size, self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))  # B x N x C
        return embed


#### codes from TiTok


class ResidualAttentionBlock(nn.Module):

    _debug = False

    @classmethod
    def enable_debug_mode(cls):
        cls._debug = True
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio=4.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            dropout: float = 0,
            use_gate: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.linear_gate = nn.Linear(d_model, d_model) if use_gate else None
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            activation = act_layer()
            mlp_width = int(d_model * mlp_ratio)
            input_width = mlp_width
            if isinstance(activation, GEGLU):
                input_width *= 2

            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, input_width)),
                ("gelu", activation),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

        self._attn = None
        self._debug = False

    def enable_debug(self):
        self._debug = True

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        if self._debug:
            output, attn = self.attn(x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
            # self._attn = attn
        else:
            output = self.attn(x, x, x, attn_mask=attn_mask, need_weights=self.__class__._debug)[0]
        return output

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        attn_output = self.attention(x=self.ln_1(x), attn_mask=attn_mask)
        attn_output = self.drop_path(attn_output)
        if self.linear_gate is not None:
            gate = self.linear_gate(x)
            attn_output = torch.sigmoid(gate) * attn_output

        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class ResidualCrossAttnBlock(nn.Module):

    _debug = False
    @classmethod
    def enable_debug_mode(cls):
        cls._debug = True
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio=4.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            dropout: float = 0,
            use_self_attn: bool = True,
    ):
        super().__init__()

        self.ln_cross = norm_layer(d_model)
        self.ln_source = norm_layer(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

        if use_self_attn:
            self.ln_1 = norm_layer(d_model)
            self.self_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        else:
            self.ln_1 = None
            self.self_attn = None

        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            activation = act_layer()
            mlp_width = int(d_model * mlp_ratio)
            input_width = mlp_width
            if isinstance(activation, GEGLU):
                input_width *= 2

            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, input_width)),
                ("gelu", activation),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self._debug = False
        self._attn = None

    def attention(self, x: torch.Tensor, attn_layer, source: torch.Tensor = None,
                  attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None, skip_debug=False):
        if source is None:
            source = x

        debug = self.__class__._debug and not skip_debug
        out = attn_layer(x, source, source,
                         attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                         need_weights=debug, average_attn_weights=False)
        if debug:
            self._attn = out[1]
        return out[0]

    def forward(self, x: torch.Tensor, source: torch.Tensor,
                attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None,
                ):
        cross_attn_output = self.attention(self.ln_cross(x), self.cross_attn, self.ln_source(source),
                                           attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.drop_path(cross_attn_output)

        if self.self_attn is not None:
            attn_output = self.attention(self.ln_1(x), self.self_attn, skip_debug=True)
            x = x + self.drop_path(attn_output)

        if self.mlp_ratio > 0:
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class ResidualMultipleCrossAttnBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            num_cross_attn=1,
            mlp_ratio=4.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            dropout: float = 0,
            use_self_attn: bool = True,
    ):
        super().__init__()

        self.ln_cross_list = nn.ModuleList()
        self.ln_source_list = nn.ModuleList()
        self.cross_attn_list = nn.ModuleList()
        for _ in range(num_cross_attn):
            self.ln_cross_list.append(norm_layer(d_model))
            self.ln_source_list.append(norm_layer(d_model))
            self.cross_attn_list.append(nn.MultiheadAttention(d_model, n_head, batch_first=True))

        if use_self_attn:
            self.ln_1 = norm_layer(d_model)
            self.self_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        else:
            self.ln_1 = None
            self.self_attn = None

        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            activation = act_layer()
            mlp_width = int(d_model * mlp_ratio)
            input_width = mlp_width
            if isinstance(activation, GEGLU):
                input_width *= 2

            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, input_width)),
                ("gelu", activation),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self._debug = False
        self._attn = None

    def enable_debug(self):
        self._debug = True

    def attention(self, x: torch.Tensor, attn_layer, source: torch.Tensor = None,
                  attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None, skip_debug=False):
        if source is None:
            source = x

        debug = self._debug and not skip_debug
        out = attn_layer(x, source, source,
                         attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                         need_weights=debug, average_attn_weights=False)
        if debug:
            self._attn = out[1]
        return out[0]

    def forward(self, x: torch.Tensor, source: torch.Tensor,
                attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None):
        for ln_cross, ln_source, cross_attn in zip(self.ln_cross_list, self.ln_source_list, self.cross_attn_list):
            cross_attn_output = self.attention(ln_cross(x), cross_attn, ln_source(source),
                                               attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            x = x + self.drop_path(cross_attn_output)

        if self.self_attn is not None:
            attn_output = self.attention(self.ln_1(x), self.self_attn, skip_debug=True)
            x = x + self.drop_path(attn_output)

        if self.mlp_ratio > 0:
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x
