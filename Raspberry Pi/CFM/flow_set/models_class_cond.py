"""
models_class_cond.py — 수정 버전
========================================
변경 사항:
  1. LatentArrayTransformer: n_latents 파라미터 추가, pos_emb를 동적 크기로 변경
  2. FlowModel: COD-VAE config.yaml 기준 n_latents=32, channels=32
  3. 팩토리 함수: cod_m32_d32_flow() 로 이름 변경 (기존 kl_d512_m512_l16_flow도 호환용 유지)
  4. EDMPrecond, EDMLoss 등 기존 디퓨전 코드는 그대로 유지 (기존 체크포인트 호환)
========================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
import numpy as np


# ─────────────────────────────────────────────────────────
# 유틸리티 클래스들 (원본 유지)
# ─────────────────────────────────────────────────────────

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        if context_dim is None:
            context_dim = query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        if context is None:
            context = x
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(timestep)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim,
            heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = AdaLayerNorm(dim)
        self.norm2 = AdaLayerNorm(dim)
        self.norm3 = AdaLayerNorm(dim)
        self.checkpoint = checkpoint

        init_values = 0
        drop_path = 0.0

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, t, context=None):
        x = self.drop_path1(self.ls1(self.attn1(self.norm1(x, t)))) + x
        x = self.drop_path2(self.ls2(self.attn2(self.norm2(x, t), context=context))) + x
        x = self.drop_path3(self.ls3(self.ff(self.norm3(x, t)))) + x
        return x


# ─────────────────────────────────────────────────────────
# LatentArrayTransformer — 수정됨
# [변경] n_latents 파라미터 추가, pos_emb를 동적 크기로
# ─────────────────────────────────────────────────────────

class LatentArrayTransformer(nn.Module):
    def __init__(self, in_channels, t_channels, n_heads, d_head,
                 n_latents=512,  # [추가] pos_emb 크기를 결정하는 파라미터
                 depth=1, dropout=0., context_dim=None, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.t_channels = t_channels

        self.proj_in = nn.Linear(in_channels, inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(inner_dim)

        if out_channels is None:
            self.proj_out = nn.Linear(inner_dim, in_channels, bias=False)
        else:
            self.num_cls = out_channels
            self.proj_out = nn.Linear(inner_dim, out_channels, bias=False)

        self.context_dim = context_dim

        self.map_noise = PositionalEmbedding(t_channels)
        self.map_layer0 = nn.Linear(in_features=t_channels, out_features=inner_dim)
        self.map_layer1 = nn.Linear(in_features=inner_dim, out_features=inner_dim)

        # [수정] 512 하드코딩 → n_latents로 동적 설정
        self.pos_emb = nn.Embedding(n_latents, inner_dim)

    def forward(self, x, t, cond=None):
        t_emb = self.map_noise(t)[:, None]
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))

        x = self.proj_in(x)

        # pos_emb.weight shape = [n_latents, inner_dim], x shape = [batch, n_latents, inner_dim]
        x = x + self.pos_emb.weight[None]

        for block in self.transformer_blocks:
            x = block(x, t_emb, context=cond)

        x = self.norm(x)
        x = self.proj_out(x)
        return x


# ─────────────────────────────────────────────────────────
# FlowModel — 수정됨
# [변경] n_latents를 LatentArrayTransformer에 전달
# ─────────────────────────────────────────────────────────

class FlowModel(nn.Module):
    def __init__(self, n_latents=32, channels=32, n_heads=8, d_head=64, depth=12, num_classes=55):
        super().__init__()
        self.n_latents = n_latents
        self.channels = channels
        self.category_emb = nn.Embedding(num_classes, n_heads * d_head)

        self.model = LatentArrayTransformer(
            in_channels=channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            n_latents=n_latents,  # [추가] pos_emb 크기 전달
            depth=depth
        )

    def forward(self, x, t, class_labels=None, **model_kwargs):
        # 카테고리 임베딩 처리
        if class_labels.dtype == torch.float32:
            cond_emb = class_labels
        else:
            cond_emb = self.category_emb(class_labels).unsqueeze(1)

        x = x.to(torch.float32)

        # PositionalEmbedding의 max_positions=10000 기준 설계 → t ∈ [0,1]을 1000배 스케일링
        t_scaled = t.to(torch.float32).reshape(-1) * 1000.0

        # Velocity 예측
        velocity = self.model(x, t_scaled, cond=cond_emb, **model_kwargs)

        return velocity


# ─────────────────────────────────────────────────────────
# 팩토리 함수 — Flow 모델용 (COD-VAE M=32 기준)
# ─────────────────────────────────────────────────────────

def cod_m32_d32_flow():
    """COD-VAE config: num_latents=32, query_dim=32 → [batch, 32, 32]"""
    return FlowModel(n_latents=32, channels=32, num_classes=55)

# 기존 이름 호환용 (flowmating_lightning.py에서 import하는 이름)
def kl_d512_m512_l16_flow():
    """호환용: 실제로는 COD-VAE M=32 설정을 사용"""
    return cod_m32_d32_flow()


# ─────────────────────────────────────────────────────────
# 이하 EDM 관련 코드 (기존 디퓨전 체크포인트 호환용, 변경 없음)
# ─────────────────────────────────────────────────────────

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, inputs, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([inputs.shape[0], 1, 1], device=inputs.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(inputs) if augment_pipe is not None else (inputs, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss.mean()


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


class EDMPrecond(torch.nn.Module):
    def __init__(self,
        n_latents=512,
        channels=8,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float('inf'),
        sigma_data=1,
        n_heads=8,
        d_head=64,
        depth=12,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.channels = channels
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.model = LatentArrayTransformer(
            in_channels=channels, t_channels=256,
            n_heads=n_heads, d_head=d_head,
            n_latents=n_latents,  # [수정] pos_emb 크기도 전달
            depth=depth
        )

        self.category_emb = nn.Embedding(55, n_heads * d_head)

    def emb_category(self, class_labels):
        return self.category_emb(class_labels).unsqueeze(1)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        if class_labels.dtype == torch.float32:
            cond_emb = class_labels
        else:
            cond_emb = self.category_emb(class_labels).unsqueeze(1)

        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), cond=cond_emb, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    @torch.no_grad()
    def sample(self, cond, batch_seeds=None):
        if cond is not None:
            batch_size, device = *cond.shape, cond.device
            if batch_seeds is None:
                batch_seeds = torch.arange(batch_size)
        else:
            device = batch_seeds.device
            batch_size = batch_seeds.shape[0]

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, self.n_latents, self.channels], device=device)
        return edm_sampler(self, latents, cond, randn_like=rnd.randn_like)


# EDM 팩토리 함수들 (기존 호환)
def kl_d512_m512_l8_edm():
    return EDMPrecond(n_latents=512, channels=8)

def kl_d512_m512_l16_edm():
    return EDMPrecond(n_latents=512, channels=16)

def kl_d512_m512_l32_edm():
    return EDMPrecond(n_latents=512, channels=32)

def kl_d512_m512_l4_d24_edm():
    return EDMPrecond(n_latents=512, channels=4, depth=24)

def kl_d512_m512_l8_d24_edm():
    return EDMPrecond(n_latents=512, channels=8, depth=24)

def kl_d512_m512_l32_d24_edm():
    return EDMPrecond(n_latents=512, channels=32, depth=24)