"""
flowmating_lightning_test.py — time conditioning 테스트 버전
========================================
기존 flowmating_lightning.py와 동일하되,
time conditioning만 변경: t*1000 → log((1-t)/t)/4

사용법: train_flow.py에서 import만 바꾸면 됨
  from CFM.flowmating_lightning_test import FlowMatchingLightning
========================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from omegaconf import OmegaConf
import numpy as np
import ot

from cod.models.vae.vae import CompactLatentVAE

from flow_matching.path.scheduler.scheduler import CondOTScheduler
from flow_matching.path.affine import CondOTProbPath

# 기존 모델 아키텍처 재사용
from CFM.flow_set.models_class_cond import LatentArrayTransformer


def minibatch_ot_coupling(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    B = x_0.shape[0]
    device = x_0.device
    x_0_flat = x_0.reshape(B, -1).float().detach().cpu()
    x_1_flat = x_1.reshape(B, -1).float().detach().cpu()
    a = np.ones(B) / B
    b = np.ones(B) / B
    M = torch.cdist(x_0_flat, x_1_flat, p=2).pow(2).numpy()
    T = ot.emd(a, b, M)
    indices = np.argmax(T, axis=0)
    return x_0[torch.from_numpy(indices).long().to(device)]


class FlowModel(nn.Module):
    """기존 FlowModel과 동일하되 time conditioning만 변경."""
    def __init__(self, n_latents=32, channels=32, n_heads=8, d_head=64,
                 depth=12, num_classes=55):
        super().__init__()
        self.n_latents = n_latents
        self.channels = channels
        self.category_emb = nn.Embedding(num_classes, n_heads * d_head)

        self.model = LatentArrayTransformer(
            in_channels=channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            n_latents=n_latents,
            depth=depth
        )

    def forward(self, x, t, class_labels=None, **model_kwargs):
        if class_labels.dtype == torch.float32:
            cond_emb = class_labels
        else:
            cond_emb = self.category_emb(class_labels).unsqueeze(1)

        x = x.to(torch.float32)
        t_val = t.to(torch.float32).reshape(-1)

        # time conditioning: EDM의 c_noise 방식
        # log((1-t)/t)/4 → t=0(노이즈)에서 큰 값, t=1(데이터)에서 작은 값
        t_clamped = t_val.clamp(min=1e-4, max=1 - 1e-4)
        c_noise = torch.log((1 - t_clamped) / t_clamped) / 4.0

        velocity = self.model(x, c_noise, cond=cond_emb, **model_kwargs)
        return velocity


class FlowMatchingLightning(pl.LightningModule):
    def __init__(self, vae_config_path, vae_weights_path, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        torch.set_float32_matmul_precision('high')

        # ── 1. VAE 로드 및 동결 ──
        vae_full_config = OmegaConf.load(vae_config_path)
        vae_config_node = vae_full_config.get('model', vae_full_config)
        exclude_keys = ['_target_', '_base_', '_overwrite_']
        filtered_params = {k: v for k, v in vae_config_node.items() if k not in exclude_keys}

        self.vae = CompactLatentVAE(**filtered_params)

        ckpt = torch.load(vae_weights_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))
        new_state_dict = {
            (k.replace('model.', '', 1) if k.startswith('model.') else k): v
            for k, v in state_dict.items()
        }
        self.vae.load_state_dict(new_state_dict, strict=False)
        self.vae.eval()
        self.vae.requires_grad_(False)

        # ── 2. Flow 모델 초기화 ──
        self.flow_model = FlowModel(n_latents=32, channels=32, num_classes=55)
        self.n_latents = self.flow_model.n_latents
        self.channels = self.flow_model.channels

        # ── 3. CondOT path (기존과 동일) ──
        self.prob_path = CondOTProbPath()

        print(f"[TEST] time conditioning: log((1-t)/t)/4")
        print(f"[TEST] n_latents={self.n_latents}, channels={self.channels}")

    def _shared_step(self, batch):
        if isinstance(batch, dict):
            points = batch['surface']
            categories = batch['category_ids']
        else:
            points = batch[0]
            categories = batch[-1]

        # ── VAE 인코딩 ──
        with torch.no_grad():
            embed_features = self.vae.encode_embed(points)
            x_1, _ = self.vae.encode_latents(embed_features)

            if x_1.dim() == 2:
                x_1 = x_1.view(x_1.shape[0], self.n_latents, -1)
            elif x_1.dim() == 3 and x_1.shape[1] != self.n_latents:
                x_1 = x_1.reshape(x_1.shape[0], self.n_latents, -1)

            assert x_1.shape[1] == self.n_latents
            assert x_1.shape[2] == self.channels

        # ── Flow Matching 학습 ──
        x_0 = torch.randn_like(x_1)
        x_0 = minibatch_ot_coupling(x_0, x_1)

        t = torch.rand((x_1.shape[0],), device=x_1.device)
        path_sample = self.prob_path.sample(x_0=x_0, x_1=x_1, t=t)

        with torch.cuda.amp.autocast(enabled=False):
            x_t = path_sample.x_t.float()
            t_input = path_sample.t.float()
            target_v = path_sample.dx_t.float()

            predicted_v = self.flow_model(x_t, t_input, class_labels=categories)

            loss = F.mse_loss(predicted_v, target_v)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.flow_model.parameters(),
            lr=self.lr,
            weight_decay=0.05
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
