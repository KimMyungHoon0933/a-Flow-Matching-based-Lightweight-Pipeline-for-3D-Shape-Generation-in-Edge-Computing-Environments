"""
flowmating_lightning.py — 수정 버전
========================================
변경 사항:
  1. import: kl_d512_m512_l16_flow → cod_m32_d32_flow (또는 호환 이름 유지)
  2. self.n_latents를 flow_model에서 가져오므로 자동 일치
  3. VAE 출력 shape에 대한 방어적 reshape 추가
========================================
"""

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from omegaconf import OmegaConf

from cod.models.vae.vae import CompactLatentVAE

# [수정] 새 팩토리 함수 import (기존 이름도 호환됨)
from CFM.flow_set.models_class_cond import cod_m32_d32_flow

# Meta Flow Matching 라이브러리 임포트
from flow_matching.path.scheduler.scheduler import CondOTScheduler
from flow_matching.path.affine import CondOTProbPath


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
        # [수정] COD-VAE config 기준: n_latents=32, channels=32
        self.flow_model = cod_m32_d32_flow()

        # [수정] n_latents, channels를 flow_model에서 가져옴 → 항상 일치 보장
        self.n_latents = self.flow_model.n_latents
        self.channels = self.flow_model.channels

        # ── 3. Flow Matching 궤적 생성기 ──
        self.prob_path = CondOTProbPath()

        # 차원 확인 로그
        print(f"[FlowMatchingLightning] flow_model: n_latents={self.n_latents}, channels={self.channels}")
        print(f"[FlowMatchingLightning] 기대 입력 shape: [batch, {self.n_latents}, {self.channels}]")

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

            # [수정] VAE 출력 shape에 따라 방어적 reshape
            # 이미 [batch, n_latents, channels]이면 그대로, [batch, flat]이면 reshape
            if x_1.dim() == 2:
                x_1 = x_1.view(x_1.shape[0], self.n_latents, -1)
            elif x_1.dim() == 3 and x_1.shape[1] != self.n_latents:
                # 3차원이지만 n_latents가 다른 경우 flatten 후 reshape
                x_1 = x_1.reshape(x_1.shape[0], self.n_latents, -1)

            # shape 검증
            assert x_1.shape[1] == self.n_latents, \
                f"VAE 출력 n_latents 불일치: 기대={self.n_latents}, 실제={x_1.shape[1]}"
            assert x_1.shape[2] == self.channels, \
                f"VAE 출력 channels 불일치: 기대={self.channels}, 실제={x_1.shape[2]}"

        # ── Flow Matching 학습 ──
        # 1. 순수 노이즈 생성
        x_0 = torch.randn_like(x_1)

        # 2. 시간 스텝 무작위 샘플링 (0.0 ~ 1.0)
        t = torch.rand((x_1.shape[0],), device=x_1.device)

        # 3. OT-CFM 궤적 샘플링
        path_sample = self.prob_path.sample(x_0=x_0, x_1=x_1, t=t)

        with torch.cuda.amp.autocast(enabled=False):
            x_t = path_sample.x_t.float()
            t_input = path_sample.t.float()
            target_v = path_sample.dx_t.float()  # 정답 velocity (x_1 - x_0)

            # 4. 모델 velocity 예측
            predicted_v = self.flow_model(x_t, t_input, class_labels=categories)

            # 5. MSE 손실
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
            weight_decay=self.weight_decay
        )
        return optimizer