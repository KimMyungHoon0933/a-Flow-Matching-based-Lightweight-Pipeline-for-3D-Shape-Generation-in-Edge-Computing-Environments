import torch
import torch.nn as nn
import lightning.pytorch as pl
from omegaconf import OmegaConf

from cod.models.vae.vae import CompactLatentVAE
from duffusion.Shape2VecSet.models_class_cond import EDMPrecond

class CategoryConditionedDiffusion(pl.LightningModule):
    def __init__(self, vae_config_path, vae_weights_path, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        # 연산 정밀도 설정
        torch.set_float32_matmul_precision('high')

        # 1. Phase: VAE 로드 및 완전 동결
        vae_full_config = OmegaConf.load(vae_config_path)
        vae_config_node = vae_full_config.get('model', vae_full_config)
        vae_params = vae_config_node.get('params', vae_config_node)

        exclude_keys = ['_target_', '_base_', '_overwrite_']
        filtered_params = {k: v for k, v in vae_params.items() if k not in exclude_keys}

        self.vae = CompactLatentVAE(**filtered_params)
        
        ckpt = torch.load(vae_weights_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))

        new_state_dict = { (k.replace('model.', '', 1) if k.startswith('model.') else k): v for k, v in state_dict.items() }
        self.vae.load_state_dict(new_state_dict, strict=False)
        self.vae.eval()
        self.vae.requires_grad_(False)

        # 2. Phase: 디퓨전 모델 연결
        self.n_latents = 32
        self.channels = filtered_params.get('latent_dim', 512) 
        self.diffusion_model = EDMPrecond(n_latents=self.n_latents, channels=self.channels)
        self.sigma_data = 0.5

    def edm_loss(self, net, x, categories, sigma):
        """EDM Loss 계산 (Dtype 및 Autocast 문제 해결)"""
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        # [핵심 수정] 정밀도 문제 해결
        # EDMPrecond의 assertion을 통과하기 위해 autocast를 비활성화하고 float32로 강제합니다.
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            sigma = sigma.float()
            # sigma가 [B, 1, 1]이므로 unsqueeze 없이 곱해도 3차원이 유지됩니다.
            n = torch.randn_like(x) * sigma
            
            # 모델 호출
            D_yn = net(x + n, sigma, class_labels=categories) 
            
            # 출력값 D_yn과 x가 모두 float32이므로 손실 계산이 안전합니다.
            loss = weight * ((D_yn - x) ** 2)
            
        return loss.mean()

    def _shared_step(self, batch):
        if isinstance(batch, dict):
            points = batch['surface']
            categories = batch['category_ids']
        else:
            points = batch[0]
            categories = batch[-1]

        with torch.no_grad():
            embed_features = self.vae.encode_embed(points)
            z, _ = self.vae.encode_latents(embed_features) 
            
            # 차원 정렬 [Batch, 32, Channels]
            batch_size = z.shape[0]
            z = z.view(batch_size, self.n_latents, -1)
        
        # sigma 생성 [B, 1, 1]
        rnd_normal = torch.randn([z.shape[0], 1, 1], device=z.device)
        sigma = (rnd_normal * 1.2 - 1.2).exp()

        return self.edm_loss(self.diffusion_model, z, categories, sigma)

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.diffusion_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)