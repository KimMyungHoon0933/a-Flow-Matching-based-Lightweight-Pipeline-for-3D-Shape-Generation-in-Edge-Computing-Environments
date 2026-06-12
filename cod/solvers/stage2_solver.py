import functools

import torch
from torch import nn
from torch.nn import functional as F

from external.pointops.pointops.functions import pointops
from cod.data.base import BaseDataModule
from cod.models.vae.base import BaseAutoencoder
from .base import BaseSolver
from cod.utils.vis import points_to_img
from cod.metrics.occupancy import Accuracy, IoU
from cod.losses.recon import OccupancyReconstructionLoss, UncertaintyLoss
from cod.utils.sched import get_lr
from cod.utils.training import compute_effective_lr, load_model_weights_from_checkpoint
from cod.utils.recon import chunked_reconstruct


class VAEStage2Solver(BaseSolver):

    def __init__(self,
                 dm: BaseDataModule,
                 model: BaseAutoencoder,
                 autoencoder_checkpoint_path: str,
                 lr: float = 1e-4,
                 logit_threshold: float = 0,

                 eval_chunk_size: int = 100000,
                 val_vis_interval: int = 10,
                 coeff_kl: float = 1e-3,
                 coeff_feat: float = 1.0,
                 coeff_recon: float = 1.0,
                 ):
        super().__init__(track_max_score=True)

        self.lr = lr
        self.eval_chunk_size = eval_chunk_size
        self.val_vis_interval = val_vis_interval
        self.coeff_kl = coeff_kl
        self.coeff_feat = coeff_feat
        self.coeff_recon = coeff_recon

        self.model = model
        self.dm = dm
        self.logit_threshold = logit_threshold

        self.targets_norm = nn.LayerNorm(model.embed_dim, elementwise_affine=False)
        self.criterion = OccupancyReconstructionLoss(vol_coeff=1.0, near_coeff=0.1)
        self.uncertainty_criterion = UncertaintyLoss()
        self.accuracy_metric = Accuracy()
        self.iou_metric = IoU()

        self.make_points_img = functools.partial(points_to_img, zdir='y', view_angle=(30, -45), point_size=3)
        self.create_image_log_buffer('recon_gt', log_once=True)
        self.create_image_log_buffer('recon')

        state_dict = load_model_weights_from_checkpoint(autoencoder_checkpoint_path, prefix='model')
        self.model.load_autoencoder_weights(state_dict)

    def configure_optimizers(self):
        lr = compute_effective_lr(self.lr, self.dm.batch_size, self.get_trainer_if_exists())
        params = [x for x in self.parameters() if x.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 70, 80, 90], gamma=0.5)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        pc = batch['surface']
        query_points = batch['query_points']
        labels = batch['labels']
        with torch.no_grad():
            z_enc = self.model.encode_embed(pc)

        z, posterior = self.model.encode_latents(z_enc)
        z_recon = self.model.decode_latents(z)
        z_enc = self.targets_norm(z_enc)
        feat_loss = F.mse_loss(z_recon, z_enc.detach())

        recon = self.model.decode_embed(z_recon)[0]
        logits = self.model.decode_queries(recon, query_points)
        recon_loss = self.criterion(logits, labels)[0]

        kl_loss = self._compute_kl_loss(posterior)

        loss = feat_loss * self.coeff_feat + recon_loss * self.coeff_recon + kl_loss * self.coeff_kl
        self.log('train/feat_loss', feat_loss.item(), rank_zero_only=True)
        self.log('train/recon_loss', recon_loss.item(), rank_zero_only=True)
        self.log('train/kl_loss', kl_loss.item(), rank_zero_only=True)
        self.log('train/loss', loss.item(), rank_zero_only=True)

        return {
            'loss': loss,
        }

    def _compute_kl_loss(self, posterior):
        if not hasattr(posterior, 'kl'):
            return None
        kl = posterior.kl()
        kl_loss = (torch.sum(kl) / kl.size(0)) * self.coeff_kl

        return kl_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        lr = get_lr(self.trainer.optimizers)
        self.log('train/lr', lr, on_step=True, on_epoch=False, rank_zero_only=True)

    def validation_step(self, batch, batch_idx):
        pc = batch['surface']
        query_points = batch['query_points']
        labels = batch['labels']
        eval_chunk_size = self.eval_chunk_size if self.eval_chunk_size > 0 else query_points.size(1)
        if self.is_debug:
            ## for sanity check
            eval_chunk_size = 10000
            query_points = query_points[:, :eval_chunk_size * 2]
            labels = labels[:, :eval_chunk_size * 2]

        logits, preds = chunked_reconstruct(self.model, pc, query_points, eval_chunk_size, self.logit_threshold)

        ## visualization (set val_vis_interval = -1 to disable)
        batch_size = pc.size(0)
        for b in range(pc.size(0)):
            idx = batch_idx * batch_size + b
            if (self.val_vis_interval > 0) and (idx % self.val_vis_interval == 0):
                pred_points = query_points[b, preds[b].bool()]
                if pred_points.size(1) > 2048:
                    pred_points = pointops.fps(pred_points.unsqueeze(0).float(), 2048)[0]

                pred_img = self.make_points_img(pred_points)
                gt_img = self.make_points_img(pc[b])
                self.add_log_buffer_items('recon', pred_img)
                self.add_log_buffer_items('recon_gt', gt_img)

        self.accuracy_metric.update(preds, labels)
        self.iou_metric.update(preds, labels)

    def on_validation_epoch_end(self):
        self.log_buffered()
        accuracy = self.accuracy_metric.compute()
        iou = self.iou_metric.compute()

        self.log('val/accuracy', accuracy, on_epoch=True, rank_zero_only=True, sync_dist=True)
        self.log('val/iou', iou, on_epoch=True, rank_zero_only=True, sync_dist=True)
        self.track_score(iou)
        print(f'accuracy: {accuracy:.1f}%, iou: {iou:.1f}%')
