import functools

import torch

from external.pointops.pointops.functions import pointops
from cod.data.base import BaseDataModule
from cod.models.vae.base import BaseAutoencoder
from .base import BaseSolver
from cod.utils.vis import points_to_img
from cod.metrics.occupancy import Accuracy, IoU
from cod.losses.recon import OccupancyReconstructionLoss, UncertaintyLoss
from cod.utils.sched import CosineAnnealingLR, get_lr
from cod.utils.training import compute_effective_lr
from cod.utils.recon import chunked_reconstruct


class AutoencoderSolver(BaseSolver):

    def __init__(self,
                 dm: BaseDataModule,
                 model: BaseAutoencoder,
                 lr: float = 1e-4,
                 use_cosine_annealing: bool = False,
                 warmup_epochs: int = 5,
                 logit_threshold: float = 0,

                 eval_chunk_size: int = 100000,
                 val_vis_interval: int = 10,
                 coeff_uncertainty: float = 0,
                 coeff_init: float = 1,
                 ):
        super().__init__(track_max_score=True)

        self.lr = lr
        self.use_cosine_annealing = use_cosine_annealing
        self.warmup_epochs = warmup_epochs
        self.scheduler = None
        self.eval_chunk_size = eval_chunk_size
        self.val_vis_interval = val_vis_interval
        self.coeff_uncertainty = coeff_uncertainty
        self.coeff_init = coeff_init

        self.model = model
        self.dm = dm
        self.logit_threshold = logit_threshold

        self.criterion = OccupancyReconstructionLoss(vol_coeff=1.0, near_coeff=0.1)
        self.uncertainty_criterion = UncertaintyLoss()
        self.accuracy_metric = Accuracy()
        self.iou_metric = IoU()

        self.make_points_img = functools.partial(points_to_img, zdir='y', view_angle=(30, -45), point_size=3)
        self.create_image_log_buffer('recon_gt', log_once=True)
        self.create_image_log_buffer('recon')
        self.create_image_log_buffer('uncertainty')

    def configure_optimizers(self):
        lr = compute_effective_lr(self.lr, self.dm.batch_size, self.get_trainer_if_exists())
        params = [x for x in self.parameters() if x.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr)
        if self.use_cosine_annealing:
            self.scheduler = CosineAnnealingLR(optimizer, warmup_epochs=self.warmup_epochs,
                                               total_epochs=self.trainer.max_epochs, lr=lr, min_lr=1e-6)

        return [optimizer], []

    def training_step(self, batch, batch_idx):
        pc = batch['surface']
        query_points = batch['query_points']
        outputs = self.model(pc, query_points)
        ## loss
        logits = outputs['logits']
        labels = batch['labels']
        num_vol_points = batch['num_vol_points'][0].item()

        recon_loss, vol_loss, near_loss = self.criterion(logits, labels, num_vol_points=num_vol_points)
        self.log('train/vol_loss', vol_loss.item(), rank_zero_only=True)
        self.log('train/near_loss', near_loss.item(), rank_zero_only=True)
        self.log('train/recon_loss', recon_loss.item(), rank_zero_only=True)

        init_out = outputs['init_out']
        init_loss = self.criterion(init_out, labels, num_vol_points)[0]
        self.log('train/init_loss', init_loss.item(), rank_zero_only=True)

        uncertainty_loss = self.uncertainty_criterion(outputs['uncertainty'], init_out, labels)
        self.log('train/uncertainty_loss', uncertainty_loss.item(), rank_zero_only=True)

        loss = recon_loss + init_loss * self.coeff_init + uncertainty_loss * self.coeff_uncertainty
        self.log('train/loss', loss.item())

        return {
            'loss': loss,
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.scheduler is not None:
            epoch = self.trainer.global_step * self.trainer.accumulate_grad_batches / len(self.trainer.train_dataloader)
            self.scheduler.step(epoch)

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
