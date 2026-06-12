import os
from os import path
import json
import functools

import numpy as np
import trimesh

import engine
from cod.data.base import BaseDataModule
from cod.models.vae.base import BaseAutoencoder
from .base import BaseSolver
from cod.utils.recon import create_grid_queries, occupancy_to_mesh, chunked_reconstruct, compute_cd_of_mesh
from cod.utils.mp import MultiProcessRunner
from cod.metrics.occupancy import Accuracy, IoU


class ReconEvaluator(BaseSolver):

    def __init__(self,
                 dm: BaseDataModule,
                 model: BaseAutoencoder,
                 eval_chunk_size: int = 250000,
                 logit_threshold: float = 0,
                 cd_threshold: float = 0.02,
                 force_mesh: bool = False,
                 num_cd_workers: int = 16,
                 ):
        super().__init__()

        self.model = model
        self.eval_chunk_size = eval_chunk_size
        self.logit_threshold = logit_threshold
        self.cd_threshold = cd_threshold
        self.force_mesh = force_mesh
        self.grid_resolution = dm.eval_grid_size
        self.num_cd_workers = num_cd_workers

        self.accuracy_metric = Accuracy()
        self.iou_metric = IoU()
        grid_queries = create_grid_queries(dm.eval_grid_size)
        self.register_buffer('grid_queries', grid_queries, persistent=False)

        self.output_dir = engine.to_experiment_dir('outputs/recon')
        self.mesh_dir = path.join(self.output_dir, 'mesh')

    def on_test_start(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.mesh_dir, exist_ok=True)
        self.model.eval()

    def test_step(self, batch, batch_idx):
        indices = batch['idx']
        pc = batch['surface']
        query_points = batch['query_points']
        labels = batch['labels']
        eval_chunk_size = self.eval_chunk_size if self.eval_chunk_size > 0 else query_points.size(1)

        ## volumetric iou
        logits, preds = chunked_reconstruct(self.model, pc, query_points, eval_chunk_size, self.logit_threshold)
        self.accuracy_metric.update(preds, labels)
        self.iou_metric.update(preds, labels)

        ## saving meshes for CD
        grid_queries = self.grid_queries.expand(pc.size(0), -1, -1)
        logits, preds = chunked_reconstruct(self.model, pc, grid_queries,
                                            eval_chunk_size, self.logit_threshold)
        for b in range(pc.size(0)):
            idx = indices[b].item()
            if preds[b].sum().item() == 0:
                ## handling empty prediction
                continue
            mesh_path = path.join(self.mesh_dir, f'{idx}.obj')
            if path.exists(mesh_path) and not self.force_mesh:
                continue
            mesh = occupancy_to_mesh(logits[b], self.grid_resolution)
            if (mesh.vertices.shape[0] == 0) or (mesh.faces.shape[0] == 0):
                continue
            mesh.export(mesh_path)

    def on_test_end(self):
        iou = self.iou_metric.compute().item()
        accuracy = self.accuracy_metric.compute().item()
        print(f'accuracy: {accuracy:.1f}%, iou: {iou:.1f}%')
        iou_result_path = path.join(self.output_dir, 'iou.json')
        with open(iou_result_path, 'w') as f:
            json.dump({'iou': iou, 'accuracy': accuracy}, f)

    def measure_cd(self, test_dataset):
        indices = list(range(len(test_dataset)))
        worker_fn = functools.partial(self._measure_cd_one_item, dataset=test_dataset)

        runner = MultiProcessRunner(indices, worker_fn=worker_fn,
                                    num_workers=self.num_cd_workers)
        outputs = runner.run()
        all_cd = []
        all_fscore = []
        for cat_id, cd, f_score in outputs:
            all_cd.append(cd)
            all_fscore.append(f_score)

        cd = sum(all_cd) / len(all_cd)
        fscore = sum(all_fscore) / len(all_fscore)
        print(f'Avg. CD: {cd:.4f} F-score: {fscore * 100:.1f}')
        cd_result_path = path.join(self.output_dir, 'cd.json')
        with open(cd_result_path, 'w') as f:
            json.dump({'cd': cd, 'fscore': fscore}, f)

    def _get_test_cd_dataset(self, dm):
        dataset = dm.get_dataset('test')
        dataset.use_queries = False
        dataset.use_full_surface = True
        return dataset

    def _measure_cd_one_item(self, idx, _, dataset):
        item = dataset[idx]
        cat_id = item['category_ids']
        mesh_path = path.join(self.mesh_dir, f'{idx}.obj')
        if not path.exists(mesh_path):
            print('not exist:', idx)
            mesh = None
        else:
            mesh = trimesh.load(mesh_path, force='obj')
            if isinstance(mesh, trimesh.PointCloud):
                print('invalid obj: ', idx)
                mesh = None

        surface = item['full_surface'].numpy()
        cd, fscore = compute_cd_of_mesh(mesh, surface, threshold=self.cd_threshold)

        return cat_id, cd, fscore
