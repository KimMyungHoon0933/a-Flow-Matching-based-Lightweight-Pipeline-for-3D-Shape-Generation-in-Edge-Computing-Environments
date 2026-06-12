import os
from os import path
from typing import Callable, Any
import json
import functools

import torch
from lightning.app.storage.path import num_workers
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from tqdm import tqdm

import engine
from .base import BaseDataModule
from .hdf5 import HDF5Dataset


class ShapeNetCategoryNames:
    def __init__(self):
        super().__init__()

        with open('assets/shapenet_split/shapenet_synset_dict.json', 'r') as f:
            data = json.load(f)
        self.data = data

    def get_name(self, idx):
        cat_id = CATEGORY_IDS[idx]
        return self.data[cat_id]


class ShapeNetOccupancyDataModule(BaseDataModule):
    def __init__(self,
                 root_dir: str,

                 num_workers: int = 4,
                 batch_size: int = 16,
                 eval_batch_size: int = -1,
                 val_only: bool = False,
                 shuffle: bool = True,

                 raw_root_dir: str = None,
                 skip_queries: bool = False,
                 skip_eval_queries: bool = False,
                 num_samples_per_category: int = -1,
                 num_generation_samples: int = -1,
                 is_unconditional_gen: bool = False,
                 categories=None,
                 use_sampling_dataset: bool = False,
                 return_full_surface: bool = False,
                 use_surface_iou: bool = False,
                 dataset_params=None,
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.root_dir = root_dir
        self.num_workers = num_workers
        self.val_only = val_only
        self.shuffle = shuffle
        self.skip_queries = skip_queries
        self.skip_eval_queries = skip_eval_queries
        self.num_samples_per_category = num_samples_per_category
        self.is_unconditional_gen = is_unconditional_gen
        self.num_generation_samples = num_generation_samples
        self.categories = categories
        self.use_sampling_dataset = use_sampling_dataset
        self.return_full_surface = return_full_surface
        self.use_surface_iou = use_surface_iou

        # for preprocessing
        self.raw_root_dir = raw_root_dir

        self.dataset_builder = functools.partial(ShapeNetOccupancyDataset, **dataset_params)
        pass

    def dataset(self, phase):
        is_train = phase == 'train'
        transform = AxisScaling((0.75, 1.25), True) if is_train else None
        sampling = is_train

        if self.use_sampling_dataset and not is_train:
            num_samples_per_category = self.num_samples_per_category
            return ShapeNetCategoryDataset(self.categories, num_samples_per_category)

        num_samples_per_category = -1
        num_generation_samples = -1
        if self.is_unconditional_gen:
            num_generation_samples = self.num_generation_samples

        return self.dataset_builder(split=phase,
                                    root_dir=self.root_dir,
                                    return_surface=True,
                                    surface_sampling=True,
                                    sampling=sampling,
                                    transform=transform,
                                    skip_queries=self.skip_queries if is_train else self.skip_eval_queries,
                                    num_samples_per_category=num_samples_per_category,
                                    num_generation_samples=num_generation_samples,
                                    categories=self.categories,
                                    return_full_surface=self.return_full_surface,
                                    use_surface_iou=self.use_surface_iou,
                                    )

    def train_dataloader(self):
        return self.dataloader(self.dataset('train'), 'train')

    def val_dataloader(self):
        return self.dataloader(self.dataset('val'), 'val')

    def dataloader(self, dataset: Dataset, phase: str):
        is_train = (phase == 'train') and self.shuffle
        batch_size = self.batch_size if is_train else self.eval_batch_size
        num_workers = self.num_workers if batch_size > self.num_workers else batch_size
        prefetch = 2 if is_train else None
        if len(dataset) < batch_size:
            batch_size = len(dataset)
        if len(dataset) < num_workers:
            num_workers = len(dataset)

        return DataLoader(dataset, batch_size=batch_size,
                          prefetch_factor=prefetch,
                          drop_last=is_train,
                          shuffle=is_train, num_workers=num_workers)

    def available_preprocess_splits(self):
        return ['train', 'val', 'test']

    def preprocess_data(self, split: str):
        os.makedirs(self.root_dir, exist_ok=True)
        query_root_dir = path.join(self.raw_root_dir, 'ShapeNetV2_point')
        surface_root_dir = path.join(self.raw_root_dir, 'ShapeNetV2_surface')

        ## identify items
        items = []
        class_dirs = [x for x in os.listdir(query_root_dir) if os.path.isdir(path.join(query_root_dir, x))]
        for class_dir in class_dirs:
            split_list_path = path.join(query_root_dir, class_dir, f'{split}.lst')
            with open(split_list_path, 'r') as f:
                lines = f.read().split('\n')

            items.extend([(class_dir, x.replace('.npz', '')) for x in lines if x != ''])
            pass

        file = h5py.File(path.join(self.root_dir, f'{split}.h5'), 'w')
        failed_count = 0
        for item in tqdm(items):
            class_id, object_id = item
            try:
                query_data = np.load(path.join(query_root_dir, class_id, f'{object_id}.npz'))
                surface_data = np.load(path.join(surface_root_dir, class_id, '4_pointcloud', f'{object_id}.npz'))
                scale_data = np.load(path.join(query_root_dir, class_id, f'{object_id}.npy'))
                vol_points, vol_label = query_data['vol_points'], query_data['vol_label']
                near_points, near_label = query_data['near_points'], query_data['near_label']
                surface_points = surface_data['points']
            except Exception as e:
                print(e)
                failed_count += 1
                continue

            group = file.create_group(f'{class_id}/{object_id}')
            group['vol_points'] = vol_points.astype(np.float16)
            group['vol_label'] = vol_label
            group['near_points'] = near_points.astype(np.float16)
            group['near_label'] = near_label
            group['surface_points'] = surface_points.astype(np.float32)
            group.attrs['scale'] = scale_data.item()

        file.close()
        print(f'Failed to load {failed_count} items')


"""
ShapeNet Occupancy Dataset. Code mainly borrowed from "https://github.com/1zb/3DShape2VecSet/blob/master/util/shapenet.py"
"""
CATEGORY_IDS = [
    '02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884',
    '02843684', '02871439', '02876657', '02880940', '02924116', '02933112', '02942699',
    '02946921', '02954340', '02958343', '02992529', '03001627', '03046257', '03085013',
    '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137',
    '03593526', '03624134', '03636649', '03642806', '03691459', '03710193', '03759954',
    '03761084', '03790512', '03797390', '03928116', '03938244', '03948459', '03991062',
    '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267',
    '04379243', '04401088', '04460130', '04468005', '04530566', '04554684'
]


class ShapeNetCategoryDataset(Dataset):
    def __init__(self, categories=None, num_samples=10):
        if categories is None:
            categories = CATEGORY_IDS
        categories.sort()

        self.categories = categories
        self.num_samples = num_samples

    def __len__(self):
        return len(self.categories) * self.num_samples

    def __getitem__(self, idx):
        category_idx = idx // self.num_samples
        inner_idx = idx % self.num_samples
        category = self.categories[category_idx]
        return {
            'idx': idx,
            'inner_idx': inner_idx,
            'surface': torch.zeros(1, 3).float(),
            'category_ids': CATEGORY_IDS.index(category),
        }


class ShapeNetOccupancyDataset(HDF5Dataset):
    def __init__(self, root_dir, split=None,
                 categories=None,
                 transform=None,
                 sampling=True,
                 num_query_points=1024,
                 return_surface=True,
                 surface_sampling=True,
                 pc_size=2048,
                 repeat=16,
                 chunk_size=2000,
                 num_items=-1,
                 skip_queries=False,
                 oversample_ratio=10,
                 num_samples_per_category=-1,
                 num_generation_samples=-1,
                 return_full_surface=False,
                 use_surface_iou=False,
                 ):
        file_path = path.join(root_dir, f'{split}.h5')
        super().__init__(file_path)

        self.pc_size = pc_size

        self.transform = transform
        self.num_query_points = num_query_points
        self.sampling = sampling
        self.split = split
        self.oversample_ratio = oversample_ratio

        self.root_dir = root_dir
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling
        self.chunk_size = chunk_size
        self.replica = repeat
        self.skip_queries = skip_queries
        self.return_full_surface = return_full_surface
        self.use_surface_iou = use_surface_iou

        if categories is None:
            categories = CATEGORY_IDS
        categories.sort()

        self.items = []
        for category in categories:
            object_ids = self.file[category].keys()
            if num_samples_per_category > 0:
                object_ids = list(object_ids)[:num_samples_per_category]

            self.items.extend([(category, object_id) for object_id in object_ids])

        ## for debugging
        if num_items > 0:
            self.items = self.items[:num_items]
        elif num_generation_samples > 0:
            indices = np.random.choice(len(self.items), num_generation_samples, replace=False)
            self.items = self.items[indices]
        self.close_file()

    def __getitem__(self, idx):
        idx = idx % len(self.items)
        category, object_id = self.items[idx]
        file = self.file
        group = file[f'{category}/{object_id}']

        vol_points = group['vol_points']
        vol_label = group['vol_label']
        near_points = group['near_points']
        near_label = group['near_label']
        scale = float(group.attrs['scale'])
        surface = None
        full_surface = None
        if self.return_surface:
            surface = group['surface_points']
            full_surface = surface[:]
            if self.surface_sampling:
                ind = np.random.choice(surface.shape[0], self.pc_size, replace=False)
                surface = full_surface[ind]
            else:
                surface = full_surface

            surface = surface * scale
            if self.return_full_surface:
                full_surface = full_surface * scale
            surface = torch.from_numpy(surface.astype(np.float32))

        if self.skip_queries:
            vol_points = np.zeros((1, 3))
            vol_label = np.zeros((1, 3))
            near_points = np.zeros((1, 3))
            near_label = np.zeros((1, 3))
        elif self.sampling:
            vol_points, vol_label = self._two_stage_sampling([vol_points, vol_label], self.num_query_points)
            near_points, near_label = self._two_stage_sampling([near_points, near_label], self.num_query_points)
        else:
            vol_points = vol_points[:]
            vol_label = vol_label[:]
            near_points = near_points[:]
            near_label = near_label[:]
        vol_points = torch.from_numpy(vol_points).float()
        vol_label = torch.from_numpy(vol_label).float()

        if self.split == 'train':
            near_points = torch.from_numpy(near_points)
            near_label = torch.from_numpy(near_label).float()

            points = torch.cat([vol_points, near_points], dim=0)
            labels = torch.cat([vol_label, near_label], dim=0)
        elif self.use_surface_iou:
            points = near_points
            labels = near_label
        else:
            points = vol_points
            labels = vol_label

        max_val = torch.abs(surface).max().item()
        if self.transform:
            surface, points, max_val = self.transform(surface, points)

        item = {
            'idx': idx,
            'query_points': points,
            'labels': labels,
            'category_ids': CATEGORY_IDS.index(category),
            'max_val': max_val,
            'num_vol_points': self.num_query_points,
        }
        if self.return_surface:
            item['surface'] = surface
        if self.return_full_surface:
            item['full_surface'] = torch.from_numpy(full_surface).float()

        return item

    def _two_stage_sampling(self, points_list, num_samples):
        chunk_size = self.chunk_size
        num_total_blocks = points_list[0].shape[0] // chunk_size
        num_blocks = ((num_samples * self.oversample_ratio) // chunk_size) + 1
        block_indices = np.random.choice(num_total_blocks, num_blocks, replace=False)
        block_indices = np.sort(block_indices)
        sampled_points_list = []

        point_indices = None
        for points in points_list:
            blocks = []
            for block_idx in block_indices:
                start = block_idx * chunk_size
                end = min((block_idx + 1) * chunk_size, points_list[0].shape[0])
                blocks.append(points[start:end])

            points = np.concatenate(blocks, axis=0)
            if point_indices is None:
                point_indices = np.random.choice(points.shape[0], num_samples, replace=False)
            sampled_points_list.append(points[point_indices])

        return sampled_points_list

    def __len__(self):
        if self.split != 'train':
            return len(self.items)
        else:
            return len(self.items) * self.replica


class AxisScaling:
    def __init__(self, interval=(0.75, 1.25), jitter=True, eps=1e-10):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        self.eps = eps

    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        point = point * scaling

        ## TODO: clamping
        max_val = max(torch.abs(surface).max().item(), 0.1)
        scale = (1 / max_val) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, point, max_val
