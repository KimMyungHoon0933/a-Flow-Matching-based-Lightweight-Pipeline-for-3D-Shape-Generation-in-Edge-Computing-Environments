"""
find_nearest_gt.py — 생성된 메쉬와 가장 유사한 ShapeNet GT를 Chamfer Distance로 검색
====================================================================================
사용법:
  python find_nearest_gt.py \
    --mesh output_flow.obj \
    --test_h5 /data/kimm0902_files/datasets/shapenet/test.h5 \
    --category chair \
    --num_points 2048 \
    --top_k 5
====================================================================================
"""

import argparse
import numpy as np
import torch
import trimesh
import h5py
from tqdm import tqdm

SHAPENET_CATEGORY_IDS = [
    '02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884',
    '02843684', '02871439', '02876657', '02880940', '02924116', '02933112', '02942699',
    '02946921', '02954340', '02958343', '02992529', '03001627', '03046257', '03085013',
    '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137',
    '03593526', '03624134', '03636649', '03642806', '03691459', '03710193', '03759954',
    '03761084', '03790512', '03797390', '03928116', '03938244', '03948459', '03991062',
    '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267',
    '04379243', '04401088', '04460130', '04468005', '04530566', '04554684'
]

CATEGORY_NAME_TO_SYNSET = {
    "airplane": "02691156", "car": "02958343", "chair": "03001627",
    "table": "04379243", "watercraft": "04530566", "sofa": "04256520",
    "rifle": "04090263", "lamp": "03636649"
}


def chamfer_distance(pc1, pc2):
    """두 포인트 클라우드 간 Chamfer Distance (numpy)"""
    # pc1: [N, 3], pc2: [M, 3]
    diff = pc1[:, None, :] - pc2[None, :, :]  # [N, M, 3]
    dist = (diff ** 2).sum(axis=-1)            # [N, M]
    cd = dist.min(axis=1).mean() + dist.min(axis=0).mean()
    return cd


def main():
    parser = argparse.ArgumentParser("생성 메쉬와 가장 유사한 ShapeNet GT 검색")
    parser.add_argument("--mesh", type=str, required=True, help="생성된 .obj 파일 경로")
    parser.add_argument("--test_h5", type=str, default="/data/kimm0902_files/datasets/shapenet/test.h5")
    parser.add_argument("--train_h5", type=str, default="/data/kimm0902_files/datasets/shapenet/train.h5")
    parser.add_argument("--category", type=str, default="chair")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--top_k", type=int, default=5, help="상위 K개 유사 GT 출력")
    parser.add_argument("--search", type=str, default="both", choices=["test", "train", "both"],
                        help="검색 대상 (test/train/both)")
    args = parser.parse_args()

    synset = CATEGORY_NAME_TO_SYNSET.get(args.category)
    if synset is None:
        raise ValueError(f"지원하지 않는 카테고리: {args.category}\n지원: {list(CATEGORY_NAME_TO_SYNSET.keys())}")

    # ── 생성 메쉬 로드 ──
    print(f"생성 메쉬 로딩: {args.mesh}")
    mesh = trimesh.load(args.mesh, force='mesh')
    gen_pc = mesh.sample(args.num_points).astype(np.float32)
    print(f"  vertices={mesh.vertices.shape[0]}, faces={mesh.faces.shape[0]}")
    print(f"  샘플링 포인트: {gen_pc.shape}, 범위: [{gen_pc.min():.3f}, {gen_pc.max():.3f}]")

    # ── GT 검색 ──
    h5_files = []
    if args.search in ["test", "both"] and args.test_h5:
        h5_files.append(("test", args.test_h5))
    if args.search in ["train", "both"] and args.train_h5:
        h5_files.append(("train", args.train_h5))

    all_results = []

    for split_name, h5_path in h5_files:
        print(f"\n{split_name}.h5에서 카테고리 '{args.category}' ({synset}) 검색 중...")

        try:
            with h5py.File(h5_path, 'r') as f:
                if synset not in f:
                    print(f"  경고: {synset}이 {h5_path}에 없음")
                    continue

                category_group = f[synset]
                object_ids = list(category_group.keys())
                print(f"  오브젝트 수: {len(object_ids)}")

                for obj_id in tqdm(object_ids, desc=f"CD 계산 ({split_name})"):
                    obj_group = category_group[obj_id]

                    if 'surface_points' not in obj_group:
                        continue

                    surface = obj_group['surface_points'][:]
                    scale = float(obj_group.attrs.get('scale', 1.0))
                    surface = surface * scale

                    # 포인트 샘플링
                    if surface.shape[0] >= args.num_points:
                        idx = np.random.choice(surface.shape[0], args.num_points, replace=False)
                    else:
                        idx = np.random.choice(surface.shape[0], args.num_points, replace=True)
                    gt_pc = surface[idx].astype(np.float32)

                    cd = chamfer_distance(gen_pc, gt_pc)
                    all_results.append((cd, obj_id, synset, split_name))

        except FileNotFoundError:
            print(f"  파일 없음: {h5_path}")
            continue

    if len(all_results) == 0:
        print("검색 결과 없음")
        return

    # ── 결과 정렬 ──
    all_results.sort(key=lambda x: x[0])

    print(f"\n{'='*60}")
    print(f"  상위 {args.top_k}개 유사 GT (카테고리: {args.category})")
    print(f"  검색 대상: {len(all_results)}개 오브젝트")
    print(f"{'='*60}")

    for i, (cd, obj_id, syn, split) in enumerate(all_results[:args.top_k]):
        print(f"  #{i+1}  CD={cd:.6f}  ID={obj_id}  synset={syn}  split={split}")

    # 가장 유사한 GT의 ShapeNet 경로
    best_cd, best_id, best_syn, best_split = all_results[0]
    print(f"\n가장 유사한 GT:")
    print(f"  ShapeNet ID: {best_syn}/{best_id}")
    print(f"  Chamfer Distance: {best_cd:.6f}")
    print(f"  split: {best_split}")


if __name__ == "__main__":
    main()
