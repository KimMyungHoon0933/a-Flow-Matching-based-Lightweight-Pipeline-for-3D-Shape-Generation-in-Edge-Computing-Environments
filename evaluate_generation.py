"""
evaluate_generation.py — Flow vs Diffusion 생성 품질 평가
============================================================
수행 순서:
  1. Flow/Diffusion 모델로 카테고리별 N개 메쉬 생성 (.obj 저장)
  2. test.h5에서 해당 카테고리 GT surface 포인트 추출
  3. 생성 메쉬에서 포인트 샘플링 (trimesh)
  4. pairwise CD/EMD 계산 → MMD, COV, 1-NNA 집계

사용법:
  # 1단계: 메쉬 생성 (이미 generate_flow.py로 생성했다면 스킵)
  python evaluate_generation.py \
    --mode generate \
    --method flow \
    --vae_config /path/to/config.yaml \
    --vae_weights /path/to/weights.pt \
    --fm_pth /path/to/flow_checkpoint.ckpt \
    --categories chair airplane table \
    --num_samples 200 \
    --output_dir ./eval_meshes/flow

  # 2단계: 메트릭 계산
  python evaluate_generation.py \
    --mode evaluate \
    --mesh_dir ./eval_meshes/flow \
    --test_h5 /data/kimm0902_files/datasets/shapenet/test.h5 \
    --categories chair airplane table \
    --num_points 2048
============================================================
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
import h5py
from tqdm import tqdm

# ─────────────────────────────────────────────────────────
# 카테고리 매핑 (COD-VAE shapenet.py 기준)
# ─────────────────────────────────────────────────────────

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

CATEGORY_NAME_TO_ID = {
    "airplane": 0, "car": 16, "chair": 18, "table": 49,
    "watercraft": 53, "sofa": 47, "rifle": 44, "lamp": 28
}

CATEGORY_ID_TO_SYNSET = {v: SHAPENET_CATEGORY_IDS[v] for v in CATEGORY_NAME_TO_ID.values()}
CATEGORY_NAME_TO_SYNSET = {k: CATEGORY_ID_TO_SYNSET[v] for k, v in CATEGORY_NAME_TO_ID.items()}


# ─────────────────────────────────────────────────────────
# 메트릭 함수들 (evaluation_metrics(nna).py에서 가져옴)
# ─────────────────────────────────────────────────────────

def distChamfer(a, b):
    """PyTorch 기반 Chamfer Distance"""
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def _pairwise_CD(sample_pcs, ref_pcs, batch_size):
    """sample_pcs와 ref_pcs 사이의 pairwise CD 행렬 계산"""
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []

    for i in tqdm(range(N_sample), desc="Pairwise CD"):
        cd_lst = []
        sample_i = sample_pcs[i]

        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_exp = sample_i.view(1, -1, 3).expand(batch_size_ref, -1, -1).contiguous()

            dl, dr = distChamfer(sample_exp, ref_batch)
            cd = (dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1)
            cd_lst.append(cd)

        cd_lst = torch.cat(cd_lst, dim=1)
        all_cd.append(cd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # [N_sample, N_ref]
    return all_cd


def lgan_mmd_cov(all_dist):
    """MMD와 COV 계산"""
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'MMD': mmd,
        'COV': cov,
    }


def knn(Mxx, Mxy, Myy, k, sqrt=False):
    """1-NNA 계산"""
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }
    s.update({
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def compute_metrics_cd(sample_pcs, ref_pcs, batch_size=32):
    """
    MMD-CD, COV-CD, 1-NNA-CD를 한번에 계산.
    sample_pcs: [N_gen, num_points, 3] 생성된 포인트 클라우드
    ref_pcs:    [N_ref, num_points, 3] GT 포인트 클라우드
    """
    print(f"\n{'='*50}")
    print(f"  메트릭 계산 시작")
    print(f"  생성 샘플: {sample_pcs.shape[0]}개, GT 샘플: {ref_pcs.shape[0]}개")
    print(f"  포인트 수: {sample_pcs.shape[1]}")
    print(f"{'='*50}")

    # Pairwise CD: ref vs sample
    print("\n[1/3] ref vs sample pairwise CD 계산 중...")
    M_rs_cd = _pairwise_CD(ref_pcs, sample_pcs, batch_size)

    # MMD, COV
    print("[2/3] MMD, COV 계산 중...")
    res_cd = lgan_mmd_cov(M_rs_cd.t())

    # 1-NNA: ref-ref, sample-sample도 필요
    print("[3/3] 1-NNA 계산 중...")
    print("  ref vs ref pairwise CD...")
    M_rr_cd = _pairwise_CD(ref_pcs, ref_pcs, batch_size)
    print("  sample vs sample pairwise CD...")
    M_ss_cd = _pairwise_CD(sample_pcs, sample_pcs, batch_size)

    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)

    results = {
        'MMD-CD': res_cd['MMD'].item(),
        'COV-CD': res_cd['COV'].item() * 100,  # 퍼센트로
        '1-NNA-CD': one_nn_cd_res['acc'].item() * 100,  # 퍼센트로
    }

    return results


# ─────────────────────────────────────────────────────────
# GT 포인트 클라우드 추출 (test.h5)
# ─────────────────────────────────────────────────────────

def load_gt_point_clouds(test_h5_path, category_synset, num_points=2048):
    """test.h5에서 특정 카테고리의 GT surface 포인트 클라우드 로드"""
    gt_pcs = []

    with h5py.File(test_h5_path, 'r') as f:
        if category_synset not in f:
            print(f"  경고: 카테고리 {category_synset}가 test.h5에 없음")
            return None

        category_group = f[category_synset]
        object_ids = list(category_group.keys())
        print(f"  GT 오브젝트 수: {len(object_ids)}")

        for obj_id in tqdm(object_ids, desc=f"GT 로딩 ({category_synset})"):
            obj_group = category_group[obj_id]

            if 'surface_points' in obj_group:
                surface = obj_group['surface_points'][:]
                scale = float(obj_group.attrs.get('scale', 1.0))
                surface = surface * scale
            else:
                continue

            # num_points개 샘플링
            if surface.shape[0] >= num_points:
                idx = np.random.choice(surface.shape[0], num_points, replace=False)
            else:
                idx = np.random.choice(surface.shape[0], num_points, replace=True)
            pc = surface[idx].astype(np.float32)
            gt_pcs.append(pc)

    if len(gt_pcs) == 0:
        return None

    gt_pcs = np.stack(gt_pcs, axis=0)  # [N, num_points, 3]
    return gt_pcs


# ─────────────────────────────────────────────────────────
# 생성 메쉬에서 포인트 클라우드 추출
# ─────────────────────────────────────────────────────────

def load_generated_point_clouds(mesh_dir, num_points=2048):
    """생성된 .obj 파일들에서 포인트 클라우드 샘플링"""
    gen_pcs = []
    obj_files = sorted([f for f in os.listdir(mesh_dir) if f.endswith('.obj')])

    if len(obj_files) == 0:
        print(f"  경고: {mesh_dir}에 .obj 파일이 없음")
        return None

    print(f"  생성 메쉬 수: {len(obj_files)}")

    for obj_file in tqdm(obj_files, desc="생성 메쉬 로딩"):
        mesh_path = os.path.join(mesh_dir, obj_file)
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
                print(f"  스킵 (빈 메쉬): {obj_file}")
                continue
            pc = mesh.sample(num_points).astype(np.float32)
            gen_pcs.append(pc)
        except Exception as e:
            print(f"  스킵 (에러): {obj_file} - {e}")
            continue

    if len(gen_pcs) == 0:
        return None

    gen_pcs = np.stack(gen_pcs, axis=0)
    return gen_pcs


# ─────────────────────────────────────────────────────────
# 메쉬 생성 (Flow)
# ─────────────────────────────────────────────────────────

def generate_meshes_flow(vae_config, vae_weights, fm_pth, category_name, category_id,
                         num_samples, output_dir, steps=4, density=128):
    """Flow Matching으로 메쉬 대량 생성"""
    sys.path.append(os.path.abspath('./CFM'))
    from CFM.flowmating_lightning import FlowMatchingLightning
    from flow_matching.solver.ode_solver import ODESolver
    from cod.utils.recon import create_grid_queries
    import mcubes
    import gc

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    model = FlowMatchingLightning(vae_config_path=vae_config, vae_weights_path=vae_weights)
    ckpt = torch.load(fm_pth, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device).eval()

    n_latents = model.flow_model.n_latents
    channels = model.flow_model.channels
    step_size = 1.0 / steps

    class FlowWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m
        def forward(self, x, t, **extras):
            if t.dim() == 0:
                t = t.repeat(x.shape[0])
            return self.model(x, t, class_labels=extras.get('class_labels'))

    wrapped = FlowWrapper(model.flow_model)
    solver = ODESolver(velocity_model=wrapped)

    queries = create_grid_queries(density).to(device)
    gap = 2. / density
    cat_dir = os.path.join(output_dir, category_name)
    os.makedirs(cat_dir, exist_ok=True)

    batch_size = min(num_samples, 256)
    generated = 0

    print(f"  생성 시작: {category_name} (ID={category_id}), 총 {num_samples}개")

    with torch.no_grad():
        while generated < num_samples:
            current_batch = min(batch_size, num_samples - generated)
            x_init = torch.randn([current_batch, n_latents, channels], device=device)
            labels = torch.full((current_batch,), category_id, dtype=torch.long, device=device)

            sampled_z = solver.sample(
                x_init=x_init, step_size=step_size,
                method='euler', class_labels=labels
            ).float().contiguous()

            for j in range(current_batch):
                z = sampled_z[j:j+1].view(1, n_latents, channels)
                decoded = model.vae.decode_latents(z)
                context, _, _, _ = model.vae.decode_embed(decoded)

                logits = []
                chunk_size = 100000
                for k in range(0, queries.size(1), chunk_size):
                    chunk = queries[:, k:k+chunk_size]
                    logit = model.vae.decode_queries(context, chunk)
                    logits.append(logit)
                logits = torch.cat(logits, dim=1)

                volume = logits.view(density, density, density).permute(1, 0, 2).cpu().numpy()
                verts, faces = mcubes.marching_cubes(volume, 0)
                mc_gap = 2. / (density - 1)
                verts *= mc_gap
                verts -= 1

                if verts.shape[0] > 0 and faces.shape[0] > 0:
                    mesh = trimesh.Trimesh(verts, faces)
                    mesh.export(os.path.join(cat_dir, f"{generated:05d}.obj"))

                generated += 1
                if generated % 10 == 0:
                    print(f"    {generated}/{num_samples} 완료")

    # 메모리 해제
    del model, wrapped, solver
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  완료: {cat_dir}에 {generated}개 생성됨")


# ─────────────────────────────────────────────────────────
# 메쉬 생성 (Diffusion / EDM)
# ─────────────────────────────────────────────────────────

def generate_meshes_diffusion(vae_config, vae_weights, diff_pth, category_name, category_id,
                              num_samples, output_dir, density=128):
    """EDM Diffusion으로 메쉬 대량 생성"""
    from duffusion.diffusion_lightning import CategoryConditionedDiffusion
    from cod.utils.recon import create_grid_queries
    import mcubes
    import gc

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    print(f"  디퓨전 모델 로딩: {diff_pth}")
    model = CategoryConditionedDiffusion.load_from_checkpoint(
        checkpoint_path=diff_pth,
        vae_config_path=vae_config,
        vae_weights_path=vae_weights,
        map_location='cpu'
    )
    model.to(device).eval()

    queries = create_grid_queries(density).to(device)
    gap = 2. / density
    cat_dir = os.path.join(output_dir, category_name)
    os.makedirs(cat_dir, exist_ok=True)

    generated = 0
    print(f"  생성 시작: {category_name} (ID={category_id}), 총 {num_samples}개")

    with torch.no_grad():
        for i in range(num_samples):
            cond_tensor = torch.tensor([category_id], dtype=torch.long, device=device)
            batch_seeds = torch.randint(0, 1000000, (1,)).to(device)

            sampled_z = model.diffusion_model.sample(
                cond=cond_tensor, batch_seeds=batch_seeds
            ).float().contiguous()

            # VAE 디코딩
            decoded = model.vae.decode_latents(sampled_z)
            context, _, _, _ = model.vae.decode_embed(decoded)

            logits = []
            chunk_size = 100000
            for k in range(0, queries.size(1), chunk_size):
                chunk = queries[:, k:k + chunk_size]
                logit = model.vae.decode_queries(context, chunk)
                logits.append(logit)
            logits = torch.cat(logits, dim=1)

            volume = logits.view(density, density, density).permute(1, 0, 2).cpu().numpy()
            verts, faces = mcubes.marching_cubes(volume, 0)
            mc_gap = 2. / (density - 1)
            verts *= mc_gap
            verts -= 1

            if verts.shape[0] > 0 and faces.shape[0] > 0:
                mesh = trimesh.Trimesh(verts, faces)
                mesh.export(os.path.join(cat_dir, f"{generated:05d}.obj"))

            generated += 1
            if generated % 10 == 0:
                print(f"    {generated}/{num_samples} 완료")

    # 메모리 해제
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  완료: {cat_dir}에 {generated}개 생성됨")


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser("3D 생성 품질 평가")

    parser.add_argument('--mode', type=str, required=True, choices=['generate', 'evaluate', 'both'],
                        help='generate: 메쉬 생성, evaluate: 메트릭 계산, both: 둘 다')

    # 생성 관련
    parser.add_argument('--method', type=str, default='flow', choices=['flow', 'diffusion'])
    parser.add_argument('--vae_config', type=str, default=None)
    parser.add_argument('--vae_weights', type=str, default=None)
    parser.add_argument('--fm_pth', type=str, default=None, help='Flow 체크포인트')
    parser.add_argument('--diff_pth', type=str, default=None, help='Diffusion 체크포인트')
    parser.add_argument('--num_samples', type=int, default=500, help='카테고리당 생성 수 (GT 테스트셋 수 이상 권장, COD-VAE는 2000)')
    parser.add_argument('--steps', type=int, default=18, help='Flow sampling 스텝')
    parser.add_argument('--output_dir', type=str, default='./eval_meshes/flow_v7')

    # 평가 관련
    parser.add_argument('--mesh_dir', type=str, default=None, help='생성된 메쉬 디렉토리')
    parser.add_argument('--test_h5', type=str, default='/data/kimm0902_files/datasets/shapenet/test.h5')
    parser.add_argument('--categories', nargs='+', default=['airplane', 'chair', 'table','car','rifle'],
                        help='평가할 카테고리 (예: airplane chair table)')
    parser.add_argument('--num_points', type=int, default=2048, help='포인트 클라우드 샘플링 수')
    parser.add_argument('--batch_size', type=int, default=512, help='pairwise CD 계산 배치 크기')
    parser.add_argument('--device', type=str, default='cuda:0')

    return parser.parse_args()


def main():
    args = get_args()

    categories = args.categories

    # ── 생성 단계 ──
    if args.mode in ['generate', 'both']:
        print("\n" + "█" * 60)
        print("  1단계: 메쉬 생성")
        print("█" * 60)

        # GT 수를 미리 확인해서 num_samples를 자동 조정
        gt_counts = {}
        if args.test_h5 and os.path.exists(args.test_h5):
            with h5py.File(args.test_h5, 'r') as f:
                for cat_name in categories:
                    cat_synset = CATEGORY_NAME_TO_SYNSET.get(cat_name)
                    if cat_synset and cat_synset in f:
                        gt_counts[cat_name] = len(f[cat_synset].keys())

        for cat_name in categories:
            cat_id = CATEGORY_NAME_TO_ID[cat_name]

            # GT 수 이상으로 생성 (최소 GT 수, 사용자 지정이 더 크면 그것 사용)
            gt_n = gt_counts.get(cat_name, 0)
            num_to_generate = max(args.num_samples, gt_n)
            print(f"\n▶ 카테고리: {cat_name} (ID={cat_id}), GT={gt_n}개, 생성={num_to_generate}개")

            if args.method == 'flow':
                if args.fm_pth is None:
                    raise ValueError("--fm_pth를 지정해야 합니다 (Flow 체크포인트 경로)")
                generate_meshes_flow(
                    args.vae_config, args.vae_weights, args.fm_pth,
                    cat_name, cat_id, num_to_generate, args.output_dir, args.steps
                )
            elif args.method == 'diffusion':
                if args.diff_pth is None:
                    raise ValueError("--diff_pth를 지정해야 합니다 (Diffusion 체크포인트 경로)")
                generate_meshes_diffusion(
                    args.vae_config, args.vae_weights, args.diff_pth,
                    cat_name, cat_id, num_to_generate, args.output_dir
                )

    # ── 평가 단계 ──
    if args.mode in ['evaluate', 'both']:
        print("\n" + "█" * 60)
        print("  2단계: 메트릭 계산")
        print("█" * 60)

        mesh_dir = args.mesh_dir if args.mesh_dir else args.output_dir
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        all_results = {}

        for cat_name in categories:
            cat_synset = CATEGORY_NAME_TO_SYNSET[cat_name]
            cat_mesh_dir = os.path.join(mesh_dir, cat_name)

            print(f"\n▶ 카테고리: {cat_name} (synset={cat_synset})")

            # GT 포인트 클라우드 로드
            print("  GT 포인트 클라우드 로딩...")
            gt_pcs = load_gt_point_clouds(args.test_h5, cat_synset, args.num_points)
            if gt_pcs is None:
                print(f"  건너뜀: GT 데이터 없음")
                continue

            # 생성 메쉬에서 포인트 클라우드 추출
            print("  생성 메쉬에서 포인트 추출...")
            gen_pcs = load_generated_point_clouds(cat_mesh_dir, args.num_points)
            if gen_pcs is None:
                print(f"  건너뜀: 생성 메쉬 없음")
                continue

            # 정규화 없이 사용 (데이터가 이미 [-1, 1] 범위)

            # [디버그] 좌표 범위 출력
            print(f"  GT  범위: [{gt_pcs.min():.3f}, {gt_pcs.max():.3f}], shape={gt_pcs.shape}")
            print(f"  Gen 범위: [{gen_pcs.min():.3f}, {gen_pcs.max():.3f}], shape={gen_pcs.shape}")

            # [수정] COV/MMD 프로토콜: 생성 수 >= GT 수 보장
            # COD-VAE 논문: COV/MMD는 |Sg| = 5*|Sr|, 1-NNA는 |Sg| = |Sr|
            # 생성 수가 부족하면 경고
            n_gt = gt_pcs.shape[0]
            n_gen = gen_pcs.shape[0]
            if n_gen < n_gt:
                print(f"  ⚠️  경고: 생성 수({n_gen}) < GT 수({n_gt}). COV가 낮게 나올 수 있음.")
                print(f"     권장: --num_samples를 {n_gt} 이상으로 설정")

            # [디버그] 단일 쌍 CD 확인
            test_dl, test_dr = distChamfer(
                torch.from_numpy(gen_pcs[0:1]).float().to(device),
                torch.from_numpy(gt_pcs[0:1]).float().to(device)
            )
            test_cd = (test_dl.mean() + test_dr.mean()).item()
            print(f"  [디버그] 샘플 0 vs GT 0 CD = {test_cd:.6f}")

            # 텐서 변환
            gt_tensor = torch.from_numpy(gt_pcs).float().to(device)
            gen_tensor = torch.from_numpy(gen_pcs).float().to(device)

            # 메트릭 계산
            results = compute_metrics_cd(gen_tensor, gt_tensor, batch_size=args.batch_size)
            all_results[cat_name] = results

            print(f"\n  ┌──────────────────────────────────┐")
            print(f"  │ {cat_name:^32s} │")
            print(f"  ├──────────────────────────────────┤")
            print(f"  │ MMD-CD:   {results['MMD-CD']:.6f}             │")
            print(f"  │ COV-CD:   {results['COV-CD']:.2f}%               │")
            print(f"  │ 1-NNA-CD: {results['1-NNA-CD']:.2f}%               │")
            print(f"  └──────────────────────────────────┘")

        # 평균 결과
        if len(all_results) > 0:
            avg_mmd = np.mean([v['MMD-CD'] for v in all_results.values()])
            avg_cov = np.mean([v['COV-CD'] for v in all_results.values()])
            avg_nna = np.mean([v['1-NNA-CD'] for v in all_results.values()])

            print(f"\n{'='*50}")
            print(f"  전체 평균 (카테고리 {len(all_results)}개)")
            print(f"  MMD-CD:   {avg_mmd:.6f}")
            print(f"  COV-CD:   {avg_cov:.2f}%")
            print(f"  1-NNA-CD: {avg_nna:.2f}%")
            print(f"{'='*50}")

            # 결과 저장
            save_path = os.path.join(mesh_dir, 'evaluation_results.json')
            save_data = {
                'per_category': {k: {mk: float(mv) for mk, mv in v.items()} for k, v in all_results.items()},
                'average': {
                    'MMD-CD': float(avg_mmd),
                    'COV-CD': float(avg_cov),
                    '1-NNA-CD': float(avg_nna),
                }
            }
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"  결과 저장: {save_path}")


if __name__ == "__main__":
    main()