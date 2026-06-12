"""
reconstruct_gt.py — h5에서 특정 오브젝트를 VAE로 복원하여 mesh로 저장
==================================================================
사용법:
  python reconstruct_gt.py \
    --h5 /data/kimm0902_files/datasets/shapenet/train.h5 \
    --synset 03001627 \
    --object_id 54c9f96ffc35d0c2eec2ef73f04d4ff7 \
    --vae_config "/root/re/COD-VAE/vae_m32 copy/config.yaml" \
    --vae_weights "/root/re/COD-VAE/vae_m32 copy/weights.pt" \
    --output_gt gt_original.obj \
    --output_recon gt_reconstructed.obj
==================================================================
"""

import argparse
import numpy as np
import torch
import h5py
import mcubes
import trimesh
from omegaconf import OmegaConf

from cod.models.vae.vae import CompactLatentVAE
from cod.utils.recon import create_grid_queries


def load_vae(config_path, weights_path, device):
    cfg = OmegaConf.load(config_path)
    vae_cfg = cfg.get('model', cfg)
    exclude = ['_target_', '_base_', '_overwrite_']
    params = {k: v for k, v in vae_cfg.items() if k not in exclude}

    vae = CompactLatentVAE(**params)
    ckpt = torch.load(weights_path, map_location='cpu')
    state = ckpt.get('model', ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))
    state = {(k.replace('model.', '', 1) if k.startswith('model.') else k): v for k, v in state.items()}
    vae.load_state_dict(state, strict=False)
    vae.to(device).eval()
    vae.requires_grad_(False)
    return vae


def main():
    parser = argparse.ArgumentParser("GT 오브젝트 VAE 복원")
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--synset", type=str, required=True, help="예: 03001627")
    parser.add_argument("--object_id", type=str, required=True)
    parser.add_argument("--vae_config", type=str, required=True)
    parser.add_argument("--vae_weights", type=str, required=True)
    parser.add_argument("--output_gt", type=str, default="gt_original.obj")
    parser.add_argument("--output_recon", type=str, default="gt_reconstructed.obj")
    parser.add_argument("--density", type=int, default=128)
    parser.add_argument("--pc_size", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ── 1. h5에서 surface points 로드 ──
    print(f"h5 로딩: {args.h5}")
    with h5py.File(args.h5, 'r') as f:
        key = f"{args.synset}/{args.object_id}"
        if key not in f:
            raise ValueError(f"오브젝트를 찾을 수 없음: {key}")

        group = f[key]
        surface = group['surface_points'][:]
        scale = float(group.attrs.get('scale', 1.0))
        surface = surface * scale

    print(f"  surface points: {surface.shape}, scale={scale:.4f}")
    print(f"  범위: [{surface.min():.3f}, {surface.max():.3f}]")

    # ── 2. GT 포인트 클라우드를 mesh로 저장 ──
    gt_pc = trimesh.PointCloud(surface)
    gt_pc.export(args.output_gt.replace('.obj', '_pointcloud.ply'))
    print(f"  GT 포인트 클라우드 저장: {args.output_gt.replace('.obj', '_pointcloud.ply')}")

    # ── 3. VAE 로드 ──
    print(f"VAE 로딩...")
    vae = load_vae(args.vae_config, args.vae_weights, device)

    # ── 4. surface points 샘플링 → VAE encode → decode → mesh ──
    # pc_size개 샘플링
    if surface.shape[0] >= args.pc_size:
        idx = np.random.choice(surface.shape[0], args.pc_size, replace=False)
    else:
        idx = np.random.choice(surface.shape[0], args.pc_size, replace=True)
    pc = torch.from_numpy(surface[idx].astype(np.float32)).unsqueeze(0).to(device)
    print(f"  입력 포인트: {pc.shape}")

    with torch.no_grad():
        # Encode
        z_embed = vae.encode_embed(pc)
        z_latent, posterior = vae.encode_latents(z_embed)
        print(f"  latent shape: {z_latent.shape}")
        print(f"  latent 통계: mean={z_latent.mean():.4f}, std={z_latent.std():.4f}")

        # Decode
        decoded = vae.decode_latents(z_latent)
        context, _, _, _ = vae.decode_embed(decoded)

        # Query grid
        queries = create_grid_queries(args.density).to(device)
        logits = []
        chunk_size = 100000
        for i in range(0, queries.size(1), chunk_size):
            chunk = queries[:, i:i + chunk_size]
            logit = vae.decode_queries(context, chunk)
            logits.append(logit)
        logits = torch.cat(logits, dim=1)

    # ── 5. Marching Cubes → mesh 저장 ──
    density = args.density
    gap = 2. / (density - 1)
    volume = logits.view(density, density, density).permute(1, 0, 2).cpu().numpy()
    verts, faces = mcubes.marching_cubes(volume, 0)
    verts *= gap
    verts -= 1

    mesh = trimesh.Trimesh(verts, faces)
    mesh.export(args.output_recon)

    print(f"\n결과:")
    print(f"  GT 포인트 클라우드: {args.output_gt.replace('.obj', '_pointcloud.ply')}")
    print(f"  VAE 복원 메쉬:     {args.output_recon}")
    print(f"  vertices={verts.shape[0]}, faces={faces.shape[0]}")


if __name__ == "__main__":
    main()
