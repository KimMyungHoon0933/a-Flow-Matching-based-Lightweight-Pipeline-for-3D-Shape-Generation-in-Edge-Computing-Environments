"""
benchmark_10runs.py — Diffusion vs Flow 10회 평균 벤치마크
===========================================================
사용법:
  python benchmark_10runs.py \
    --diffusion_ckpt <디퓨전 체크포인트> \
    --fm_pth <Flow 체크포인트> \
    --vae_config <VAE config> \
    --vae_weights <VAE 가중치> \
    --runs 10
===========================================================
"""

import argparse
import torch
import torch.nn.functional
import numpy as np
import sys
import os
import gc
import time
import mcubes
import trimesh

# ==========================================
# 환경 변수 설정
# ==========================================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKLDNN_DISABLED'] = '1'

torch.backends.mkldnn.enabled = False
if hasattr(torch.backends, 'xnnpack'):
    torch.backends.xnnpack.enabled = False
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)

# einsum 패치
def safe_einsum_linear(input, weight, bias=None):
    orig_shape = input.shape
    x = input.reshape(-1, orig_shape[-1]).float()
    w = weight.float()
    out = torch.einsum('xi,ji->xj', x, w)
    if bias is not None:
        out = out + bias.float()
    return out.reshape(*orig_shape[:-1], out.shape[-1])

torch.nn.functional.linear = safe_einsum_linear

sys.path.append(os.path.abspath('./CFM'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'external/pointops')))

from cod.utils.recon import create_grid_queries

CATEGORY_ID = 18  # chair

def get_args():
    parser = argparse.ArgumentParser("Diffusion vs Flow 10회 벤치마크")
    parser.add_argument("--diffusion_ckpt", type=str, required=True)
    parser.add_argument("--fm_pth", type=str, required=True)
    parser.add_argument("--vae_config", type=str, required=True)
    parser.add_argument("--vae_weights", type=str, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--density", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4, help="Flow ODE solver step 수")
    return parser.parse_args()


def benchmark_diffusion(args):
    """Diffusion (EDM 18-step) 벤치마크"""
    from duffusion.diffusion_lightning import CategoryConditionedDiffusion

    print("\n" + "=" * 50)
    print(f"Diffusion (EDM) 벤치마크 — {args.runs}회")
    print("=" * 50)

    device = torch.device('cpu')

    # 모델 로드
    print("모델 로딩 중...")
    model = CategoryConditionedDiffusion.load_from_checkpoint(
        checkpoint_path=args.diffusion_ckpt,
        vae_config_path=args.vae_config,
        vae_weights_path=args.vae_weights,
        map_location=device
    )
    model.to(device).eval().to(torch.float32)

    cond = torch.tensor([CATEGORY_ID], dtype=torch.long, device=device)
    queries = create_grid_queries(args.density).to(device)

    sampling_times = []
    vae_times = []
    mc_times = []
    total_times = []

    for i in range(args.runs):
        gc.collect()
        total_start = time.time()

        with torch.no_grad():
            # 샘플링
            t0 = time.time()
            batch_seeds = torch.randint(0, 1000000, (1,), device=device)
            sampled_z = model.diffusion_model.sample(cond=cond, batch_seeds=batch_seeds)
            sampled_z = sampled_z.to(torch.float32).clone().contiguous()
            t_sampling = time.time() - t0

            # VAE 디코딩
            t0 = time.time()
            decoded = model.vae.decode_latents(sampled_z)
            context, _, _, _ = model.vae.decode_embed(decoded)
            logits = []
            for j in range(0, queries.size(1), 50000):
                chunk = queries[:, j:j+50000]
                logit = model.vae.decode_queries(context, chunk)
                logits.append(logit)
            logits = torch.cat(logits, dim=1)
            t_vae = time.time() - t0

        # Marching Cubes
        t0 = time.time()
        density = args.density
        gap = 2. / density
        volume = logits.view(density, density, density).cpu().numpy()
        verts, faces = mcubes.marching_cubes(volume, 0)
        t_mc = time.time() - t0

        t_total = time.time() - total_start

        sampling_times.append(t_sampling)
        vae_times.append(t_vae)
        mc_times.append(t_mc)
        total_times.append(t_total)

        print(f"  [{i+1}/{args.runs}] 샘플링={t_sampling:.2f}s  VAE={t_vae:.2f}s  MC={t_mc:.2f}s  전체={t_total:.2f}s")

    del model
    gc.collect()

    return {
        'sampling': sampling_times,
        'vae': vae_times,
        'mc': mc_times,
        'total': total_times,
    }


def benchmark_flow(args):
    """Flow Matching (OT-CFM 4-step) 벤치마크"""
    from CFM.flowmating_lightning import FlowMatchingLightning
    from flow_matching.solver.ode_solver import ODESolver

    print("\n" + "=" * 50)
    print(f"Flow Matching (OT-CFM 4-step) 벤치마크 — {args.runs}회")
    print("=" * 50)

    device = torch.device('cpu')

    # 모델 로드
    print("모델 로딩 중...")
    model = FlowMatchingLightning(
        vae_config_path=args.vae_config,
        vae_weights_path=args.vae_weights
    )
    ckpt = torch.load(args.fm_pth, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.to(device).eval().to(torch.float32)

    n_latents = model.flow_model.n_latents
    channels = model.flow_model.channels

    class FlowWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m
        def forward(self, x, t, **extras):
            if t.dim() == 0:
                t = t.repeat(x.shape[0])
            return self.model(x, t, class_labels=extras.get('class_labels'))

    cond = torch.tensor([CATEGORY_ID], dtype=torch.long, device=device)
    queries = create_grid_queries(args.density).to(device)

    sampling_times = []
    vae_times = []
    mc_times = []
    total_times = []

    step_size = 1.0 / args.steps
    wrapped = FlowWrapper(model.flow_model)
    solver = ODESolver(velocity_model=wrapped)

    for i in range(args.runs):
        gc.collect()
        total_start = time.time()

        with torch.no_grad():
            # 샘플링
            t0 = time.time()
            x_init = torch.randn([1, n_latents, channels], device=device)
            sampled_z = solver.sample(
                x_init=x_init,
                step_size=step_size,
                method='euler',
                class_labels=cond
            )
            sampled_z = sampled_z.to(torch.float32).clone().contiguous()
            t_sampling = time.time() - t0

            # VAE 디코딩
            t0 = time.time()
            sampled_z = sampled_z.view(1, n_latents, channels)
            decoded = model.vae.decode_latents(sampled_z)
            context, _, _, _ = model.vae.decode_embed(decoded)
            logits = []
            chunk_size = 100000
            for j in range(0, queries.size(1), chunk_size):
                chunk = queries[:, j:j+chunk_size]
                logit = model.vae.decode_queries(context, chunk)
                logits.append(logit)
            logits = torch.cat(logits, dim=1)
            t_vae = time.time() - t0

        # Marching Cubes
        t0 = time.time()
        density = args.density
        volume = logits.view(density, density, density).permute(1, 0, 2).cpu().numpy()
        verts, faces = mcubes.marching_cubes(volume, 0)
        mc_gap = 2. / (density - 1)
        verts *= mc_gap
        verts -= 1
        t_mc = time.time() - t0

        t_total = time.time() - total_start

        sampling_times.append(t_sampling)
        vae_times.append(t_vae)
        mc_times.append(t_mc)
        total_times.append(t_total)

        print(f"  [{i+1}/{args.runs}] 샘플링={t_sampling:.2f}s  VAE={t_vae:.2f}s  MC={t_mc:.2f}s  전체={t_total:.2f}s")

    del model, wrapped, solver
    gc.collect()

    return {
        'sampling': sampling_times,
        'vae': vae_times,
        'mc': mc_times,
        'total': total_times,
    }


def print_results(diff_results, flow_results, runs):
    """결과 정리 및 출력"""

    def avg(lst):
        return sum(lst) / len(lst)

    def std(lst):
        m = avg(lst)
        return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5

    print("\n")
    print("=" * 70)
    print(f"벤치마크 결과 ({runs}회 평균)")
    print("=" * 70)
    print(f"{'단계':<20} {'Diffusion (18-step)':<25} {'Flow (4-step)':<25} {'가속비':<10}")
    print("-" * 70)

    for stage, name in [('sampling', 'Noise Sampling'), ('vae', 'VAE Decoding'), ('mc', 'Marching Cubes'), ('total', 'Total')]:
        d_avg = avg(diff_results[stage])
        d_std = std(diff_results[stage])
        f_avg = avg(flow_results[stage])
        f_std = std(flow_results[stage])
        ratio = d_avg / f_avg if f_avg > 0 else 0

        if stage == 'total':
            print("-" * 70)

        print(f"{name:<20} {d_avg:.2f}s (±{d_std:.2f})       {f_avg:.2f}s (±{f_std:.2f})       {ratio:.1f}x")

    print("=" * 70)

    # 파일로도 저장
    with open("benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write(f"벤치마크 결과 ({runs}회 평균)\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'단계':<20} {'Diffusion (18-step)':<25} {'Flow (4-step)':<25} {'가속비':<10}\n")
        f.write("-" * 70 + "\n")

        for stage, name in [('sampling', 'Noise Sampling'), ('vae', 'VAE Decoding'), ('mc', 'Marching Cubes'), ('total', 'Total')]:
            d_avg = avg(diff_results[stage])
            d_std = std(diff_results[stage])
            f_avg = avg(flow_results[stage])
            f_std = std(flow_results[stage])
            ratio = d_avg / f_avg if f_avg > 0 else 0
            if stage == 'total':
                f.write("-" * 70 + "\n")
            f.write(f"{name:<20} {d_avg:.2f}s (±{d_std:.2f})       {f_avg:.2f}s (±{f_std:.2f})       {ratio:.1f}x\n")

        f.write("=" * 70 + "\n\n")

        # 개별 결과도 기록
        f.write("개별 실행 결과:\n\n")
        f.write("Diffusion:\n")
        for i, (s, v, m, t) in enumerate(zip(diff_results['sampling'], diff_results['vae'], diff_results['mc'], diff_results['total'])):
            f.write(f"  [{i+1}] 샘플링={s:.2f}s  VAE={v:.2f}s  MC={m:.2f}s  전체={t:.2f}s\n")

        f.write("\nFlow:\n")
        for i, (s, v, m, t) in enumerate(zip(flow_results['sampling'], flow_results['vae'], flow_results['mc'], flow_results['total'])):
            f.write(f"  [{i+1}] 샘플링={s:.2f}s  VAE={v:.2f}s  MC={m:.2f}s  전체={t:.2f}s\n")

    print("\n결과가 benchmark_results.txt에 저장되었습니다.")


if __name__ == "__main__":
    args = get_args()

    print("=" * 50)
    print(f"Diffusion vs Flow 벤치마크 ({args.runs}회)")
    print(f"카테고리: chair (ID={CATEGORY_ID})")
    print(f"density: {args.density}")
    print("=" * 50)

    # Diffusion 먼저
    diff_results = benchmark_diffusion(args)

    # Flow
    flow_results = benchmark_flow(args)

    # 결과 정리
    print_results(diff_results, flow_results, args.runs)